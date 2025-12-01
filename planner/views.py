"""
API views for the cutting optimization planner.
"""

from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.views import APIView
from django.utils import timezone
from django.http import FileResponse, Http404
from django.shortcuts import get_object_or_404
import os

from .models import (
    StockBlock,
    PartSpecification,
    CuttingJob,
    Configuration,
    ConfigurationSet
)
from .serializers import (
    StockBlockSerializer,
    PartSpecificationSerializer,
    CuttingJobSerializer,
    CuttingJobCreateSerializer,
    ConfigurationSerializer,
    ConfigurationSetSerializer,
    Top3ConfigurationsRequestSerializer,
    Top3ConfigurationsResponseSerializer,
)
from .services import get_cutting_service

# Import the new API wrapper
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pack_manually_api import compute_top3_approaches


class StockBlockViewSet(viewsets.ModelViewSet):
    """
    ViewSet for StockBlock model.

    Provides CRUD operations for stock blocks.
    """
    queryset = StockBlock.objects.all()
    serializer_class = StockBlockSerializer
    filterset_fields = ['material_type']
    search_fields = ['name', 'material_type']
    ordering_fields = ['created_at', 'name', 'volume']
    ordering = ['-created_at']


class PartSpecificationViewSet(viewsets.ModelViewSet):
    """
    ViewSet for PartSpecification model.

    Provides CRUD operations for part specifications.
    """
    queryset = PartSpecification.objects.all()
    serializer_class = PartSpecificationSerializer
    filterset_fields = ['thickness']
    search_fields = ['name']
    ordering_fields = ['created_at', 'name', 'volume']
    ordering = ['name']


class CuttingJobViewSet(viewsets.ModelViewSet):
    """
    ViewSet for CuttingJob model.

    Provides:
    - POST /api/jobs/ - Create and run a cutting job
    - GET /api/jobs/{id}/ - Get job status and results
    - GET /api/jobs/ - List all jobs
    """
    queryset = CuttingJob.objects.all()
    serializer_class = CuttingJobSerializer
    filterset_fields = ['status']
    ordering_fields = ['created_at', 'completed_at']
    ordering = ['-created_at']

    def get_serializer_class(self):
        """Use different serializers for create vs read."""
        if self.action == 'create':
            return CuttingJobCreateSerializer
        return CuttingJobSerializer

    def create(self, request, *args, **kwargs):
        """
        Create and run a new cutting job.

        POST /api/jobs/
        {
            "stock_dimensions": {"length": 800, "width": 500, "height": 2000},
            "parts": [
                {"name": "G14", "quantity": 5, "W1": 598.8, "W2": 566.3, "D": 444.5, "thickness": 67.8, "alpha": 2.094}
            ],
            "config_params": {"saw_kerf": 0.0}
        }
        """
        # Validate input
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        # Create job object
        stock_block_id = serializer.validated_data.get('stock_block_id')
        stock_block = None
        if stock_block_id:
            try:
                stock_block = StockBlock.objects.get(id=stock_block_id)
            except StockBlock.DoesNotExist:
                pass

        job = CuttingJob.objects.create(
            stock_dimensions=serializer.validated_data['stock_dimensions'],
            parts_spec=serializer.validated_data['parts'],
            config_params=serializer.validated_data.get('config_params', {}),
            stock_block=stock_block,
            status='running',
            started_at=timezone.now()
        )

        # Run optimization (synchronously for now, can be moved to Celery later)
        try:
            service = get_cutting_service()
            results = service.run_cutting_job(
                stock_dimensions=job.stock_dimensions,
                parts_spec=job.parts_spec,
                config_params=job.config_params
            )

            # Update job with results
            job.results = results
            job.visualization_files = results.get('visualization_files', [])
            job.status = 'completed'
            job.completed_at = timezone.now()
            job.save()

        except Exception as e:
            # Mark job as failed
            job.status = 'failed'
            job.error_message = str(e)
            job.completed_at = timezone.now()
            job.save()

            return Response(
                {
                    'error': str(e),
                    'job_id': job.id,
                    'status': 'failed'
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

        # Return job details
        response_serializer = CuttingJobSerializer(job)
        return Response(response_serializer.data, status=status.HTTP_201_CREATED)

    def retrieve(self, request, *args, **kwargs):
        """
        Get job status and results.

        GET /api/jobs/{id}/
        """
        job = self.get_object()
        serializer = self.get_serializer(job)
        return Response(serializer.data)

    @action(detail=True, methods=['post'])
    def rerun(self, request, pk=None):
        """
        Rerun a job with the same parameters.

        POST /api/jobs/{id}/rerun/
        """
        original_job = self.get_object()

        # Create new job with same parameters
        new_job = CuttingJob.objects.create(
            stock_dimensions=original_job.stock_dimensions,
            parts_spec=original_job.parts_spec,
            config_params=original_job.config_params,
            stock_block=original_job.stock_block,
            status='running',
            started_at=timezone.now()
        )

        # Run optimization
        try:
            service = get_cutting_service()
            results = service.run_cutting_job(
                stock_dimensions=new_job.stock_dimensions,
                parts_spec=new_job.parts_spec,
                config_params=new_job.config_params
            )

            new_job.results = results
            new_job.visualization_files = results.get('visualization_files', [])
            new_job.status = 'completed'
            new_job.completed_at = timezone.now()
            new_job.save()

        except Exception as e:
            new_job.status = 'failed'
            new_job.error_message = str(e)
            new_job.completed_at = timezone.now()
            new_job.save()

            return Response(
                {'error': str(e), 'job_id': new_job.id},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

        serializer = self.get_serializer(new_job)
        return Response(serializer.data, status=status.HTTP_201_CREATED)


class Top3ConfigurationsView(APIView):
    """
    API endpoint to compute and return top 3 packing configurations.

    POST /api/configurations/top3/
    {
        "stock_dimensions": {"length": 800, "width": 500, "height": 2000},
        "parts": [
            {"name": "G14", "W1": 598.8, "W2": 566.3, "D": 444.5, "thickness": 67.8, "alpha": 2.094},
            {"name": "G13", "W1": 598.8, "W2": 568.7, "D": 397.7, "thickness": 67.8, "alpha": 2.175}
        ],
        "config_params": {
            "saw_kerf": 0.0,
            "merging_plane_orders": ["XY-X", "XY-Y", "XZ-X"]
        },
        "top_n": 3
    }
    """

    def post(self, request):
        """Compute top N configurations."""
        # Validate input
        input_serializer = Top3ConfigurationsRequestSerializer(data=request.data)
        input_serializer.is_valid(raise_exception=True)

        # Create configuration set
        config_set = ConfigurationSet.objects.create(
            stock_dimensions=input_serializer.validated_data['stock_dimensions'],
            parts_spec=input_serializer.validated_data['parts'],
            config_params=input_serializer.validated_data.get('config_params', {}),
            status='running'
        )

        # Run optimization
        try:
            top_n = input_serializer.validated_data.get('top_n', 3)

            # Use the new API wrapper
            result = compute_top3_approaches(
                stock_dimensions=config_set.stock_dimensions,
                parts=config_set.parts_spec,
                config_params=config_set.config_params,
                top_n=top_n
            )

            # Save configurations to database
            for config_data in result['top_approaches']:
                Configuration.objects.create(
                    configuration_set=config_set,
                    primary_part_name=config_data['primary_part'],
                    merging_plane_order=config_data['merging_plane_order'],
                    stock_dimensions=config_set.stock_dimensions,
                    parts_spec=config_set.parts_spec,
                    config_params=config_set.config_params,
                    total_parts=config_data['total_parts'],
                    total_volume_used=result[f"Approach_{config_data['rank']}"].get('total_volume_used', 0),
                    waste_percentage=config_data['waste'],
                    is_extractable=True,  # All results from pack_manually are extractable
                    parts_breakdown=config_data['parts_breakdown'],
                    visualization_file=config_data['visualization_file'],
                )

            config_set.status = 'completed'
            config_set.completed_at = timezone.now()
            config_set.save()

            # Return response in the format expected by pack_manually_api
            return Response({
                'configuration_set_id': config_set.id,
                'configurations': result['top_approaches']
            }, status=status.HTTP_200_OK)

        except Exception as e:
            config_set.status = 'failed'
            config_set.error_message = str(e)
            config_set.completed_at = timezone.now()
            config_set.save()

            import traceback
            return Response(
                {
                    'error': str(e),
                    'traceback': traceback.format_exc(),
                    'configuration_set_id': config_set.id
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class VisualizationFileView(APIView):
    """
    API endpoint to serve visualization files.

    GET /api/visualizations/{filepath}/
    """

    def get(self, request, filepath):
        """Serve visualization HTML file."""
        # Construct full path
        base_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs')
        full_path = os.path.join(base_dir, filepath)

        # Security: ensure path is within outputs directory
        full_path = os.path.abspath(full_path)
        base_dir = os.path.abspath(base_dir)

        if not full_path.startswith(base_dir):
            raise Http404("Invalid file path")

        if not os.path.exists(full_path):
            raise Http404("File not found")

        # Serve file
        try:
            return FileResponse(open(full_path, 'rb'), content_type='text/html')
        except Exception as e:
            raise Http404(f"Error serving file: {e}")


class ConfigurationSetViewSet(viewsets.ReadOnlyModelViewSet):
    """
    ViewSet for ConfigurationSet model (read-only).

    Provides:
    - GET /api/configuration-sets/ - List all configuration sets
    - GET /api/configuration-sets/{id}/ - Get specific configuration set
    """
    queryset = ConfigurationSet.objects.all()
    serializer_class = ConfigurationSetSerializer
    filterset_fields = ['status']
    ordering_fields = ['created_at', 'completed_at']
    ordering = ['-created_at']
