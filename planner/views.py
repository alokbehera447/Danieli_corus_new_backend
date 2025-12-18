# ================================
# GLOBAL OPTIMIZATION STATE (DEV)
# ================================

GLOBAL_OPTIMIZATION_STATE = {
    "helper": None
}
"""
API views for the cutting optimization planner.
"""
from rest_framework import viewsets, status
from rest_framework.decorators import action, api_view, permission_classes
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.permissions import IsAuthenticated
from django.utils import timezone
from django.http import FileResponse, Http404, JsonResponse
from django.shortcuts import get_object_or_404
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
import os
import pandas as pd
import json
import sys
import time

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

# Import from your new modules
try:
    from .modules.packing_module import OptimizationEngine, pack_trapezoidal_prisms as new_pack_trapezoidal_prisms
    from .modules.packing_orchestrator import Prisms, run_final_code, get_block_details
except ImportError as e:
    print(f"Warning: Could not import packing modules: {e}")
    OptimizationEngine = None
    new_pack_trapezoidal_prisms = None
    Prisms = None
    run_final_code = None
    get_block_details = None




# ================================
# FILE UPLOAD VIEW FUNCTION
# ================================

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def upload_excel_file(request):
    """
    Handle Excel file upload and return processed block data.
    
    POST /api/upload/
    Content-Type: multipart/form-data
    Body: file (Excel file with block data)
    
    Returns:
    {
        "success": true,
        "data": [
            {
                "MARK": "G14",
                "Bottom Length": 150.0,
                "Top Length": 100.0,
                "Width": 80.0,
                "Height": 40.0,
                "Nos": 5
            },
            ...
        ],
        "totalRows": 10,
        "message": "Successfully processed 10 blocks"
    }
    """
    try:
        # Check if file was uploaded
        if 'file' not in request.FILES:
            return Response(
                {'success': False, 'error': 'No file uploaded'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        file = request.FILES['file']
        
        # Validate file type
        file_name = file.name.lower()
        if not (file_name.endswith('.xlsx') or 
                file_name.endswith('.xls') or 
                file_name.endswith('.csv')):
            return Response(
                {'success': False, 'error': 'Invalid file type. Please upload .xlsx, .xls, or .csv files'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        print(f"[File Upload] Processing file: {file_name}")
        
        # Read the file based on type
        try:
            if file_name.endswith('.csv'):
                df = pd.read_csv(file)
            else:
                df = pd.read_excel(file, engine='openpyxl')
        except Exception as e:
            return Response(
                {'success': False, 'error': f'Error reading file: {str(e)}'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        print(f"[File Upload] DataFrame columns: {df.columns.tolist()}")
        print(f"[File Upload] DataFrame shape: {df.shape}")
        
        # Clean column names (remove whitespace, lowercase for matching)
        df.columns = [str(col).strip() for col in df.columns]
        
        # Define expected columns and their possible variations
        column_variations = {
            'MARK': ['MARK', 'mark', 'Mark', 'Block ID', 'BLOCK ID', 'block_id', 'ID', 'id'],
            'Bottom Length': ['Bottom Length', 'BottomLength', 'Bottom_Length', 'bottom length', 
                             'bottom_length', 'Bottom', 'BLENGTH', 'B Length', 'Base Length', 
                             'base length', 'BASE LENGTH', 'Long Base', 'A(W1)', 'A', 'W1'],
            'Top Length': ['Top Length', 'TopLength', 'Top_Length', 'top length', 'top_length', 
                          'Top', 'TLENGTH', 'T Length', 'Short Base', 'short base', 'SHORT BASE', 
                          'B(W2)', 'B', 'W2'],
            'Width': ['Width', 'width', 'WIDTH', 'W', 'w', 'Breadth', 'breadth', 'BREADTH', 
                     'D(length)', 'D', 'length'],
            'Height': ['Height', 'height', 'HEIGHT', 'H', 'h', 'Thickness', 'thickness', 
                      'THICKNESS', 'Depth', 'depth', 'DEPTH'],
            'Nos': ['Nos', 'nos', 'NOS', 'Quantity', 'quantity', 'QTY', 'qty', 'Count', 
                   'count', 'COUNT', 'Number', 'number', 'NUMBER', 'Units', 'units', 'UNITS']
        }
        
        # Map actual columns to standard names
        column_mapping = {}
        for standard_name, variations in column_variations.items():
            for col in df.columns:
                # Case-insensitive matching
                if str(col).lower() in [v.lower() for v in variations]:
                    column_mapping[col] = standard_name
                    print(f"[File Upload] Mapped column '{col}' -> '{standard_name}'")
                    break
        
        # Apply the mapping
        df.rename(columns=column_mapping, inplace=True)
        
        print(f"[File Upload] After renaming columns: {df.columns.tolist()}")
        
        # Check if we have the essential MARK column
        if 'MARK' not in df.columns:
            return Response(
                {'success': False, 'error': 'File must contain a MARK/Block ID column'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Process each row
        processed_data = []
        for index, row in df.iterrows():
            # Skip empty rows (where MARK is NaN or empty)
            mark_value = row.get('MARK')
            if pd.isna(mark_value) or str(mark_value).strip() == '':
                continue
            
            block_data = {}
            
            # Process each standard column
            standard_columns = ['MARK', 'Bottom Length', 'Top Length', 'Width', 'Height', 'Nos']
            
            for col in standard_columns:
                if col in df.columns:
                    value = row[col]
                    # Convert to appropriate type
                    if pd.isna(value):
                        block_data[col] = None
                    elif col == 'MARK':
                        # Keep MARK as string
                        block_data[col] = str(value).strip()
                    else:
                        # Try to convert numeric columns to float
                        try:
                            if col == 'Nos':
                                block_data[col] = int(float(value))
                            else:
                                block_data[col] = float(value)
                        except (ValueError, TypeError):
                            # If conversion fails, keep as string or None
                            try:
                                block_data[col] = str(value).strip()
                            except:
                                block_data[col] = None
                else:
                    block_data[col] = None
            
            processed_data.append(block_data)
        
        print(f"[File Upload] Processed {len(processed_data)} rows")
        
        if len(processed_data) == 0:
            return Response(
                {'success': False, 'error': 'No valid data found in the file'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Return success response
        return Response({
            'success': True,
            'data': processed_data,
            'totalRows': len(processed_data),
            'message': f'Successfully processed {len(processed_data)} blocks'
        }, status=status.HTTP_200_OK)
        
    except Exception as e:
        print(f"[File Upload] Error: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return Response(
            {
                'success': False, 
                'error': f'Error processing file: {str(e)}'
            },
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


# ================================
# VISUALIZATION FUNCTIONS
# ================================

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def generate_block_visualization(request, block_code):
    try:
        from django.conf import settings
        from django.core.cache import cache
        from datetime import datetime
        import os, time

        # 🔒 Wait for helper (Gunicorn-safe)
        helper = None
        for _ in range(20):
            helper = cache.get("latest_helper")
            if helper is not None:
                break
            time.sleep(0.2)

        if helper is None:
            return Response({
                "success": False,
                "error": "Optimization data not ready. Please retry."
            }, status=400)

        block_index = int(block_code.replace("B", "")) - 1

        if block_index < 0 or block_index >= len(helper.all_big_blocks):
            return Response({
                "success": False,
                "error": "Invalid block code"
            }, status=400)

        block = helper.all_big_blocks[block_index]

        viz_dir = os.path.join(settings.MEDIA_ROOT, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)

        filename = f"block_{block.unique_code}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        filepath = os.path.join(viz_dir, filename)

        block.draw_it(only_scrap=False, save_path=filepath)

        # ⏳ wait for file write
        for _ in range(30):
            if os.path.exists(filepath) and os.path.getsize(filepath) > 1024:
                break
            time.sleep(0.1)

        return Response({
            "success": True,
            "visualization_url": f"/media/visualizations/{filename}"
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return Response({
            "success": False,
            "error": str(e)
        }, status=500)




@api_view(['POST'])
@permission_classes([IsAuthenticated])
def generate_scrap_visualization(request, scrap_code):
    try:
        from django.conf import settings
        from django.core.cache import cache
        from datetime import datetime
        import os, time

        helper = None
        for _ in range(20):
            helper = cache.get("latest_helper")
            if helper is not None:
                break
            time.sleep(0.2)

        if helper is None:
            return Response({
                "success": False,
                "error": "Optimization data not ready. Please retry."
            }, status=400)

        scrap = next(
            (s for s in helper.all_scrap if s.unique_code == scrap_code),
            None
        )

        if scrap is None:
            return Response({
                "success": False,
                "error": "Invalid scrap code"
            }, status=404)

        viz_dir = os.path.join(settings.MEDIA_ROOT, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)

        filename = f"scrap_{scrap.unique_code}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        filepath = os.path.join(viz_dir, filename)

        scrap.draw_scrap(save_path=filepath)

        for _ in range(30):
            if os.path.exists(filepath) and os.path.getsize(filepath) > 1024:
                break
            time.sleep(0.1)

        return Response({
            "success": True,
            "visualization_url": f"/media/visualizations/{filename}"
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return Response({
            "success": False,
            "error": str(e)
        }, status=500)



@api_view(['GET'])
def get_visualization_file(request, filename):
    """
    Serve visualization HTML file
    
    GET /api/visualization/file/{filename}/
    """
    try:
        # Construct full path
        from django.conf import settings
        viz_dir = os.path.join(settings.MEDIA_ROOT, 'visualizations')
        filepath = os.path.join(viz_dir, filename)
        
        # Security check
        filepath = os.path.abspath(filepath)
        viz_dir = os.path.abspath(viz_dir)
        
        if not filepath.startswith(viz_dir):
            raise Http404("Invalid file path")
        
        if not os.path.exists(filepath):
            raise Http404("File not found")
        
        # Serve file
        return FileResponse(open(filepath, 'rb'), content_type='text/html')
        
    except Exception as e:
        raise Http404(f"Error serving file: {e}")


# ================================
# MAIN OPTIMIZATION ENDPOINT (UPDATED)
# ================================

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def upload_and_optimize(request):
    """
    Upload Excel file and run optimization with custom parent blocks
    
    POST /api/upload-optimize/
    Content-Type: multipart/form-data
    
    Form Data:
    - file: Excel file
    - selected_blocks: JSON array of selected block IDs (optional)
    - parent_blocks: JSON array of parent block sizes (required)
    - buffer_spacing: float (default: 2.0)
    """
    try:
        # Get uploaded file
        excel_file = request.FILES.get('file')
        if not excel_file:
            return Response({
                'success': False,
                'error': 'No file uploaded'
            }, status=400)
        
        # Get parameters
        selected_blocks_json = request.POST.get('selected_blocks', '[]')
        parent_blocks_json = request.POST.get('parent_blocks', '[]')
        buffer_spacing = float(request.POST.get('buffer_spacing', '2.0'))
        
        try:
            selected_blocks = json.loads(selected_blocks_json)
            parent_blocks_data = json.loads(parent_blocks_json)
        except json.JSONDecodeError as e:
            return Response({
                'success': False,
                'error': f'Invalid JSON in parameters: {str(e)}'
            }, status=400)
        
        print(f"\n=== OPTIMIZATION REQUEST ===")
        print(f"File: {excel_file.name}")
        print(f"Selected blocks: {selected_blocks}")
        print(f"Parent blocks: {parent_blocks_data}")
        print(f"Buffer spacing: {buffer_spacing}")
        
        # Validate and prepare parent block sizes
        parent_block_sizes = []
        
        # Handle the format sent from frontend
        for block in parent_blocks_data:
            if isinstance(block, dict):
                # Format: {"label": "800×350×1870", "dimensions": {"length": 1870, "width": 800, "height": 350}}
                if 'dimensions' in block:
                    dims = block['dimensions']
                    parent_block_sizes.append([
                        dims.get('length', 0),
                        dims.get('width', 0),
                        dims.get('height', 0)
                    ])
                # Format: {"length": 1870, "width": 800, "height": 350}
                elif 'length' in block and 'width' in block and 'height' in block:
                    parent_block_sizes.append([
                        block['length'],
                        block['width'],
                        block['height']
                    ])
            elif isinstance(block, list) and len(block) == 3:
                # Format: [1870, 800, 350]
                parent_block_sizes.append(block)
        
        if not parent_block_sizes:
            return Response({
                'success': False,
                'error': 'No valid parent blocks provided'
            }, status=400)
        
        # Validate dimensions
        for i, size in enumerate(parent_block_sizes):
            if len(size) != 3:
                return Response({
                    'success': False,
                    'error': f'Parent block {i}: Must have exactly 3 dimensions (length, width, height)'
                }, status=400)
            
            length, width, height = size
            if length <= 0 or width <= 0 or height <= 0:
                return Response({
                    'success': False,
                    'error': f'Parent block {i}: All dimensions must be positive'
                }, status=400)
        
        print(f"Parent block sizes to use: {parent_block_sizes}")
        
        # Process Excel file and run optimization...
        # ... rest of your existing upload_and_optimize code
        
        # Process Excel file
        if OptimizationEngine is None:
            return Response({
                'success': False,
                'error': 'Optimization engine not available'
            }, status=500)
        
        engine = OptimizationEngine(
            stock_dimensions={'length': 2000, 'width': 500, 'height': 500},
            parts_data=[],
            buffer_spacing=buffer_spacing
        )
        
        # Process Excel data
        parts_data = engine.process_excel_data(excel_file)
        
        if not parts_data:
            return Response({
                'success': False,
                'error': 'No valid data found in Excel file'
            }, status=400)
        
        print(f"Processed {len(parts_data)} parts from Excel")
        
        # Filter selected blocks if specified
        if selected_blocks:
            parts_data = [p for p in parts_data if p['MARK'] in selected_blocks]
            print(f"Filtered to {len(parts_data)} selected parts")
        
        # Check if packing modules are available
        if Prisms is None or run_final_code is None:
            return Response({
                'success': False,
                'error': 'Packing modules not available'
            }, status=500)
        
        # Create prism objects
        all_prisms = []
        for part in parts_data:
            try:
                size = [
                    part['Bottom Length'],
                    part['Top Length'],
                    part['Width'],
                    part['Height']
                ]
                prism = Prisms(
                    code=part['MARK'],
                    size=size,
                    quantity=part['Nos']
                )
                all_prisms.append(prism)
            except Exception as e:
                print(f"Error creating prism {part.get('MARK', 'Unknown')}: {e}")
                continue
        
        if not all_prisms:
            return Response({
                'success': False,
                'error': 'No valid prism objects created'
            }, status=400)
        
        print(f"Created {len(all_prisms)} prism objects")
        
        # Run the packing algorithm
        try:
            helper = run_final_code(
                all_prisms=all_prisms,
                buffer=buffer_spacing,
                parent_block_sizes=parent_block_sizes
            )

            # 🔥 STORE HELPER (Django-safe)
            # GLOBAL_OPTIMIZATION_STATE["helper"] = helper

            from django.core.cache import cache

            cache.set(
                "latest_helper",
                helper,
                timeout=60 * 60  # 1 hour
            )

        except Exception as e:
            print(f"ERROR in run_final_code: {str(e)}")
            import traceback
            traceback.print_exc()
            return Response({
                'success': False,
                'error': f'Packing algorithm failed: {str(e)}'
            }, status=500)
        
        # Get detailed results
        if get_block_details is None:
            return Response({
                'success': False,
                'error': 'Block details module not available'
            }, status=500)
        
        block_details = get_block_details(helper)
        
        print(f"Packing complete!")
        print(f"Total blocks created: {len(helper.all_big_blocks)}")
        print(f"Total scraps: {len(helper.all_scrap)}")
        
        # Calculate totals
        total_parts_packed = 0
        total_prism_volume = 0
        total_requested = 0
        
        for prism in all_prisms:
            packed = prism.quantity - prism.prism_left
            total_parts_packed += packed
            total_prism_volume += prism.volume * packed
            total_requested += prism.quantity
        
        # Calculate total stock volume
        total_stock_volume = 0
        for block in helper.all_big_blocks:
            total_stock_volume += block.volume
        
        # Calculate efficiency
        if total_stock_volume > 0:
            efficiency = (total_prism_volume / total_stock_volume) * 100
        else:
            efficiency = 0
        
        # Prepare detailed block information
        blocks_info = []
        for block in helper.all_big_blocks:
            # Count prisms in this block
            prism_counts = {}
            for entry in block.prism_details:
                prism = entry['prism']
                count = len(entry['coordinates'])
                if prism.code not in prism_counts:
                    prism_counts[prism.code] = 0
                prism_counts[prism.code] += count
            
            prism_list = [{"code": code, "count": count} for code, count in prism_counts.items()]
            
            blocks_info.append({
                'code': block.unique_code,
                'size': block.size,
                'efficiency': round(block.get_efficiency(), 2),
                'prisms': prism_list,
                'volume': float(block.volume)
            })
        
        # Prepare scrap information
        scraps_info = []
        for scrap in helper.all_scrap:
            scraps_info.append({
                'code': scrap.unique_code,
                'size': scrap.size,
                'volume': float(scrap.volume)
            })
        
        # Prepare response
        results = {
            'success': True,
            'efficiency': round(efficiency, 2),
            'total_parts_packed': total_parts_packed,
            'total_parts_requested': total_requested,
            'packing_percentage': round(total_parts_packed / total_requested * 100, 2) if total_requested > 0 else 0,
            'total_blocks_created': len(helper.all_big_blocks),
            'total_stock_volume': round(total_stock_volume, 2),
            'total_prism_volume': round(total_prism_volume, 2),
            'waste_percentage': round(100 - efficiency, 2),
            'blocks': blocks_info,
            'scraps': scraps_info,
            'parent_blocks_used': parent_block_sizes,
            'message': f'Created {len(helper.all_big_blocks)} stock blocks with {efficiency:.2f}% efficiency'
        }
        
        return Response(results)
        
    except Exception as e:
        print(f"ERROR in upload_and_optimize: {str(e)}")
        import traceback
        traceback.print_exc()
        return Response({
            'success': False,
            'error': str(e)
        }, status=500)


# ================================
# TEST ENDPOINTS
# ================================

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def test_complete_orchestrator(request):
    """Test the complete orchestrator system"""
    try:
        # Use test data
        test_parts_data = [{
            'MARK': 'G14',
            'Bottom Length': 598.8,
            'Top Length': 566.3,
            'Width': 444.5,
            'Height': 67.8,
            'Nos': 5
        }, {
            'MARK': 'G15',
            'Bottom Length': 598.8,
            'Top Length': 581.0,
            'Width': 242.5,
            'Height': 93.6,
            'Nos': 3
        }]
        
        # Get parent blocks from request or use defaults
        parent_blocks_data = request.data.get('parent_blocks', [
            {'length': 1870, 'width': 800, 'height': 350},
            {'length': 2000, 'width': 800, 'height': 400}
        ])
        
        parent_block_sizes = []
        for block in parent_blocks_data:
            if isinstance(block, dict) and 'length' in block and 'width' in block and 'height' in block:
                parent_block_sizes.append([block['length'], block['width'], block['height']])
        
        if not parent_block_sizes:
            parent_block_sizes = [[2000, 500, 500]]
        
        print(f"\n=== TESTING ORCHESTRATOR ===")
        print(f"Parent block sizes: {parent_block_sizes}")
        
        from .modules.packing_module import OptimizationEngine
        
        engine = OptimizationEngine(
            stock_dimensions={'length': 2000, 'width': 500, 'height': 500},
            parts_data=test_parts_data,
            buffer_spacing=2.0
        )
        
        results = engine.optimize(selected_blocks=None, parent_block_sizes=parent_block_sizes)
        
        return Response({
            'test_success': True,
            'results': results
        })
        
    except Exception as e:
        import traceback
        return Response({
            'test_success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }, status=500)


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def debug_excel_data(request):
    """
    Debug endpoint to see Excel file contents
    """
    try:
        if 'file' not in request.FILES:
            return Response({'success': False, 'error': 'No file uploaded'}, status=400)
        
        excel_file = request.FILES['file']
        
        # Read the file
        if excel_file.name.endswith('.csv'):
            df = pd.read_csv(excel_file)
        else:
            df = pd.read_excel(excel_file, engine='openpyxl')
        
        # Get basic info
        file_info = {
            'filename': excel_file.name,
            'size': excel_file.size,
            'rows': len(df),
            'columns': len(df.columns),
            'column_names': df.columns.tolist(),
            'first_5_rows': df.head().to_dict('records'),
            'dtypes': {col: str(df[col].dtype) for col in df.columns}
        }
        
        # Check for required columns
        required_columns = ['MARK', 'Bottom Length', 'Top Length', 'Width', 'Height', 'Nos']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        # Check data types
        numeric_issues = []
        for col in ['Bottom Length', 'Top Length', 'Width', 'Height', 'Nos']:
            if col in df.columns:
                non_numeric = df[col].apply(lambda x: not (isinstance(x, (int, float)) or (isinstance(x, str) and x.replace('.', '', 1).isdigit())))
                if non_numeric.any():
                    numeric_issues.append(f"{col}: {non_numeric.sum()} non-numeric values")
        
        return Response({
            'success': True,
            'file_info': file_info,
            'missing_columns': missing_columns,
            'numeric_issues': numeric_issues,
            'sample_data': df.head(10).to_dict('records')
        })
        
    except Exception as e:
        return Response({'success': False, 'error': str(e)}, status=500)


# ================================
# EXISTING VIEWSETS (UNCHANGED)
# ================================

class StockBlockViewSet(viewsets.ModelViewSet):
    """
    ViewSet for StockBlock model.
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
    """
    queryset = CuttingJob.objects.all()
    serializer_class = CuttingJobSerializer
    filterset_fields = ['status']
    ordering_fields = ['created_at', 'completed_at']
    ordering = ['-created_at']

    def get_serializer_class(self):
        if self.action == 'create':
            return CuttingJobCreateSerializer
        return CuttingJobSerializer

    def create(self, request, *args, **kwargs):
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

        # Run optimization
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

        response_serializer = CuttingJobSerializer(job)
        return Response(response_serializer.data, status=status.HTTP_201_CREATED)

    def retrieve(self, request, *args, **kwargs):
        job = self.get_object()
        serializer = self.get_serializer(job)
        return Response(serializer.data)

    @action(detail=True, methods=['post'])
    def rerun(self, request, pk=None):
        original_job = self.get_object()

        new_job = CuttingJob.objects.create(
            stock_dimensions=original_job.stock_dimensions,
            parts_spec=original_job.parts_spec,
            config_params=original_job.config_params,
            stock_block=original_job.stock_block,
            status='running',
            started_at=timezone.now()
        )

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


class ConfigurationSetViewSet(viewsets.ReadOnlyModelViewSet):
    """
    ViewSet for ConfigurationSet model.
    """
    queryset = ConfigurationSet.objects.all()
    serializer_class = ConfigurationSetSerializer
    filterset_fields = ['status']
    ordering_fields = ['created_at', 'completed_at']
    ordering = ['-created_at']


# ================================
# OTHER EXISTING ENDPOINTS
# ================================

class Top3ConfigurationsView(APIView):
    """
    API endpoint to compute and return top 3 packing configurations.
    """
    def post(self, request):
        input_serializer = Top3ConfigurationsRequestSerializer(data=request.data)
        input_serializer.is_valid(raise_exception=True)

        config_set = ConfigurationSet.objects.create(
            stock_dimensions=input_serializer.validated_data['stock_dimensions'],
            parts_spec=input_serializer.validated_data['parts'],
            config_params=input_serializer.validated_data.get('config_params', {}),
            status='running'
        )

        try:
            top_n = input_serializer.validated_data.get('top_n', 3)
            
            # Import the API wrapper
            try:
                from pack_manually_api import compute_top3_approaches
            except ImportError:
                return Response({
                    'success': False,
                    'error': 'pack_manually_api module not found'
                }, status=500)

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
                    is_extractable=True,
                    parts_breakdown=config_data['parts_breakdown'],
                    visualization_file=config_data['visualization_file'],
                )

            config_set.status = 'completed'
            config_set.completed_at = timezone.now()
            config_set.save()

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
    """
    def get(self, request, filepath):
        base_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs', 'visualizations')
        full_path = os.path.join(base_dir, filepath)

        full_path = os.path.abspath(full_path)
        base_dir = os.path.abspath(base_dir)

        if not full_path.startswith(base_dir):
            raise Http404("Invalid file path")

        if not os.path.exists(full_path):
            raise Http404("File not found")

        try:
            return FileResponse(open(full_path, 'rb'), content_type='text/html')
        except Exception as e:
            raise Http404(f"Error serving file: {e}")


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def api_trapezoidal_packing(request):
    """
    API endpoint for trapezoidal packing (JSON input)
    """
    try:
        data = request.data
        
        stock_dimensions = data.get('stock_dimensions', {
            'length': 2000,
            'width': 500,
            'height': 500
        })
        
        parts = data.get('parts', [])
        config_params = data.get('config_params', {})
        top_n = data.get('top_n', 3)
        
        if new_pack_trapezoidal_prisms is None:
            return Response(
                {'success': False, 'error': 'Trapezoidal packing module not available'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
        
        results = new_pack_trapezoidal_prisms(
            stock_dimensions=stock_dimensions,
            parts=parts,
            config_params=config_params,
            top_n=top_n
        )
        
        return Response(results, status=status.HTTP_200_OK)
        
    except Exception as e:
        print(f"[Trapezoidal Packing] Error: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return Response(
            {'success': False, 'error': str(e)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@csrf_exempt
@require_POST
def upload_optimize_django(request):
    """
    Django view decorator version of upload_and_optimize
    """
    try:
        from rest_framework.request import Request
        from rest_framework.parsers import MultiPartParser, FormParser
        
        drf_request = Request(request, parsers=[MultiPartParser(), FormParser()])
        return upload_and_optimize(drf_request)
        
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)}, status=500)