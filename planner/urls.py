"""
URL configuration for the planner app.
"""

from django.urls import path, include
from rest_framework.routers import DefaultRouter

from .views import (
    StockBlockViewSet,
    PartSpecificationViewSet,
    CuttingJobViewSet,
    ConfigurationSetViewSet,
    Top3ConfigurationsView,
    VisualizationFileView,
    delete_optimization_history,
    get_optimization_history,
    get_optimization_details,
    rename_optimization,
    delete_optimization_history,
    
    # File upload and optimization
    upload_excel_file,
    upload_and_optimize,
    
    # Visualization endpoints
    generate_block_visualization,
    generate_scrap_visualization,
    get_visualization_file,
    
    # Other endpoints
    api_trapezoidal_packing,
    upload_optimize_django,
    debug_excel_data,
    test_complete_orchestrator,
)

# Create router for viewsets
router = DefaultRouter()
router.register(r'stock-blocks', StockBlockViewSet, basename='stockblock')
router.register(r'part-specifications', PartSpecificationViewSet, basename='partspecification')
router.register(r'jobs', CuttingJobViewSet, basename='cuttingjob')
router.register(r'configuration-sets', ConfigurationSetViewSet, basename='configurationset')

# URL patterns
app_name = 'planner'

urlpatterns = [
    # Router URLs
    path('', include(router.urls)),
    
    # ================================
    # FILE UPLOAD & OPTIMIZATION
    # ================================
    path('upload/', upload_excel_file, name='upload-excel'),
    path('upload-optimize/', upload_and_optimize, name='upload-optimize'),
    path('upload-optimize-django/', upload_optimize_django, name='upload-optimize-django'),
    
    # ================================
    # 3D VISUALIZATION ENDPOINTS
    # ================================
    path('visualization/block/<str:block_code>/', generate_block_visualization, name='generate-block-visualization'),
    path('visualization/scrap/<str:scrap_code>/', generate_scrap_visualization, name='generate-scrap-visualization'),
    path('visualization/file/<str:filename>/', get_visualization_file, name='get-visualization-file'),
    path('visualizations/<path:filepath>/', VisualizationFileView.as_view(), name='visualization-file'),
    
    # ================================
    # OTHER ENDPOINTS
    # ================================
    path('configurations/top3/', Top3ConfigurationsView.as_view(), name='top3-configurations'),
    path('trapezoidal-packing/', api_trapezoidal_packing, name='trapezoidal-packing'),
    
    # ================================
    # DEBUG & TEST ENDPOINTS
    # ================================
    path('debug-excel/', debug_excel_data, name='debug-excel'),
    path('test-complete-orchestrator/', test_complete_orchestrator, name='test-complete-orchestrator'),

    # In planner/urls.py, add these to urlpatterns:

    path('optimization-history/', get_optimization_history, name='optimization-history'),
    path('optimization-history/<int:history_id>/', get_optimization_details, name='optimization-details'),
    path('optimization-history/<int:history_id>/rename/', rename_optimization, name='rename-optimization'),
    path('optimization-history/delete/', delete_optimization_history, name='delete-optimization-history'),
    ]