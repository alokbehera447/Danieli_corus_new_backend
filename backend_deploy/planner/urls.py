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
    upload_excel_file,  # Import the new file upload function
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

    # File upload endpoint
    path('upload/', upload_excel_file, name='upload-excel'),
    
    # Custom endpoints
    path('configurations/top3/', Top3ConfigurationsView.as_view(), name='top3-configurations'),
    path('visualizations/<path:filepath>/', VisualizationFileView.as_view(), name='visualization-file'),
]