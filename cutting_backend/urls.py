"""
URL configuration for cutting_backend project.
"""
from django.conf import settings
from django.conf.urls.static import static
from rest_framework_simplejwt.views import TokenObtainPairView, TokenRefreshView
from django.contrib import admin
from django.urls import path, include
from drf_spectacular.views import (
    SpectacularAPIView,
    SpectacularSwaggerView,
    SpectacularRedocView,
)

urlpatterns = [
    # Admin
    path('admin/', admin.site.urls),
    path("auth/login/", TokenObtainPairView.as_view(), name="jwt_login"),
    path("auth/refresh/", TokenRefreshView.as_view(), name="jwt_refresh"),

    # API endpoints
    path('api/', include('planner.urls')),

    # API documentation (OpenAPI/Swagger)
    path('api/schema/', SpectacularAPIView.as_view(), name='schema'),
    path('api/schema/swagger-ui/', SpectacularSwaggerView.as_view(url_name='schema'), name='swagger-ui'),
    path('api/schema/redoc/', SpectacularRedocView.as_view(url_name='schema'), name='redoc'),

    # DRF browsable API auth
    path('api-auth/', include('rest_framework.urls')),
]
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
# Customize admin site
admin.site.site_header = "Cutting Optimization Admin"
admin.site.site_title = "Cutting Optimization"
admin.site.index_title = "Welcome to Cutting Optimization Administration"
