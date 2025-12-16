"""
Django admin configuration for the planner app.
"""

from django.contrib import admin
from .models import (
    StockBlock,
    PartSpecification,
    CuttingJob,
    Configuration,
    ConfigurationSet
)


@admin.register(StockBlock)
class StockBlockAdmin(admin.ModelAdmin):
    """Admin interface for StockBlock model."""

    list_display = ['name', 'length', 'width', 'height', 'volume', 'material_type', 'created_at']
    list_filter = ['material_type', 'created_at']
    search_fields = ['name', 'material_type']
    readonly_fields = ['created_at', 'updated_at', 'volume']
    ordering = ['-created_at']

    fieldsets = [
        ('Basic Information', {
            'fields': ['name', 'material_type']
        }),
        ('Dimensions', {
            'fields': ['length', 'width', 'height', 'volume']
        }),
        ('Timestamps', {
            'fields': ['created_at', 'updated_at'],
            'classes': ['collapse']
        }),
    ]


@admin.register(PartSpecification)
class PartSpecificationAdmin(admin.ModelAdmin):
    """Admin interface for PartSpecification model."""

    list_display = ['name', 'W1', 'W2', 'D', 'thickness', 'alpha', 'volume', 'created_at']
    list_filter = ['thickness', 'created_at']
    search_fields = ['name']
    readonly_fields = ['created_at', 'updated_at', 'volume', 'C']
    ordering = ['name']

    fieldsets = [
        ('Basic Information', {
            'fields': ['name']
        }),
        ('Dimensions', {
            'fields': ['W1', 'W2', 'D', 'thickness', 'alpha', 'C', 'volume']
        }),
        ('Timestamps', {
            'fields': ['created_at', 'updated_at'],
            'classes': ['collapse']
        }),
    ]


@admin.register(CuttingJob)
class CuttingJobAdmin(admin.ModelAdmin):
    """Admin interface for CuttingJob model."""

    list_display = ['id', 'status', 'stock_block', 'created_at', 'completed_at', 'duration_seconds']
    list_filter = ['status', 'created_at', 'completed_at']
    search_fields = ['id', 'error_message']
    readonly_fields = ['created_at', 'started_at', 'completed_at', 'stock_volume', 'duration_seconds']
    ordering = ['-created_at']

    fieldsets = [
        ('Status', {
            'fields': ['status', 'error_message']
        }),
        ('Timestamps', {
            'fields': ['created_at', 'started_at', 'completed_at', 'duration_seconds']
        }),
        ('Specification', {
            'fields': ['stock_dimensions', 'stock_volume', 'stock_block', 'parts_spec', 'config_params'],
            'classes': ['collapse']
        }),
        ('Results', {
            'fields': ['results', 'visualization_files', 'report_files', 'export_files'],
            'classes': ['collapse']
        }),
    ]

    def duration_seconds(self, obj):
        """Display job duration."""
        duration = obj.duration_seconds
        if duration is not None:
            return f"{duration:.2f}s"
        return "-"
    duration_seconds.short_description = 'Duration'


@admin.register(Configuration)
class ConfigurationAdmin(admin.ModelAdmin):
    """Admin interface for Configuration model."""

    list_display = [
        'id', 'primary_part_name', 'merging_plane_order', 'total_parts',
        'waste_percentage', 'efficiency_percentage', 'is_extractable', 'created_at'
    ]
    list_filter = ['primary_part_name', 'merging_plane_order', 'is_extractable', 'created_at']
    search_fields = ['primary_part_name']
    readonly_fields = ['created_at', 'summary', 'efficiency_percentage']
    ordering = ['waste_percentage', '-total_parts']

    fieldsets = [
        ('Basic Information', {
            'fields': ['primary_part_name', 'merging_plane_order', 'job', 'configuration_set']
        }),
        ('Specification', {
            'fields': ['stock_dimensions', 'parts_spec', 'config_params'],
            'classes': ['collapse']
        }),
        ('Results', {
            'fields': [
                'total_parts', 'total_volume_used', 'waste_percentage',
                'efficiency_percentage', 'is_extractable', 'parts_breakdown', 'summary'
            ]
        }),
        ('Details', {
            'fields': ['placements', 'visualization_file'],
            'classes': ['collapse']
        }),
        ('Timestamps', {
            'fields': ['created_at'],
            'classes': ['collapse']
        }),
    ]

    def efficiency_percentage(self, obj):
        """Display efficiency percentage."""
        return f"{obj.efficiency_percentage:.2f}%"
    efficiency_percentage.short_description = 'Efficiency'


@admin.register(ConfigurationSet)
class ConfigurationSetAdmin(admin.ModelAdmin):
    """Admin interface for ConfigurationSet model."""

    list_display = ['id', 'status', 'config_count', 'created_at', 'completed_at']
    list_filter = ['status', 'created_at']
    readonly_fields = ['created_at', 'completed_at', 'config_count']
    ordering = ['-created_at']

    fieldsets = [
        ('Status', {
            'fields': ['status', 'error_message', 'config_count']
        }),
        ('Timestamps', {
            'fields': ['created_at', 'completed_at']
        }),
        ('Specification', {
            'fields': ['stock_dimensions', 'parts_spec', 'config_params'],
            'classes': ['collapse']
        }),
    ]

    def config_count(self, obj):
        """Display number of configurations."""
        return obj.configurations.count()
    config_count.short_description = 'Configurations'
