"""
Django models for the cutting optimization planner.

These models store job specifications, configurations, and results.
The actual computation is handled by the service layer which interfaces with src/ modules.
"""

from django.db import models
from django.contrib.postgres.fields import ArrayField
from django.core.validators import MinValueValidator
import json


class StockBlock(models.Model):
    """
    Represents a stock block specification.

    Can be reused across multiple jobs.
    """
    name = models.CharField(max_length=100, unique=True)
    length = models.FloatField(validators=[MinValueValidator(0.0)], help_text="Length in mm")
    width = models.FloatField(validators=[MinValueValidator(0.0)], help_text="Width in mm")
    height = models.FloatField(validators=[MinValueValidator(0.0)], help_text="Height in mm")
    material_type = models.CharField(max_length=100, default="Generic")

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'stock_blocks'
        ordering = ['-created_at']

    @property
    def volume(self):
        """Calculate volume in mm³."""
        return self.length * self.width * self.height

    @property
    def dimensions_dict(self):
        """Return dimensions as dictionary."""
        return {
            'length': self.length,
            'width': self.width,
            'height': self.height
        }

    def __str__(self):
        return f"{self.name} ({self.length}×{self.width}×{self.height}mm)"


class PartSpecification(models.Model):
    """
    Represents a trapezoidal prism part specification.

    Can be reused across multiple jobs.
    """
    name = models.CharField(max_length=100, unique=True)
    W1 = models.FloatField(validators=[MinValueValidator(0.0)], help_text="Wider width in mm")
    W2 = models.FloatField(validators=[MinValueValidator(0.0)], help_text="Narrower width in mm")
    D = models.FloatField(validators=[MinValueValidator(0.0)], help_text="Length in mm")
    thickness = models.FloatField(validators=[MinValueValidator(0.0)], help_text="Thickness in mm")
    alpha = models.FloatField(validators=[MinValueValidator(0.0)], help_text="Taper angle in degrees")

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'part_specifications'
        ordering = ['name']

    @property
    def volume(self):
        """Calculate analytical volume in mm³."""
        return ((self.W1 + self.W2) / 2.0) * self.D * self.thickness

    @property
    def C(self):
        """Calculate offset C."""
        return abs(self.W1 - self.W2) / 2.0

    @property
    def dimensions_dict(self):
        """Return dimensions as dictionary."""
        return {
            'W1': self.W1,
            'W2': self.W2,
            'D': self.D,
            'thickness': self.thickness,
            'alpha': self.alpha
        }

    def __str__(self):
        return f"{self.name} (W1={self.W1}, W2={self.W2}, D={self.D}, t={self.thickness})"


class CuttingJob(models.Model):
    """
    Represents a cutting/packing optimization job.

    Stores the job specification and results.
    """
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('running', 'Running'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    ]

    # Job metadata
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    created_at = models.DateTimeField(auto_now_add=True)
    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    error_message = models.TextField(null=True, blank=True)

    # Job specification (stored as JSON for flexibility)
    # This allows for custom stock dimensions or parts not in the database
    stock_dimensions = models.JSONField(help_text="Stock block dimensions {length, width, height}")
    parts_spec = models.JSONField(help_text="List of parts with quantities and dimensions")
    config_params = models.JSONField(default=dict, blank=True, help_text="Configuration parameters")

    # Optional: Link to database objects if they exist
    stock_block = models.ForeignKey(StockBlock, null=True, blank=True, on_delete=models.SET_NULL)

    # Results (stored as JSON)
    results = models.JSONField(null=True, blank=True, help_text="Optimization results")

    # File paths for generated outputs
    visualization_files = models.JSONField(default=list, blank=True, help_text="List of visualization file paths")
    report_files = models.JSONField(default=list, blank=True, help_text="List of report file paths")
    export_files = models.JSONField(default=list, blank=True, help_text="List of export file paths (STL, STEP, etc.)")

    class Meta:
        db_table = 'cutting_jobs'
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['status', 'created_at']),
        ]

    def __str__(self):
        return f"Job #{self.id} ({self.status}) - {self.created_at.strftime('%Y-%m-%d %H:%M')}"

    @property
    def stock_volume(self):
        """Calculate stock block volume."""
        dims = self.stock_dimensions
        return dims['length'] * dims['width'] * dims['height']

    @property
    def is_complete(self):
        """Check if job is complete."""
        return self.status in ['completed', 'failed']

    @property
    def duration_seconds(self):
        """Calculate job duration in seconds."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None


class Configuration(models.Model):
    """
    Represents a single packing configuration.

    Used for the "top 3 configurations" feature where multiple
    configurations are computed and ranked.
    """
    # Link to parent job (if this is part of a job)
    job = models.ForeignKey(CuttingJob, null=True, blank=True, on_delete=models.CASCADE, related_name='configurations')

    # Configuration metadata
    created_at = models.DateTimeField(auto_now_add=True)
    primary_part_name = models.CharField(max_length=100, help_text="Primary part type used")
    merging_plane_order = models.CharField(max_length=50, help_text="Merging plane order (e.g., XY-X)")

    # Configuration specification
    stock_dimensions = models.JSONField(help_text="Stock block dimensions")
    parts_spec = models.JSONField(help_text="List of parts used")
    config_params = models.JSONField(default=dict, blank=True)

    # Results
    total_parts = models.IntegerField(default=0)
    total_volume_used = models.FloatField(default=0.0)
    waste_percentage = models.FloatField(default=0.0)
    is_extractable = models.BooleanField(default=False)

    # Parts breakdown (stored as JSON)
    parts_breakdown = models.JSONField(default=dict, help_text="Count of each part type placed")

    # Placement details
    placements = models.JSONField(default=list, blank=True, help_text="Detailed part placements")

    # Files
    visualization_file = models.CharField(max_length=500, blank=True, help_text="Path to Plotly HTML visualization")

    class Meta:
        db_table = 'configurations'
        ordering = ['waste_percentage', '-total_parts']  # Best configurations first
        indexes = [
            models.Index(fields=['waste_percentage', 'total_parts']),
        ]

    def __str__(self):
        return f"Config #{self.id} - {self.primary_part_name} ({self.total_parts} parts, {self.waste_percentage:.1f}% waste)"

    @property
    def summary(self):
        """Generate human-readable summary."""
        parts_str = " + ".join([f"{count} {name}" for name, count in sorted(self.parts_breakdown.items(), key=lambda x: -x[1])])
        return f"{parts_str} and {self.waste_percentage:.1f}% Waste"

    @property
    def efficiency_percentage(self):
        """Calculate efficiency percentage (inverse of waste)."""
        return 100.0 - self.waste_percentage


class ConfigurationSet(models.Model):
    """
    Represents a set of configurations computed together.

    Used for the "top 3 configurations" API endpoint.
    """
    created_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True)

    # Request specification
    stock_dimensions = models.JSONField()
    parts_spec = models.JSONField()
    config_params = models.JSONField(default=dict, blank=True)

    # Status
    status = models.CharField(max_length=20, choices=CuttingJob.STATUS_CHOICES, default='pending')
    error_message = models.TextField(null=True, blank=True)

    # Computed configurations are linked via ForeignKey from Configuration model
    # Access via: configuration_set.configurations.all()

    class Meta:
        db_table = 'configuration_sets'
        ordering = ['-created_at']

    def __str__(self):
        return f"ConfigSet #{self.id} ({self.status}) - {self.configurations.count()} configs"

    @property
    def top_configurations(self, n=3):
        """Get top N configurations by efficiency."""
        return self.configurations.order_by('waste_percentage', '-total_parts')[:n]


# Link Configuration to ConfigurationSet
Configuration.add_to_class(
    'configuration_set',
    models.ForeignKey(
        ConfigurationSet,
        null=True,
        blank=True,
        on_delete=models.CASCADE,
        related_name='configurations'
    )
)
