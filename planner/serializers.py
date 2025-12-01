"""
Django REST Framework serializers for API endpoints.
"""

from rest_framework import serializers
from .models import (
    StockBlock,
    PartSpecification,
    CuttingJob,
    Configuration,
    ConfigurationSet
)


class StockBlockSerializer(serializers.ModelSerializer):
    """Serializer for StockBlock model."""

    volume = serializers.FloatField(read_only=True)
    dimensions_dict = serializers.DictField(read_only=True)

    class Meta:
        model = StockBlock
        fields = [
            'id', 'name', 'length', 'width', 'height',
            'material_type', 'volume', 'dimensions_dict',
            'created_at', 'updated_at'
        ]
        read_only_fields = ['created_at', 'updated_at']


class PartSpecificationSerializer(serializers.ModelSerializer):
    """Serializer for PartSpecification model."""

    volume = serializers.FloatField(read_only=True)
    C = serializers.FloatField(read_only=True)
    dimensions_dict = serializers.DictField(read_only=True)

    class Meta:
        model = PartSpecification
        fields = [
            'id', 'name', 'W1', 'W2', 'D', 'thickness', 'alpha',
            'volume', 'C', 'dimensions_dict',
            'created_at', 'updated_at'
        ]
        read_only_fields = ['created_at', 'updated_at']


class CuttingJobCreateSerializer(serializers.Serializer):
    """
    Serializer for creating a new cutting job.

    Accepts flexible input without requiring database objects.
    """
    stock_dimensions = serializers.DictField(
        child=serializers.FloatField(min_value=0),
        help_text="Stock dimensions: {length, width, height} in mm"
    )

    parts = serializers.ListField(
        child=serializers.DictField(),
        help_text="List of parts with specifications"
    )

    config_params = serializers.DictField(
        required=False,
        default=dict,
        help_text="Optional configuration parameters"
    )

    stock_block_id = serializers.IntegerField(
        required=False,
        allow_null=True,
        help_text="Optional: ID of existing StockBlock"
    )

    def validate_stock_dimensions(self, value):
        """Validate stock dimensions have required keys."""
        required_keys = {'length', 'width', 'height'}
        if not required_keys.issubset(value.keys()):
            raise serializers.ValidationError(
                f"stock_dimensions must contain: {required_keys}"
            )
        for key in required_keys:
            if value[key] <= 0:
                raise serializers.ValidationError(
                    f"{key} must be positive"
                )
        return value

    def validate_parts(self, value):
        """Validate parts specifications."""
        if not value:
            raise serializers.ValidationError("At least one part must be specified")

        for i, part in enumerate(value):
            # Required keys
            required_keys = {'name', 'W1', 'W2', 'D', 'thickness'}
            if not required_keys.issubset(part.keys()):
                raise serializers.ValidationError(
                    f"Part {i}: missing required keys {required_keys}"
                )

            # Validate numeric values
            for key in ['W1', 'W2', 'D', 'thickness']:
                if part[key] <= 0:
                    raise serializers.ValidationError(
                        f"Part {i}: {key} must be positive"
                    )

            # Alpha is optional, default to 2.168
            if 'alpha' not in part:
                part['alpha'] = 2.168

            # Quantity is optional, default to 1
            if 'quantity' not in part:
                part['quantity'] = 1

        return value


class CuttingJobSerializer(serializers.ModelSerializer):
    """Serializer for CuttingJob model (read operations)."""

    stock_volume = serializers.FloatField(read_only=True)
    is_complete = serializers.BooleanField(read_only=True)
    duration_seconds = serializers.FloatField(read_only=True)

    class Meta:
        model = CuttingJob
        fields = [
            'id', 'status', 'created_at', 'started_at', 'completed_at',
            'error_message', 'stock_dimensions', 'parts_spec', 'config_params',
            'stock_block', 'results', 'visualization_files', 'report_files',
            'export_files', 'stock_volume', 'is_complete', 'duration_seconds'
        ]
        read_only_fields = [
            'id', 'created_at', 'started_at', 'completed_at', 'results',
            'visualization_files', 'report_files', 'export_files'
        ]


class ConfigurationSerializer(serializers.ModelSerializer):
    """Serializer for Configuration model."""

    summary = serializers.CharField(read_only=True)
    efficiency_percentage = serializers.FloatField(read_only=True)

    class Meta:
        model = Configuration
        fields = [
            'id', 'job', 'configuration_set', 'created_at',
            'primary_part_name', 'merging_plane_order',
            'stock_dimensions', 'parts_spec', 'config_params',
            'total_parts', 'total_volume_used', 'waste_percentage',
            'is_extractable', 'parts_breakdown', 'placements',
            'visualization_file', 'summary', 'efficiency_percentage'
        ]
        read_only_fields = ['created_at']


class Top3ConfigurationsRequestSerializer(serializers.Serializer):
    """
    Serializer for requesting top 3 configurations.
    """
    stock_dimensions = serializers.DictField(
        child=serializers.FloatField(min_value=0),
        help_text="Stock dimensions: {length, width, height} in mm"
    )

    parts = serializers.ListField(
        child=serializers.DictField(),
        help_text="List of available parts with specifications"
    )

    config_params = serializers.DictField(
        required=False,
        default=dict,
        help_text="Configuration parameters (saw_kerf, merging_plane_orders, etc.)"
    )

    top_n = serializers.IntegerField(
        default=3,
        min_value=1,
        max_value=10,
        help_text="Number of top configurations to return (default: 3)"
    )

    def validate_stock_dimensions(self, value):
        """Validate stock dimensions."""
        required_keys = {'length', 'width', 'height'}
        if not required_keys.issubset(value.keys()):
            raise serializers.ValidationError(
                f"stock_dimensions must contain: {required_keys}"
            )
        return value

    def validate_parts(self, value):
        """Validate parts specifications."""
        if not value:
            raise serializers.ValidationError("At least one part must be specified")

        for i, part in enumerate(value):
            required_keys = {'name', 'W1', 'W2', 'D', 'thickness'}
            if not required_keys.issubset(part.keys()):
                raise serializers.ValidationError(
                    f"Part {i}: missing required keys {required_keys}"
                )

            # Alpha is optional
            if 'alpha' not in part:
                part['alpha'] = 2.168

        return value


class Top3ConfigurationsResponseSerializer(serializers.Serializer):
    """
    Serializer for top 3 configurations response.
    """
    rank = serializers.IntegerField()
    primary_part = serializers.CharField()
    merging_plane_order = serializers.CharField()
    total_parts = serializers.IntegerField()
    waste_percentage = serializers.FloatField()
    parts_breakdown = serializers.DictField()
    total_volume_used = serializers.FloatField()
    is_extractable = serializers.BooleanField()
    visualization_file = serializers.CharField()
    summary = serializers.CharField()


class ConfigurationSetSerializer(serializers.ModelSerializer):
    """Serializer for ConfigurationSet model."""

    configurations = ConfigurationSerializer(many=True, read_only=True)
    top_configurations = serializers.SerializerMethodField()

    class Meta:
        model = ConfigurationSet
        fields = [
            'id', 'created_at', 'completed_at', 'status', 'error_message',
            'stock_dimensions', 'parts_spec', 'config_params',
            'configurations', 'top_configurations'
        ]
        read_only_fields = ['created_at', 'completed_at']

    def get_top_configurations(self, obj):
        """Get top 3 configurations."""
        top_configs = obj.configurations.order_by('waste_percentage', '-total_parts')[:3]
        return ConfigurationSerializer(top_configs, many=True).data
