"""
Service layer for interfacing with the existing computational engine in src/.

This module provides a clean abstraction between Django and the core optimization logic.
"""

import os
import sys
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from pathlib import Path

# Add parent directory to path to import from src/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.geometry.data_classes import StockBlock as SrcStockBlock, TrapezoidalPrism
from src.utils.config import STOCK_BLOCKS, PART_SPECS, TrapezoidalPrismSpec
from src.packing.trapezoid_geometry import TrapezoidGeometry
from src.packing.orientation_explorer import OrientationExplorer
from src.visualization import GeometryExporter

# Import the hierarchical packing function from step5_mixed_parts.py
# We'll need to refactor it into an importable function
from step5_mixed_parts import (
    hierarchical_packing,
    MixedPlacedPart,
    MixedPackingResult,
    create_stock_geometry,
)


class CuttingOptimizationService:
    """
    Service for running cutting optimization computations.

    This service interfaces with the existing src/ modules and step5_mixed_parts.py
    to perform the actual optimization work.
    """

    def __init__(self, output_base_dir: str = "outputs"):
        """
        Initialize the service.

        Args:
            output_base_dir: Base directory for output files
        """
        self.output_base_dir = output_base_dir
        self.visualizations_dir = os.path.join(output_base_dir, "visualizations")
        self.reports_dir = os.path.join(output_base_dir, "reports")
        self.exports_dir = os.path.join(output_base_dir, "exports")

        # Create directories if they don't exist
        os.makedirs(self.visualizations_dir, exist_ok=True)
        os.makedirs(self.reports_dir, exist_ok=True)
        os.makedirs(self.exports_dir, exist_ok=True)

    def run_cutting_job(
        self,
        stock_dimensions: Dict[str, float],
        parts_spec: List[Dict[str, Any]],
        config_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Run a cutting optimization job.

        Args:
            stock_dimensions: Dict with 'length', 'width', 'height' in mm
            parts_spec: List of part specifications, each with:
                - name: Part name
                - quantity: Number of parts desired
                - W1, W2, D, thickness, alpha: Part dimensions
            config_params: Optional configuration parameters:
                - saw_kerf: Saw blade thickness (default: 0.0)
                - max_iterations: Max optimization iterations (default: 50)
                - merging_plane_order: Plane order (default: "XY-X")

        Returns:
            Dictionary with optimization results
        """
        if config_params is None:
            config_params = {}

        saw_kerf = config_params.get('saw_kerf', 0.0)
        merging_plane_order = config_params.get('merging_plane_order', 'XY-X')

        # Convert stock dimensions to tuple
        stock_dims = (
            stock_dimensions['length'],
            stock_dimensions['width'],
            stock_dimensions['height']
        )

        # Convert parts spec to TrapezoidalPrismSpec dict
        parts_dict = {}
        for part in parts_spec:
            name = part['name']
            parts_dict[name] = TrapezoidalPrismSpec(
                name=name,
                W1=part['W1'],
                W2=part['W2'],
                D=part['D'],
                thickness=part['thickness'],
                alpha=part.get('alpha', 2.168)
            )

        # Select primary part (use the one with highest quantity or first one)
        if parts_spec:
            primary_part_name = max(parts_spec, key=lambda p: p.get('quantity', 1))['name']
        else:
            raise ValueError("No parts specified")

        # Run hierarchical packing
        result = hierarchical_packing(
            stock_dims=stock_dims,
            primary_part_name=primary_part_name,
            merging_plane_order=merging_plane_order,
            saw_kerf=saw_kerf,
            available_parts=parts_dict,
            verbose=0  # Silent mode for API
        )

        if result[0] is None:
            raise ValueError(f"No valid packing found for {primary_part_name}")

        primary_result, sub_blocks, sub_block_results, bounded_region = result

        # Calculate totals
        parts_by_type = dict(primary_result.parts_by_type)
        total_sub_parts = 0

        for sb_result in sub_block_results:
            if sb_result is not None:
                part_name, parts = sb_result
                parts_by_type[part_name] = parts_by_type.get(part_name, 0) + len(parts)
                total_sub_parts += len(parts)

        total_volume = 0.0
        for pname, count in parts_by_type.items():
            if pname in parts_dict:
                spec = parts_dict[pname]
                part_vol = ((spec.W1 + spec.W2) / 2.0) * spec.D * spec.thickness
                total_volume += part_vol * count

        total_parts = len(primary_result.placed_parts) + total_sub_parts
        stock_volume = stock_dims[0] * stock_dims[1] * stock_dims[2]
        waste_percentage = (1 - total_volume / stock_volume) * 100

        # Generate visualization
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        job_dir = os.path.join(self.visualizations_dir, f"job_{timestamp}")
        os.makedirs(job_dir, exist_ok=True)

        exporter = GeometryExporter(job_dir)
        visualization_file = self._create_visualization(
            primary_result,
            sub_block_results,
            stock_dims,
            parts_by_type,
            waste_percentage,
            exporter,
            f"job_complete_{timestamp}"
        )

        return {
            'total_parts_placed': total_parts,
            'waste_percentage': waste_percentage,
            'total_volume_used': total_volume,
            'stock_volume': stock_volume,
            'is_extractable': True,  # Hierarchical packing ensures extractability
            'parts_breakdown': parts_by_type,
            'primary_part': primary_part_name,
            'merging_plane_order': merging_plane_order,
            'visualization_files': [visualization_file],
            'bounded_region': {
                'min_x': bounded_region[0],
                'min_y': bounded_region[1],
                'min_z': bounded_region[2],
                'max_x': bounded_region[3],
                'max_y': bounded_region[4],
                'max_z': bounded_region[5],
            },
            'sub_blocks_count': len(sub_blocks),
        }

    def compute_top_configurations(
        self,
        stock_dimensions: Dict[str, float],
        parts_spec: List[Dict[str, Any]],
        config_params: Optional[Dict[str, Any]] = None,
        top_n: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Compute top N packing configurations.

        This method runs hierarchical packing for multiple primary parts and
        merging plane orders, then returns the top N by efficiency.

        Args:
            stock_dimensions: Stock block dimensions
            parts_spec: List of available parts
            config_params: Configuration parameters:
                - saw_kerf: Saw blade thickness
                - merging_plane_orders: List of merging plane orders to try
                - primary_parts_to_test: List of part names to test as primary (optional)
            top_n: Number of top configurations to return (default: 3)

        Returns:
            List of top N configuration dictionaries
        """
        if config_params is None:
            config_params = {}

        saw_kerf = config_params.get('saw_kerf', 0.0)
        merging_plane_orders = config_params.get('merging_plane_orders', ['XY-X', 'XY-Y', 'XZ-X'])

        # Convert stock dimensions
        stock_dims = (
            stock_dimensions['length'],
            stock_dimensions['width'],
            stock_dimensions['height']
        )

        # Convert parts spec to TrapezoidalPrismSpec dict
        parts_dict = {}
        for part in parts_spec:
            name = part['name']
            parts_dict[name] = TrapezoidalPrismSpec(
                name=name,
                W1=part['W1'],
                W2=part['W2'],
                D=part['D'],
                thickness=part['thickness'],
                alpha=part.get('alpha', 2.168)
            )

        # Determine which parts to test as primary
        primary_parts_to_test = config_params.get('primary_parts_to_test')
        if primary_parts_to_test is None:
            primary_parts_to_test = sorted(parts_dict.keys())

        # Run optimization for all combinations
        all_configurations = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config_set_dir = os.path.join(self.visualizations_dir, f"configset_{timestamp}")
        os.makedirs(config_set_dir, exist_ok=True)
        exporter = GeometryExporter(config_set_dir)

        for primary_part_name in primary_parts_to_test:
            if primary_part_name not in parts_dict:
                continue

            for merging_plane_order in merging_plane_orders:
                try:
                    result = hierarchical_packing(
                        stock_dims=stock_dims,
                        primary_part_name=primary_part_name,
                        merging_plane_order=merging_plane_order,
                        saw_kerf=saw_kerf,
                        available_parts=parts_dict,
                        verbose=0
                    )

                    if result[0] is None:
                        continue

                    primary_result, sub_blocks, sub_block_results, bounded_region = result

                    # Calculate totals
                    parts_by_type = dict(primary_result.parts_by_type)
                    total_sub_parts = 0

                    for sb_result in sub_block_results:
                        if sb_result is not None:
                            part_name, parts = sb_result
                            parts_by_type[part_name] = parts_by_type.get(part_name, 0) + len(parts)
                            total_sub_parts += len(parts)

                    total_volume = 0.0
                    for pname, count in parts_by_type.items():
                        if pname in parts_dict:
                            spec = parts_dict[pname]
                            part_vol = ((spec.W1 + spec.W2) / 2.0) * spec.D * spec.thickness
                            total_volume += part_vol * count

                    total_parts = len(primary_result.placed_parts) + total_sub_parts
                    stock_volume = stock_dims[0] * stock_dims[1] * stock_dims[2]
                    waste_percentage = (1 - total_volume / stock_volume) * 100

                    # Generate visualization
                    config_name = f"{primary_part_name}_{merging_plane_order}"
                    visualization_file = self._create_visualization(
                        primary_result,
                        sub_block_results,
                        stock_dims,
                        parts_by_type,
                        waste_percentage,
                        exporter,
                        f"config_{config_name}_{timestamp}"
                    )

                    all_configurations.append({
                        'primary_part': primary_part_name,
                        'merging_plane_order': merging_plane_order,
                        'total_parts': total_parts,
                        'total_volume_used': total_volume,
                        'waste_percentage': waste_percentage,
                        'is_extractable': True,
                        'parts_breakdown': parts_by_type,
                        'visualization_file': visualization_file,
                        'summary': self._generate_summary(parts_by_type, waste_percentage),
                    })

                except Exception as e:
                    # Skip failed configurations
                    print(f"Error processing {primary_part_name} with {merging_plane_order}: {e}")
                    continue

        # Sort by waste percentage (ascending) and return top N
        all_configurations.sort(key=lambda x: x['waste_percentage'])
        top_configs = all_configurations[:top_n]

        # Add rank to each configuration
        for i, config in enumerate(top_configs, 1):
            config['rank'] = i

        return top_configs

    def _create_visualization(
        self,
        primary_result: MixedPackingResult,
        sub_block_results: List,
        stock_dims: Tuple[float, float, float],
        parts_by_type: Dict[str, int],
        waste_percentage: float,
        exporter: GeometryExporter,
        filename: str
    ) -> str:
        """
        Create Plotly HTML visualization.

        Returns:
            Relative path to visualization file
        """
        import cadquery as cq

        stock_geom = create_stock_geometry(*stock_dims)
        geometries = [(stock_geom, "Stock", "lightgray", 0.05)]

        # Add primary parts
        for part in primary_result.placed_parts:
            label = f"{part.part_spec_name}_{part.part_id}"
            geometries.append((part.geometry, label, "#2ecc71", 0.7))

        # Add sub-block parts
        sub_block_colors = ["#e74c3c", "#3498db", "#9b59b6", "#f39c12", "#1abc9c"]
        for i, result in enumerate(sub_block_results):
            if result is None:
                continue
            part_name, parts = result
            color = sub_block_colors[i % len(sub_block_colors)]

            for part in parts:
                label = f"{part_name}_{part.part_id}_SB{i+1}"
                geometries.append((part.geometry, label, color, 0.7))

        # Create title
        parts_summary = " + ".join([
            f"{count} {name}" for name, count in sorted(parts_by_type.items(), key=lambda x: -x[1])
        ])
        title = f"{parts_summary} | {waste_percentage:.1f}% Waste"

        # Export visualization
        exporter.export_combined(geometries, filename, title)

        # Return relative path from outputs directory
        full_path = os.path.join(exporter.output_dir, f"{filename}.html")
        return os.path.relpath(full_path, self.output_base_dir)

    def _generate_summary(self, parts_by_type: Dict[str, int], waste_percentage: float) -> str:
        """Generate human-readable summary."""
        parts_str = " + ".join([
            f"{count} {name}" for name, count in sorted(parts_by_type.items(), key=lambda x: -x[1])
        ])
        return f"{parts_str} and {waste_percentage:.1f}% Waste"


# Singleton instance
_service_instance = None


def get_cutting_service() -> CuttingOptimizationService:
    """Get or create the cutting optimization service singleton."""
    global _service_instance
    if _service_instance is None:
        _service_instance = CuttingOptimizationService()
    return _service_instance
