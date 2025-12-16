"""
API-compatible wrapper for pack_manually.py hierarchical packing.

This module provides a clean interface for the /api/configurations/top3/ endpoint.
"""

import os
import sys
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils.config import TrapezoidalPrismSpec
from pack_manually import hierarchical_packing, create_stock_geometry, MixedPlacedPart
from src.visualization import GeometryExporter


def compute_top3_approaches(
    stock_dimensions: Dict[str, float],
    parts: List[Dict[str, Any]],
    config_params: Optional[Dict[str, Any]] = None,
    top_n: int = 3
) -> Dict[str, Any]:
    """
    Compute top N packing approaches using hierarchical packing.

    Args:
        stock_dimensions: Stock block dimensions {length, width, height}
        parts: List of part specifications with W1, W2, D, thickness, alpha
        config_params: Configuration parameters including:
            - saw_kerf: Saw blade thickness (default: 0.0)
            - merging_plane_orders: List of orders to try (default: ["XY-X", "XY-Y", "XZ-X"])
            - primary_parts_to_test: List of part names to test (default: all parts)
        top_n: Number of top approaches to return (default: 3)

    Returns:
        Dictionary with top_approaches, Approach_1, Approach_2, Approach_3, etc.
    """
    if config_params is None:
        config_params = {}

    # Extract configuration
    saw_kerf = config_params.get('saw_kerf', 0.0)
    merging_plane_orders = config_params.get('merging_plane_orders', ['XY-X', 'XY-Y', 'XZ-X'])
    primary_parts_to_test = config_params.get('primary_parts_to_test', None)

    # Convert stock dimensions to tuple
    stock_dims = (
        stock_dimensions['length'],
        stock_dimensions['width'],
        stock_dimensions['height']
    )
    stock_volume = stock_dims[0] * stock_dims[1] * stock_dims[2]

    # Convert parts to TrapezoidalPrismSpec dictionary
    available_parts = {}
    for part in parts:
        name = part['name']
        available_parts[name] = TrapezoidalPrismSpec(
            name=name,
            W1=part['W1'],
            W2=part['W2'],
            D=part['D'],
            thickness=part['thickness'],
            alpha=part.get('alpha', 2.168)
        )

    # Determine which parts to test as primary
    if primary_parts_to_test is None:
        primary_parts_to_test = list(available_parts.keys())

    # Create output directory for visualizations
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("outputs", "visualizations", f"api_top3_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    exporter = GeometryExporter(output_dir)

    # Run hierarchical packing for all combinations
    all_configurations = []

    for primary_part_name in primary_parts_to_test:
        if primary_part_name not in available_parts:
            continue

        for merging_plane_order in merging_plane_orders:
            try:
                # Run hierarchical packing
                result = hierarchical_packing(
                    stock_dims=stock_dims,
                    primary_part_name=primary_part_name,
                    merging_plane_order=merging_plane_order,
                    saw_kerf=saw_kerf,
                    available_parts=available_parts,
                    verbose=0  # Silent mode
                )

                if result[0] is None:
                    continue

                primary_result, sub_blocks, sub_block_results, bounded_region = result

                # Calculate totals
                parts_by_type = dict(primary_result.parts_by_type)
                total_sub_parts = 0

                for sb_result in sub_block_results:
                    if sb_result is not None:
                        part_name, parts_list = sb_result
                        parts_by_type[part_name] = parts_by_type.get(part_name, 0) + len(parts_list)
                        total_sub_parts += len(parts_list)

                # Calculate total volume
                total_volume = 0.0
                for pname, count in parts_by_type.items():
                    if pname in available_parts:
                        spec = available_parts[pname]
                        part_vol = ((spec.W1 + spec.W2) / 2.0) * spec.D * spec.thickness
                        total_volume += part_vol * count

                total_parts = len(primary_result.placed_parts) + total_sub_parts
                waste_percentage = (1 - total_volume / stock_volume) * 100
                efficiency = 100 - waste_percentage

                # Generate visualization
                config_name = f"{primary_part_name}_{merging_plane_order}"
                viz_file = _create_visualization(
                    primary_result,
                    sub_block_results,
                    stock_dims,
                    parts_by_type,
                    waste_percentage,
                    exporter,
                    config_name,
                    timestamp
                )

                # Create configuration summary
                parts_summary = " + ".join([
                    f"{count} {pname}"
                    for pname, count in sorted(parts_by_type.items(), key=lambda x: -x[1])
                ])

                all_configurations.append({
                    'primary_part': primary_part_name,
                    'merging_plane_order': merging_plane_order,
                    'description': f"{parts_summary} (Primary: {primary_part_name}, Order: {merging_plane_order})",
                    'efficiency': round(efficiency, 2),
                    'waste': round(waste_percentage, 2),
                    'total_parts': total_parts,
                    'total_volume_used': total_volume,
                    'parts_breakdown': parts_by_type,
                    'visualization_file': viz_file,
                    'stock_utilization': round(efficiency, 2),
                    'parts_placed': total_parts,
                })

            except Exception as e:
                print(f"Error processing {primary_part_name} with {merging_plane_order}: {e}")
                continue

    # Sort by efficiency (descending) = waste (ascending)
    all_configurations.sort(key=lambda x: x['waste'])

    # Get top N
    top_configs = all_configurations[:top_n]

    # Format response
    response = {
        'top_approaches': [
            {
                'rank': i + 1,
                'description': config['description'],
                'efficiency': config['efficiency'],
                'waste': config['waste'],
                'total_parts': config['total_parts'],
                'parts_breakdown': config['parts_breakdown'],
                'visualization_file': config['visualization_file'],
                'primary_part': config['primary_part'],
                'merging_plane_order': config['merging_plane_order'],
            }
            for i, config in enumerate(top_configs)
        ]
    }

    # Add individual approaches
    for i, config in enumerate(top_configs, 1):
        response[f'Approach_{i}'] = {
            'description': config['description'],
            'efficiency': config['efficiency'],
            'waste': config['waste'],
            'total_parts': config['total_parts'],
            'parts_breakdown': config['parts_breakdown'],
            'visualization_file': config['visualization_file'],
            'stock_utilization': config['stock_utilization'],
            'parts_placed': config['parts_placed'],
        }

    return response


def _create_visualization(
    primary_result,
    sub_block_results: List,
    stock_dims: Tuple[float, float, float],
    parts_by_type: Dict[str, int],
    waste_percentage: float,
    exporter: GeometryExporter,
    config_name: str,
    timestamp: str
) -> str:
    """
    Create Plotly HTML visualization for a configuration.

    Returns:
        Relative path to visualization file
    """
    import cadquery as cq

    stock_geom = create_stock_geometry(*stock_dims)
    geometries = [(stock_geom, "Stock", "lightgray", 0.05)]

    # Add primary parts (green)
    for part in primary_result.placed_parts:
        label = f"{part.part_spec_name}_{part.part_id}"
        geometries.append((part.geometry, label, "#2ecc71", 0.7))

    # Add sub-block parts (different colors per sub-block)
    sub_block_colors = ["#e74c3c", "#3498db", "#9b59b6", "#f39c12", "#1abc9c"]

    for i, result in enumerate(sub_block_results):
        if result is None:
            continue
        part_name, parts_list = result
        color = sub_block_colors[i % len(sub_block_colors)]

        for part in parts_list:
            label = f"{part_name}_{part.part_id}_SB{i+1}"
            geometries.append((part.geometry, label, color, 0.7))

    # Create title
    parts_summary = " + ".join([
        f"{count} {name}"
        for name, count in sorted(parts_by_type.items(), key=lambda x: -x[1])
    ])
    title = f"{parts_summary} | {waste_percentage:.1f}% Waste"

    # Export
    filename = f"config_{config_name}_{timestamp}"
    exporter.export_combined(geometries, filename, title)

    # Return relative path
    full_path = os.path.join(exporter.output_dir, f"{filename}.html")
    return os.path.relpath(full_path, "outputs")


# Test function
if __name__ == "__main__":
    # Test with sample payload
    test_payload = {
        "stock_dimensions": {
            "length": 800,
            "width": 400,
            "height": 2000
        },
        "parts": [
            {
                "name": "G14",
                "W1": 598.8,
                "W2": 566.3,
                "D": 444.5,
                "thickness": 67.8,
                "alpha": 2.094
            }
        ],
        "config_params": {
            "saw_kerf": 0.0,
            "merging_plane_orders": ["XY-X", "XY-Y"]
        },
        "top_n": 3
    }

    print("Testing API wrapper...")
    result = compute_top3_approaches(
        stock_dimensions=test_payload['stock_dimensions'],
        parts=test_payload['parts'],
        config_params=test_payload['config_params'],
        top_n=test_payload['top_n']
    )

    import json
    print("\nResult:")
    print(json.dumps(result, indent=2))
