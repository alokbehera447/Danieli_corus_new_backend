"""
Main interface for trapezoidal prism packing algorithm.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trapezoidal_packing.fill import fill_the_box, get_scrap_vol, draw, place_box
from trapezoidal_packing.edges import (
    pre_z_edges, pre_x_edges, pre_y_edges, 
    connect_lines_same_x, process_groups_yxz, group_by_common_y,
    group_by_common_x, process_groups, y_edges_process, x_edges_process,
    get_type
)
from trapezoidal_packing.scrap import (
    get_scrap_volume_of_type1, get_scrap_volume_of_type2,
    get_scrap_volume_of_type3, get_scrap_volume_of_type4
)

def pack_trapezoidal_prisms(stock_dimensions, parts, config_params, top_n=3):
    """
    Main function to pack trapezoidal prisms into a stock block.
    
    Args:
        stock_dimensions: dict with length, width, height
        parts: list of parts with bottom_length, top_length, width, height, quantity
        config_params: dict with saw_kerf, buffer_spacing, etc.
        top_n: number of top approaches to return
    
    Returns:
        dict with packing configurations
    """
    try:
        # Extract parameters
        length = stock_dimensions.get('length', 2000)
        width = stock_dimensions.get('width', 500)
        height = stock_dimensions.get('height', 500)
        
        buffer_spacing = config_params.get('buffer_spacing', 2.0)
        saw_kerf = config_params.get('saw_kerf', 0.0)
        
        # Create a dummy trapezoidal prism object (you need to adapt this)
        # Based on your algorithm, you need to create prisms from parts
        configurations = []
        
        # This is where you'll integrate your packing algorithm
        # For now, returning a dummy response structure
        for i in range(min(top_n, 3)):
            config = {
                'rank': i + 1,
                'primary_part': parts[0]['name'] if parts else 'Unknown',
                'merging_plane_order': 'XY-X',
                'total_parts': sum(p['quantity'] for p in parts),
                'waste_percentage': 10.0 + i * 5,
                'parts_breakdown': {p['name']: p['quantity'] for p in parts},
                'total_volume_used': length * width * height * 0.8,
                'is_extractable': True,
                'visualization_file': f'visualization_{i+1}.html',
                'description': f'Approach {i+1} using XY-X plane order',
                'efficiency': 90.0 - i * 5,
                'packing_count': sum(p['quantity'] for p in parts),
                'scrap_volumes': [],
                'stock_types_used': [f"{width}×{height}×{length}"]
            }
            configurations.append(config)
        
        return {
            'success': True,
            'configurations': configurations,
            'stock_dimensions': stock_dimensions,
            'total_parts_processed': len(parts),
            'total_blocks': sum(p['quantity'] for p in parts)
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }