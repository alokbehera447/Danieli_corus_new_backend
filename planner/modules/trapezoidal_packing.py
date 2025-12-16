# planner/modules/trapezoidal_packing.py
"""
API wrapper for the trapezoidal packing functions
"""
import os
import sys

# Add path to trapezoidal_packing directory
current_dir = os.path.dirname(__file__)
project_root = os.path.dirname(os.path.dirname(current_dir))
trapezoidal_dir = os.path.join(project_root, 'trapezoidal_packing')
sys.path.insert(0, trapezoidal_dir)

# Import your existing packing functions
from fill import fill_the_box, draw, get_scrap_vol
from edges import get_type
from scrap import get_scrap_volume_of_type1, get_scrap_volume_of_type2, get_scrap_volume_of_type3, get_scrap_volume_of_type4


def run_trapezoidal_packing(stock_dimensions, parts_list, buffer=2.0):
    """
    Run trapezoidal packing algorithm
    
    Args:
        stock_dimensions: dict with length, width, height
        parts_list: list of part dictionaries
        buffer: spacing between parts
    
    Returns:
        dict with results
    """
    try:
        # This is a simplified version - you'll need to adapt it to your actual code
        results = {
            'success': True,
            'efficiency': 0,
            'total_parts': 0,
            'waste': 0,
            'details': []
        }
        
        # Here you would integrate with your actual packing code
        # For now, return dummy results
        return results
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }