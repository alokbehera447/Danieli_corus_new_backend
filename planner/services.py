"""
Service layer for cutting optimization with trapezoidal prism packing.
"""

import os
import sys
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from trapezoidal_packing.pack import pack_trapezoidal_prisms
    from trapezoidal_packing.fill import draw
except ImportError:
    pack_trapezoidal_prisms = None
    print("Warning: trapezoidal_packing module not found.")

class CuttingOptimizationService:
    """
    Service for running cutting optimization computations with trapezoidal prism packing.
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
        Run a cutting optimization job using trapezoidal prism packing.

        Args:
            stock_dimensions: Dict with 'length', 'width', 'height' in mm
            parts_spec: List of part specifications, each with:
                - name: Part name
                - quantity: Number of parts desired
                - W1, W2, D, thickness, alpha: Part dimensions (old format)
                - OR bottom_length, top_length, width, height (new format)
            config_params: Optional configuration parameters:
                - saw_kerf: Saw blade thickness (default: 0.0)
                - buffer_spacing: Spacing between parts (default: 2.0)
                - merging_plane_order: Plane order (default: "XY-X")

        Returns:
            Dictionary with optimization results
        """
        if config_params is None:
            config_params = {}

        # Convert parts from old format to new format if needed
        converted_parts = []
        for part in parts_spec:
            # Check if part uses old format (W1, W2, D, thickness)
            if all(key in part for key in ['W1', 'W2', 'D', 'thickness']):
                # Convert old format to new format
                converted_part = {
                    'name': part['name'],
                    'bottom_length': part['W1'],      # W1 -> bottom_length
                    'top_length': part['W2'],         # W2 -> top_length
                    'width': part['D'],               # D -> width
                    'height': part['thickness'],      # thickness -> height
                    'quantity': part.get('quantity', 1),
                    'alpha': part.get('alpha', 2.168)
                }
            else:
                # Already in new format
                converted_part = part.copy()
            
            # Ensure quantity exists
            if 'quantity' not in converted_part:
                converted_part['quantity'] = 1
            
            converted_parts.append(converted_part)

        try:
            if pack_trapezoidal_prisms is None:
                raise ImportError("trapezoidal_packing module not found")

            # Call trapezoidal packing algorithm
            results = pack_trapezoidal_prisms(
                stock_dimensions=stock_dimensions,
                parts=converted_parts,
                config_params=config_params,
                top_n=3
            )

            if not results.get('success', False):
                raise Exception(results.get('error', 'Optimization failed'))

            configurations = results.get('configurations', [])
            
            # Generate visualization files
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            visualization_files = []
            
            for i, config in enumerate(configurations):
                # Create visualization for each configuration
                vis_file = self._create_visualization(
                    config=config,
                    stock_dimensions=stock_dimensions,
                    timestamp=timestamp,
                    config_index=i
                )
                if vis_file:
                    visualization_files.append(vis_file)
                    # Update config with visualization file path
                    config['visualization_file'] = vis_file

            return {
                'success': True,
                'configurations': configurations,
                'visualization_files': visualization_files,
                'total_parts_processed': len(parts_spec),
                'total_blocks': sum(p.get('quantity', 1) for p in parts_spec),
                'stock_dimensions': stock_dimensions,
                'waste_percentage': configurations[0].get('waste', 0) if configurations else 0,
                'efficiency': configurations[0].get('efficiency', 0) if configurations else 0
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'configurations': [],
                'visualization_files': []
            }

    def compute_top_configurations(
        self,
        stock_dimensions: Dict[str, float],
        parts_spec: List[Dict[str, Any]],
        config_params: Optional[Dict[str, Any]] = None,
        top_n: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Compute top N packing configurations using trapezoidal prism packing.

        Args:
            stock_dimensions: Stock block dimensions
            parts_spec: List of available parts
            config_params: Configuration parameters:
                - saw_kerf: Saw blade thickness
                - buffer_spacing: Spacing between parts
                - merging_plane_orders: List of merging plane orders to try
            top_n: Number of top configurations to return (default: 3)

        Returns:
            List of top N configuration dictionaries
        """
        if config_params is None:
            config_params = {}

        # Convert parts from old format to new format if needed
        converted_parts = []
        for part in parts_spec:
            # Check if part uses old format (W1, W2, D, thickness)
            if all(key in part for key in ['W1', 'W2', 'D', 'thickness']):
                # Convert old format to new format
                converted_part = {
                    'name': part['name'],
                    'bottom_length': part['W1'],      # W1 -> bottom_length
                    'top_length': part['W2'],         # W2 -> top_length
                    'width': part['D'],               # D -> width
                    'height': part['thickness'],      # thickness -> height
                    'quantity': part.get('quantity', 1),
                    'alpha': part.get('alpha', 2.168)
                }
            else:
                # Already in new format
                converted_part = part.copy()
            
            # Ensure quantity exists
            if 'quantity' not in converted_part:
                converted_part['quantity'] = 1
            
            converted_parts.append(converted_part)

        try:
            if pack_trapezoidal_prisms is None:
                raise ImportError("trapezoidal_packing module not found")

            # Call trapezoidal packing algorithm
            results = pack_trapezoidal_prisms(
                stock_dimensions=stock_dimensions,
                parts=converted_parts,
                config_params=config_params,
                top_n=top_n
            )

            if not results.get('success', False):
                raise Exception(results.get('error', 'Optimization failed'))

            configurations = results.get('configurations', [])
            
            # Generate visualization files
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            for i, config in enumerate(configurations):
                # Create visualization for each configuration
                vis_file = self._create_visualization(
                    config=config,
                    stock_dimensions=stock_dimensions,
                    timestamp=timestamp,
                    config_index=i
                )
                if vis_file:
                    config['visualization_file'] = vis_file

            # Add rank to each configuration
            for i, config in enumerate(configurations, 1):
                config['rank'] = i
                if 'description' not in config:
                    config['description'] = f'Optimized packing approach {i}'
                if 'efficiency' not in config:
                    config['efficiency'] = 100 - config.get('waste', 0)
                if 'parts_breakdown' not in config:
                    # Create simple parts breakdown
                    parts_breakdown = {}
                    for part in converted_parts:
                        parts_breakdown[part['name']] = part.get('quantity', 1)
                    config['parts_breakdown'] = parts_breakdown

            return configurations

        except Exception as e:
            print(f"Error in compute_top_configurations: {e}")
            return []

    def _create_visualization(
        self,
        config: Dict[str, Any],
        stock_dimensions: Dict[str, float],
        timestamp: str,
        config_index: int
    ) -> str:
        """
        Create visualization for a configuration.
        
        Note: This is a placeholder. You'll need to integrate your 
        trapezoidal_packing.fill.draw() function here.
        
        Returns:
            Path to visualization file
        """
        try:
            # Create a unique filename
            filename = f"visualization_{timestamp}_config{config_index+1}.html"
            filepath = os.path.join(self.visualizations_dir, filename)
            
            # For now, create a simple HTML file
            # You should replace this with your actual drawing function
            with open(filepath, 'w') as f:
                f.write(self._generate_html_visualization(config, stock_dimensions))
            
            # Return relative path from outputs directory
            return os.path.relpath(filepath, self.output_base_dir)
            
        except Exception as e:
            print(f"Error creating visualization: {e}")
            return ""

    def _generate_html_visualization(self, config: Dict[str, Any], stock_dimensions: Dict[str, float]) -> str:
        """Generate simple HTML visualization."""
        waste = config.get('waste', 0)
        efficiency = config.get('efficiency', 100 - waste)
        primary_part = config.get('primary_part', 'Unknown')
        merging_plane_order = config.get('merging_plane_order', 'XY-X')
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Trapezoidal Prism Packing Visualization</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .container {{ max-width: 800px; margin: 0 auto; }}
                .header {{ background: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .stats {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 20px 0; }}
                .stat-box {{ background: #f8f9fa; padding: 15px; border-radius: 5px; }}
                .stat-value {{ font-size: 24px; font-weight: bold; color: #007bff; }}
                .visualization {{ 
                    background: #fff; 
                    border: 1px solid #ddd; 
                    padding: 20px; 
                    text-align: center;
                    min-height: 400px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>Trapezoidal Prism Packing Visualization</h1>
                    <p>Stock Dimensions: {stock_dimensions['length']} × {stock_dimensions['width']} × {stock_dimensions['height']} mm</p>
                </div>
                
                <div class="stats">
                    <div class="stat-box">
                        <div class="stat-label">Efficiency</div>
                        <div class="stat-value">{efficiency:.1f}%</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-label">Waste</div>
                        <div class="stat-value">{waste:.1f}%</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-label">Primary Part</div>
                        <div class="stat-value">{primary_part}</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-label">Merging Plane</div>
                        <div class="stat-value">{merging_plane_order}</div>
                    </div>
                </div>
                
                <div class="visualization">
                    <div>
                        <h3>3D Visualization</h3>
                        <p>Interactive 3D visualization would appear here.</p>
                        <p><em>Note: This is a placeholder. Actual visualization requires Plotly integration.</em></p>
                    </div>
                </div>
                
                <div style="margin-top: 20px; padding: 15px; background: #e8f4fd; border-radius: 5px;">
                    <h3>Configuration Details</h3>
                    <pre style="background: white; padding: 15px; border-radius: 3px; overflow: auto;">
{self._format_config_details(config)}
                    </pre>
                </div>
            </div>
        </body>
        </html>
        """
        return html

    def _format_config_details(self, config: Dict[str, Any]) -> str:
        """Format configuration details for display."""
        import json
        # Remove visualization file path for cleaner display
        display_config = config.copy()
        display_config.pop('visualization_file', None)
        return json.dumps(display_config, indent=2, default=str)


# Singleton instance
_service_instance = None


def get_cutting_service() -> CuttingOptimizationService:
    """Get or create the cutting optimization service singleton."""
    global _service_instance
    if _service_instance is None:
        _service_instance = CuttingOptimizationService()
    return _service_instance