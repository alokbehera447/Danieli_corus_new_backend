# planner/modules/packing_module.py
"""
Consolidated packing module that integrates with the trapezoidal_packing directory
"""
import os
import sys
import math
import numpy as np
import pandas as pd
from datetime import datetime
from django.conf import settings

# Add the trapezoidal_packing directory to Python path
trapezoidal_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', 'trapezoidal_packing')
sys.path.insert(0, trapezoidal_dir)

# Now you can import from your existing files
try:
    from trapezoidal_packing.edges import *
    from trapezoidal_packing.scrap import *
    from trapezoidal_packing.fill import *
    # from pack import *  # If you have a pack.py file
except ImportError as e:
    print(f"Warning: Could not import trapezoidal packing modules: {e}")
    # Define placeholder functions if needed
    def fill_the_box(*args, **kwargs):
        raise ImportError("fill_the_box not available")
    def get_scrap_vol(*args, **kwargs):
        raise ImportError("get_scrap_vol not available")


class Prisms:
    """Wrapper for your existing Prism class"""
    def __init__(self, code, size, quantity, roundingoff=2):
        self.code = code
        self.quantity = quantity
        self.prism_left = quantity
        self.roundingoff = roundingoff
        
        if len(size) == 4:
            self.size = size
        elif len(size) == 3:
            self.size = [size[0]] + size
        else:
            raise ValueError("Size must have 3 or 4 dimensions")
        
        self.bottom_length = self.size[0]
        self.top_length = self.size[1]
        self.width = self.size[2]
        self.height = self.size[3]
        
        # Calculate angle
        height = self.height
        length = (self.bottom_length - self.top_length) / 2
        angle_rad = math.atan(length / height) if height > 0 else 0
        self.angle = math.degrees(angle_rad)
        
        # Calculate volume
        self.volume = 0.5 * (self.bottom_length + self.top_length) * self.width * self.height
        print(f"[Prisms] Created prism {code}: bottom={self.bottom_length}, top={self.top_length}, width={self.width}, height={self.height}, volume={self.volume}")
    
    def update_prism_left(self, used_quantity):
        self.prism_left = self.prism_left - used_quantity
    
    def __repr__(self):
        return f"Prism({self.code}, bottom={self.bottom_length}, top={self.top_length}, width={self.width}, height={self.height}, qty={self.quantity})"


class OptimizationEngine:
    def __init__(self, stock_dimensions, parts_data, buffer_spacing=2.0):
        self.stock_dimensions = stock_dimensions
        self.parts_data = parts_data
        self.buffer_spacing = buffer_spacing
        self.results = {}
        print(f"[OptimizationEngine] Initialized with {len(parts_data)} parts")
    
    def process_excel_data(self, excel_file):
        """Process uploaded Excel file"""
        try:
            if excel_file.name.endswith('.csv'):
                df = pd.read_csv(excel_file)
            else:
                df = pd.read_excel(excel_file, engine='openpyxl')
            
            # Clean column names
            df.columns = [str(col).strip() for col in df.columns]
            
            # Map column names
            column_mapping = {}
            column_variations = {
                'MARK': ['MARK', 'mark', 'Mark', 'Block ID', 'BLOCK ID'],
                'Bottom Length': ['Bottom Length', 'BottomLength', 'Bottom_Length'],
                'Top Length': ['Top Length', 'TopLength', 'Top_Length'],
                'Width': ['Width', 'width', 'WIDTH'],
                'Height': ['Height', 'height', 'HEIGHT'],
                'Nos': ['Nos', 'nos', 'NOS', 'Quantity', 'quantity']
            }
            
            for standard_name, variations in column_variations.items():
                for col in df.columns:
                    if str(col).lower() in [v.lower() for v in variations]:
                        column_mapping[col] = standard_name
                        break
            
            df.rename(columns=column_mapping, inplace=True)
            
            # Process rows
            processed_data = []
            for _, row in df.iterrows():
                if pd.isna(row.get('MARK', '')):
                    continue
                
                block_data = {
                    'MARK': str(row.get('MARK', '')).strip(),
                    'Bottom Length': float(row.get('Bottom Length', 0)),
                    'Top Length': float(row.get('Top Length', 0)),
                    'Width': float(row.get('Width', 0)),
                    'Height': float(row.get('Height', 0)),
                    'Nos': int(float(row.get('Nos', 1)))
                }
                processed_data.append(block_data)
            
            return processed_data
            
        except Exception as e:
            raise Exception(f"Error processing Excel file: {str(e)}")
    
    def create_prism_objects(self, blocks_data):
        """Create Prism objects from block data"""
        prisms = []
        for block in blocks_data:
            try:
                size = [
                    block['Bottom Length'],
                    block['Top Length'],
                    block['Width'],
                    block['Height']
                ]
                prism = Prisms(
                    code=block['MARK'],
                    size=size,
                    quantity=block['Nos']
                )
                prisms.append(prism)
            except Exception as e:
                print(f"[create_prism_objects] Error creating prism for {block.get('MARK', 'Unknown')}: {e}")
                continue
        return prisms
    
    def get_fallback_results(self, error_message=""):
        """Get fallback results when packing fails"""
        print(f"[get_fallback_results] Using fallback results. Error: {error_message}")
        
        # Calculate simple efficiency based on volumes
        total_volume = 0
        total_parts = 0
        
        for part in self.parts_data:
            part_volume = 0.5 * (part['Bottom Length'] + part['Top Length']) * part['Width'] * part['Height']
            total_volume += part_volume * part['Nos']
            total_parts += part['Nos']
        
        stock_volume = (
            self.stock_dimensions['length'] *
            self.stock_dimensions['width'] *
            self.stock_dimensions['height']
        )
        
        efficiency = (total_volume / stock_volume * 100) if stock_volume > 0 else 0
        
        return {
            'success': True,
            'efficiency': round(efficiency, 2),
            'total_parts_packed': total_parts,
            'total_blocks': len(self.parts_data),
            'total_volume_used': round(total_volume, 2),
            'stock_volume': stock_volume,
            'waste_percentage': round(100 - efficiency, 2),
            'message': f'Fallback calculation: {efficiency:.2f}% efficiency. {error_message}'
        }
    
# In packing_module.py, update the optimize() method

# In packing_module.py, update the optimize() method:

    def optimize(self, selected_blocks=None, parent_block_sizes=None):
        """Run optimization using the complete orchestrator"""
        try:
            print(f"[optimize] ===== STARTING COMPLETE OPTIMIZATION =====")
            
            # Filter blocks if specified
            if selected_blocks:
                blocks_to_optimize = [b for b in self.parts_data 
                                    if b['MARK'] in selected_blocks]
            else:
                blocks_to_optimize = self.parts_data
            
            if not blocks_to_optimize:
                raise ValueError("No blocks to optimize")
            
            # Import the orchestrator
            try:
                from .packing_orchestrator import Prisms as OrchestratorPrisms, run_final_code, get_block_details
                print(f"[optimize] Successfully imported orchestrator")
            except ImportError as e:
                print(f"[optimize] ERROR: Could not import orchestrator: {e}")
                return self.get_fallback_results(f"Orchestrator import failed: {e}")
            
            all_prisms = []
            for block in blocks_to_optimize:
                try:
                    size = [
                        block['Bottom Length'],
                        block['Top Length'],
                        block['Width'],
                        block['Height']
                    ]
                    prism = OrchestratorPrisms(
                        code=block['MARK'],
                        size=size,
                        quantity=block['Nos']
                    )
                    all_prisms.append(prism)
                except Exception as e:
                    print(f"[optimize] Error creating prism {block.get('MARK', 'Unknown')}: {e}")
                    continue
                    
            if not all_prisms:
                raise ValueError("No valid prism objects created")
            
            print(f"[optimize] Created {len(all_prisms)} prism objects")
            
            # Use provided parent_block_sizes or fallback to single size
            if parent_block_sizes:
                parent_block_sizes = parent_block_sizes
            else:
                # Fallback: create single stock size
                parent_block_sizes = [
                    [self.stock_dimensions['length'], 
                     self.stock_dimensions['width'], 
                     self.stock_dimensions['height']]
                ]
            
            print(f"[optimize] Parent block sizes: {parent_block_sizes}")
            print(f"[optimize] Buffer spacing: {self.buffer_spacing}")
            print(f"[optimize] Running complete packing algorithm...")
            
            # Run the complete packing algorithm
            try:
                helper = run_final_code(
                    all_prisms=all_prisms,
                    buffer=self.buffer_spacing,
                    parent_block_sizes=parent_block_sizes
                )
            except Exception as e:
                print(f"[optimize] ERROR in run_final_code: {e}")
                import traceback
                traceback.print_exc()
                return self.get_fallback_results(f"run_final_code failed: {e}")
            
            # Get detailed results
            block_details = get_block_details(helper)
            
            print(f"[optimize] Packing complete!")
            print(f"[optimize] Total blocks created: {len(helper.all_big_blocks)}")
            print(f"[optimize] Total scraps: {len(helper.all_scrap)}")
            
            # Calculate totals
            total_parts_packed = 0
            total_prism_volume = 0
            
            for prism in all_prisms:
                packed = prism.quantity - prism.prism_left
                total_parts_packed += packed
                total_prism_volume += prism.volume * packed
            
            # Calculate total stock volume
            total_stock_volume = 0
            for block in helper.all_big_blocks:
                total_stock_volume += block.volume
            
            # Calculate efficiency
            if total_stock_volume > 0:
                efficiency = (total_prism_volume / total_stock_volume) * 100
            else:
                efficiency = 0
            
            # Prepare results
            self.results = {
                'success': True,
                'efficiency': round(efficiency, 2),
                'total_parts_packed': total_parts_packed,
                'total_blocks_created': len(helper.all_big_blocks),
                'total_stock_volume': round(total_stock_volume, 2),
                'total_prism_volume': round(total_prism_volume, 2),
                'waste_percentage': round(100 - efficiency, 2),
                'block_details': block_details,
                'message': f'Created {len(helper.all_big_blocks)} stock blocks with {efficiency:.2f}% efficiency'
            }
            
            return self.results
            
        except Exception as e:
            print(f"[optimize] ERROR: {str(e)}")
            import traceback
            traceback.print_exc()
            return self.get_fallback_results(f"Error: {str(e)}")

    # API wrapper function
def pack_trapezoidal_prisms(stock_dimensions, parts, config_params=None, top_n=3):
    """
    Main API function for trapezoidal prism packing
    """
    try:
        # Convert parts to the format expected by OptimizationEngine
        parts_data = []
        for part in parts:
            parts_data.append({
                'MARK': part.get('name', 'Unknown'),
                'Bottom Length': part.get('bottom_length', 0),
                'Top Length': part.get('top_length', 0),
                'Width': part.get('width', 0),
                'Height': part.get('height', 0),
                'Nos': part.get('quantity', 1)
            })
        
        # Create engine
        engine = OptimizationEngine(
            stock_dimensions=stock_dimensions,
            parts_data=parts_data,
            buffer_spacing=config_params.get('buffer_spacing', 2.0) if config_params else 2.0
        )
        
        # Run optimization
        results = engine.optimize()
        
        # Format response for API
        configurations = []
        for i in range(min(top_n, 1)):  # For now, single configuration
            config = {
                'rank': i + 1,
                'primary_part': parts[0]['name'] if parts else 'Unknown',
                'merging_plane_order': config_params.get('merging_plane_orders', ['XY-X'])[0] if config_params else 'XY-X',
                'total_parts': results.get('total_parts_packed', 0),
                'efficiency': results.get('efficiency', 0),
                'waste_percentage': results.get('waste_percentage', 0),
                'parts_breakdown': results.get('prism_details', []),
                'visualization_file': None,
                'packing_count': 1,
                'scrap_volumes': 0,
                'stock_types_used': [f"{stock_dimensions['width']}×{stock_dimensions['height']}×{stock_dimensions['length']}"],
                'description': f"Optimized packing with {results.get('efficiency', 0)}% efficiency"
            }
            configurations.append(config)
        
        return {
            'success': True,
            'configurations': configurations,
            'stock_dimensions': stock_dimensions,
            'total_parts_processed': len(parts),
            'total_blocks': sum(p.get('quantity', 1) for p in parts)
        }
        
    except Exception as e:
        print(f"[pack_trapezoidal_prisms] Error: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return {
            'success': False,
            'error': str(e),
            'configurations': []
        }