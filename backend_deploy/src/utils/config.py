"""
Project configuration and specifications.

This module stores all the constants and specifications for the 3D packing project:
- Stock block sizes
- Trapezoidal prism specifications
- Visualization settings
- Optimization parameters
"""

from typing import Dict, List, Tuple
from dataclasses import dataclass


# ============================================================================
# STOCK BLOCK SPECIFICATIONS
# ============================================================================

@dataclass
class StockBlockSpec:
    """Stock block size specification."""
    name: str
    length: float  # mm
    width: float   # mm
    height: float  # mm
    
    @property
    def volume(self) -> float:
        """Calculate volume in mm³."""
        return self.length * self.width * self.height


# Define available stock block sizes
STOCK_BLOCKS: Dict[str, StockBlockSpec] = {
    "size_1": StockBlockSpec(
        name="Size 1 (500×500×2000)",
        length=500,   # mm
        width=500,    # mm
        height=2000   # mm
    ),
    "size_2": StockBlockSpec(
        name="Size 2 (800×400×2000)",
        length=800,   # mm
        width=400,    # mm
        height=2000   # mm
    )
    ,  "size_3": StockBlockSpec(
        name="Size 3 (800×800×2000)",
        length=800,   # mm
        width=800,    # mm
        height=2000   # mm
    )
}


# ============================================================================
# TRAPEZOIDAL PRISM SPECIFICATIONS
# ============================================================================

@dataclass
class TrapezoidalPrismSpec:
    """Trapezoidal prism part specification."""
    name: str
    W1: float       # Wider width (mm)
    W2: float       # Narrower width (mm)
    D: float        # Length (mm)
    thickness: float  # Height (mm)
    alpha: float    # Taper angle (degrees)
    
    @property
    def volume(self) -> float:
        """Calculate analytical volume in mm³."""
        return ((self.W1 + self.W2) / 2.0) * self.D * self.thickness
    
    @property
    def C(self) -> float:
        """Calculate offset C."""
        return abs(self.W1 - self.W2) / 2.0


# Define all trapezoidal prism parts (G1-G56) from TRIAL.xlsx
PART_SPECS: Dict[str, TrapezoidalPrismSpec] = {
    "G1": TrapezoidalPrismSpec(name="G1", W1=595.4, W2=537.5, D=764.4, thickness=164.8, alpha=2.184),
    "G2": TrapezoidalPrismSpec(name="G2", W1=595.4, W2=538.0, D=757.9, thickness=74.6, alpha=2.175),
    "G3": TrapezoidalPrismSpec(name="G3", W1=594.2, W2=537.9, D=743.2, thickness=67.8, alpha=2.175),
    "G4": TrapezoidalPrismSpec(name="G4", W1=595.4, W2=541.9, D=707.2, thickness=67.8, alpha=2.172),
    "G5": TrapezoidalPrismSpec(name="G5", W1=595.4, W2=544.9, D=667.7, thickness=67.8, alpha=2.169),
    "G6": TrapezoidalPrismSpec(name="G6", W1=595.4, W2=547.3, D=635.3, thickness=67.8, alpha=2.175),
    "G7": TrapezoidalPrismSpec(name="G7", W1=597.2, W2=552.6, D=589.5, thickness=67.8, alpha=2.175),
    "G8": TrapezoidalPrismSpec(name="G8", W1=596.8, W2=556.2, D=536.2, thickness=67.8, alpha=2.175),
    "G9": TrapezoidalPrismSpec(name="G9", W1=597.2, W2=558.6, D=510.0, thickness=201.6, alpha=2.175),
    "G10": TrapezoidalPrismSpec(name="G10", W1=597.6, W2=561.7, D=474.0, thickness=67.8, alpha=2.175),
    "G11": TrapezoidalPrismSpec(name="G11", W1=598.0, W2=563.7, D=452.9, thickness=67.8, alpha=2.175),
    "G12": TrapezoidalPrismSpec(name="G12", W1=598.4, W2=566.4, D=422.6, thickness=67.8, alpha=2.175),
    "G13": TrapezoidalPrismSpec(name="G13", W1=598.8, W2=568.7, D=397.7, thickness=67.8, alpha=2.175),
    "G14": TrapezoidalPrismSpec(name="G14", W1=598.8, W2=566.3, D=444.5, thickness=67.8, alpha=2.094),
    "G15": TrapezoidalPrismSpec(name="G15", W1=597.6, W2=561.8, D=473.9, thickness=67.8, alpha=2.168),
    "G16": TrapezoidalPrismSpec(name="G16", W1=597.6, W2=561.2, D=481.0, thickness=67.8, alpha=2.168),
    "G17": TrapezoidalPrismSpec(name="G17", W1=597.6, W2=560.5, D=490.6, thickness=67.8, alpha=2.168),
    "G18": TrapezoidalPrismSpec(name="G18", W1=597.2, W2=558.7, D=508.5, thickness=67.8, alpha=2.168),
    "G19": TrapezoidalPrismSpec(name="G19", W1=597.2, W2=557.6, D=523.2, thickness=67.8, alpha=2.168),
    "G20": TrapezoidalPrismSpec(name="G20", W1=596.8, W2=556.4, D=534.2, thickness=67.8, alpha=2.168),
    "G21": TrapezoidalPrismSpec(name="G21", W1=596.8, W2=554.8, D=555.0, thickness=67.8, alpha=2.168),
    "G22": TrapezoidalPrismSpec(name="G22", W1=596.4, W2=553.1, D=572.4, thickness=67.8, alpha=2.168),
    "G23": TrapezoidalPrismSpec(name="G23", W1=596.4, W2=551.6, D=591.6, thickness=67.8, alpha=2.168),
    "G24": TrapezoidalPrismSpec(name="G24", W1=596.0, W2=550.0, D=608.0, thickness=67.8, alpha=2.168),
    "G25": TrapezoidalPrismSpec(name="G25", W1=595.6, W2=547.8, D=632.0, thickness=67.8, alpha=2.168),
    "G26": TrapezoidalPrismSpec(name="G26", W1=595.2, W2=545.4, D=658.0, thickness=67.8, alpha=2.168),
    "G27": TrapezoidalPrismSpec(name="G27", W1=594.8, W2=543.0, D=684.8, thickness=67.8, alpha=2.168),
    "G28": TrapezoidalPrismSpec(name="G28", W1=594.8, W2=541.4, D=705.8, thickness=67.8, alpha=2.168),
    "G29": TrapezoidalPrismSpec(name="G29", W1=594.4, W2=538.2, D=742.6, thickness=67.8, alpha=2.168),
    "G30": TrapezoidalPrismSpec(name="G30", W1=598.8, W2=569.9, D=382.1, thickness=67.8, alpha=2.168),
    "G31": TrapezoidalPrismSpec(name="G31", W1=599.2, W2=572.6, D=351.6, thickness=67.8, alpha=2.168),
    "G32": TrapezoidalPrismSpec(name="G32", W1=599.2, W2=574.9, D=321.2, thickness=67.8, alpha=2.168),
    "G33": TrapezoidalPrismSpec(name="G33", W1=599.6, W2=577.2, D=295.9, thickness=67.8, alpha=2.168),
    "G34": TrapezoidalPrismSpec(name="G34", W1=599.6, W2=578.9, D=273.7, thickness=67.8, alpha=2.168),
    "G35": TrapezoidalPrismSpec(name="G35", W1=600.0, W2=580.7, D=255.0, thickness=67.8, alpha=2.168),
    "G36": TrapezoidalPrismSpec(name="G36", W1=600.0, W2=582.1, D=236.6, thickness=67.8, alpha=2.168),
    "G37": TrapezoidalPrismSpec(name="G37", W1=600.0, W2=583.6, D=216.5, thickness=67.8, alpha=2.168),
    "G38": TrapezoidalPrismSpec(name="G38", W1=600.4, W2=585.7, D=194.4, thickness=67.8, alpha=2.168),
    "G39": TrapezoidalPrismSpec(name="G39", W1=600.4, W2=587.3, D=173.2, thickness=67.8, alpha=2.168),
    "G40": TrapezoidalPrismSpec(name="G40", W1=600.8, W2=589.7, D=146.8, thickness=67.8, alpha=2.168),
    "G41": TrapezoidalPrismSpec(name="G41", W1=600.8, W2=591.0, D=129.5, thickness=67.8, alpha=2.168),
    "G42": TrapezoidalPrismSpec(name="G42", W1=595.9, W2=564.6, D=413.6, thickness=82.0, alpha=2.168),
    "G43": TrapezoidalPrismSpec(name="G43", W1=595.5, W2=561.5, D=449.1, thickness=82.0, alpha=2.168),
    "G44": TrapezoidalPrismSpec(name="G44", W1=595.5, W2=560.3, D=465.2, thickness=82.0, alpha=2.168),
    "G45": TrapezoidalPrismSpec(name="G45", W1=595.1, W2=558.5, D=483.9, thickness=82.0, alpha=2.168),
    "G46": TrapezoidalPrismSpec(name="G46", W1=595.1, W2=557.3, D=499.8, thickness=82.0, alpha=2.168),
    "G47": TrapezoidalPrismSpec(name="G47", W1=596.3, W2=566.3, D=396.4, thickness=82.0, alpha=2.168),
    "G48": TrapezoidalPrismSpec(name="G48", W1=596.3, W2=566.9, D=388.6, thickness=82.0, alpha=2.168),
    "G49": TrapezoidalPrismSpec(name="G49", W1=596.3, W2=567.6, D=379.3, thickness=82.0, alpha=2.168),
    "G50": TrapezoidalPrismSpec(name="G50", W1=596.7, W2=568.8, D=368.7, thickness=82.0, alpha=2.168),
    "G51": TrapezoidalPrismSpec(name="G51", W1=596.7, W2=569.8, D=355.4, thickness=82.0, alpha=2.168),
    "G52": TrapezoidalPrismSpec(name="G52", W1=597.1, W2=571.3, D=341.2, thickness=93.6, alpha=2.168),
    "G53": TrapezoidalPrismSpec(name="G53", W1=597.1, W2=573.1, D=317.1, thickness=93.6, alpha=2.168),
    "G54": TrapezoidalPrismSpec(name="G54", W1=597.1, W2=574.4, D=299.9, thickness=93.6, alpha=2.168),
    "G55": TrapezoidalPrismSpec(name="G55", W1=597.1, W2=576.2, D=276.2, thickness=93.6, alpha=2.168),
    "G56": TrapezoidalPrismSpec(name="G56", W1=597.1, W2=577.8, D=235.6, thickness=93.6, alpha=2.345),
}

# Default part spec (G14) for backwards compatibility
PART_SPEC = PART_SPECS["G14"]


def get_part_spec(name: str) -> TrapezoidalPrismSpec:
    """
    Get part specification by name.

    Args:
        name: Part name (e.g., "G1", "G14", "G56")

    Returns:
        TrapezoidalPrismSpec object

    Raises:
        KeyError: If part name not found
    """
    if name not in PART_SPECS:
        raise KeyError(f"Part '{name}' not found. "
                      f"Available: G1-G56")
    return PART_SPECS[name]


def get_parts_by_thickness(thickness: float, tolerance: float = 0.1) -> List[TrapezoidalPrismSpec]:
    """
    Get all parts with a specific thickness.

    Args:
        thickness: Target thickness in mm
        tolerance: Tolerance for matching (default 0.1mm)

    Returns:
        List of matching TrapezoidalPrismSpec objects
    """
    return [spec for spec in PART_SPECS.values()
            if abs(spec.thickness - thickness) <= tolerance]


# ============================================================================
# VISUALIZATION SETTINGS
# ============================================================================

VISUALIZATION = {
    # Rendering modes
    "wireframe": True,              # Use wireframe for step visualization
    "show_axes": True,              # Show coordinate axes
    "show_grid": False,             # Show grid plane
    
    # Colors (RGB tuples, 0-1 range)
    "colors": {
        "stock_block": (0.7, 0.7, 0.7),      # Light gray
        "part": (0.2, 0.6, 0.9),             # Blue
        "part_valid": (0.2, 0.8, 0.3),       # Green
        "part_invalid": (0.9, 0.2, 0.2),     # Red
        "scrap": (0.9, 0.6, 0.2),            # Orange
        "leftover": (0.8, 0.8, 0.3),         # Yellow
        "cut_plane": (0.5, 0.5, 0.9),        # Light blue
    },
    
    # Opacity settings
    "alpha": {
        "stock_block": 0.3,
        "part": 0.8,
        "cut_plane": 0.3,
    },
    
    # Output settings
    "export_formats": ["html", "png", "stl"],  # Available export formats
    "dpi": 150,                                # DPI for static images
    "figure_size": (12, 8),                    # Figure size in inches
}


# ============================================================================
# COORDINATE SYSTEM
# ============================================================================

COORDINATE_SYSTEM = {
    # Origin at corner of stock block
    "origin": (0, 0, 0),
    
    # Axes convention
    # X-axis: Length direction
    # Y-axis: Width direction  
    # Z-axis: Height direction
    "axes": {
        "x": "Length",
        "y": "Width",
        "z": "Height"
    },
    
    # Units
    "units": "mm"
}


# ============================================================================
# OPTIMIZATION PARAMETERS (for future steps)
# ============================================================================

OPTIMIZATION = {
    # Genetic Algorithm
    "ga": {
        "population_size": 100,
        "generations": 50,
        "crossover_prob": 0.7,
        "mutation_prob": 0.3,
        "tournament_size": 3,
        "elitism_count": 5,
    },
    
    # Fitness function weights
    "fitness": {
        "overlap_penalty": 10000,
        "out_of_bounds_penalty": 10000,
        "non_extractable_penalty": 50000,
    },
    
    # Tolerances
    "tolerances": {
        "volume_match": 0.001,      # 0.1% tolerance for volume matching
        "geometric": 0.01,          # 0.01 mm geometric tolerance
        "overlap_check": 0.1,       # 0.1 mm overlap detection threshold
    },
    
    # Orientation exploration
    "orientations": {
        "num_faces": 3,             # Number of unique faces (cuboid symmetry)
        "rotation_increment": 2,    # Degrees (2°, 4°, or 8° to be decided)
        "top_n_keep": 20,          # Keep top N orientations for GA
    }
}


# ============================================================================
# OUTPUT PATHS
# ============================================================================

OUTPUT_PATHS = {
    "base": "outputs",
    "visualizations": "outputs/visualizations",
    "reports": "outputs/reports",
    "exports": "outputs/exports",
    "logs": "outputs/logs",
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_stock_block_spec(size_name: str) -> StockBlockSpec:
    """
    Get stock block specification by name.
    
    Args:
        size_name: Either "size_1" or "size_2"
        
    Returns:
        StockBlockSpec object
        
    Raises:
        KeyError: If size_name not found
    """
    if size_name not in STOCK_BLOCKS:
        raise KeyError(f"Stock block size '{size_name}' not found. "
                      f"Available: {list(STOCK_BLOCKS.keys())}")
    return STOCK_BLOCKS[size_name]


def print_specifications():
    """Print all project specifications."""
    print("=" * 80)
    print("PROJECT SPECIFICATIONS")
    print("=" * 80)
    
    print("\n1. STOCK BLOCKS:")
    print("-" * 80)
    for key, spec in STOCK_BLOCKS.items():
        print(f"{key}: {spec.name}")
        print(f"  Dimensions: {spec.length} × {spec.width} × {spec.height} mm")
        print(f"  Volume: {spec.volume:,.0f} mm³")
    
    print("\n2. TRAPEZOIDAL PRISM PART:")
    print("-" * 80)
    print(f"Name: {PART_SPEC.name}")
    print(f"  W1 (wider width): {PART_SPEC.W1} mm")
    print(f"  W2 (narrower width): {PART_SPEC.W2} mm")
    print(f"  D (length): {PART_SPEC.D} mm")
    print(f"  Thickness: {PART_SPEC.thickness} mm")
    print(f"  Alpha (taper angle): {PART_SPEC.alpha}°")
    print(f"  C (offset): {PART_SPEC.C:.2f} mm")
    print(f"  Analytical volume: {PART_SPEC.volume:,.2f} mm³")
    
    print("\n3. COORDINATE SYSTEM:")
    print("-" * 80)
    print(f"Origin: {COORDINATE_SYSTEM['origin']}")
    print(f"Units: {COORDINATE_SYSTEM['units']}")
    print(f"Axes:")
    for axis, name in COORDINATE_SYSTEM['axes'].items():
        print(f"  {axis.upper()}-axis: {name}")
    
    print("\n4. VISUALIZATION SETTINGS:")
    print("-" * 80)
    print(f"Wireframe mode: {VISUALIZATION['wireframe']}")
    print(f"Show axes: {VISUALIZATION['show_axes']}")
    print(f"Export formats: {', '.join(VISUALIZATION['export_formats'])}")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    """Test configuration."""
    print_specifications()
    
    # Test getting stock block
    print("\nTesting stock block retrieval:")
    stock1 = get_stock_block_spec("size_1")
    print(f"Retrieved: {stock1.name}")
    
    stock2 = get_stock_block_spec("size_2")
    print(f"Retrieved: {stock2.name}")
