"""
Core geometry data classes for 3D packing optimization.

This module defines the fundamental data structures used throughout the project:
- StockBlock: Cuboidal parent block
- TrapezoidalPrism: Desired part shape
- CutPlane: Planar cut specification
- PartPlacement: Part position and orientation
"""

import numpy as np
from typing import Tuple, Optional, Any
from dataclasses import dataclass, field


@dataclass
class StockBlock:
    """
    Cuboidal stock block (parent block).
    
    Attributes:
        length: Length along X-axis (mm)
        width: Width along Y-axis (mm)
        height: Height along Z-axis (mm)
        material_type: Type of material (default: "Generic")
        geometry: CadQuery geometry object (populated later)
    """
    length: float
    width: float
    height: float
    material_type: str = "Generic"
    geometry: Optional[Any] = field(default=None, repr=False)
    
    def __post_init__(self):
        """Calculate volume after initialization."""
        self.volume = self.length * self.width * self.height
    
    @property
    def dimensions(self) -> Tuple[float, float, float]:
        """Return dimensions as tuple (L, W, H)."""
        return (self.length, self.width, self.height)
    
    @property
    def bounding_box(self) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """Return bounding box as (min_point, max_point)."""
        return ((0, 0, 0), (self.length, self.width, self.height))
    
    def __repr__(self) -> str:
        return (f"StockBlock(L={self.length}mm, W={self.width}mm, H={self.height}mm, "
                f"V={self.volume:,.0f}mm³, material={self.material_type})")


@dataclass
class TrapezoidalPrism:
    """
    Trapezoidal prism part specification.
    
    Attributes:
        W1: Larger parallel width (mm)
        W2: Smaller parallel width (mm)
        D: Length - distance between parallel sides (mm)
        thickness: Height of the prism (mm)
        alpha: Angle of side taper (degrees)
        geometry: CadQuery geometry object (populated later)
    """
    W1: float  # Wider width
    W2: float  # Narrower width
    D: float   # Length
    thickness: float
    alpha: float  # Angle in degrees
    geometry: Optional[Any] = field(default=None, repr=False)
    
    def __post_init__(self):
        """Calculate derived properties after initialization."""
        # Calculate offset C
        self.C = abs(self.W1 - self.W2) / 2.0
        
        # Calculate analytical volume
        # V = average_width × length × thickness
        self.volume_analytical = ((self.W1 + self.W2) / 2.0) * self.D * self.thickness
        
        # Volume from geometry (to be populated when geometry is created)
        self.volume_cad = None
    
    @property
    def dimensions(self) -> Tuple[float, float, float, float, float]:
        """Return key dimensions as tuple (W1, W2, D, thickness, alpha)."""
        return (self.W1, self.W2, self.D, self.thickness, self.alpha)
    
    def validate_volume(self, tolerance: float = 0.001) -> bool:
        """
        Validate that CAD volume matches analytical volume.
        
        Args:
            tolerance: Relative tolerance (default 0.1%)
            
        Returns:
            True if volumes match within tolerance
        """
        if self.volume_cad is None:
            raise ValueError("CAD volume not yet calculated. Generate geometry first.")
        
        relative_error = abs(self.volume_analytical - self.volume_cad) / self.volume_analytical
        return relative_error < tolerance
    
    def __repr__(self) -> str:
        vol_str = f"{self.volume_analytical:,.0f}mm³"
        if self.volume_cad is not None:
            vol_str += f" (CAD: {self.volume_cad:,.0f}mm³)"
        
        return (f"TrapezoidalPrism(W1={self.W1}mm, W2={self.W2}mm, D={self.D}mm, "
                f"thickness={self.thickness}mm, α={self.alpha}°, C={self.C:.2f}mm, V={vol_str})")


@dataclass
class CutPlane:
    """
    Planar cut specification with equation and normal vector.
    
    A plane is defined by: Ax + By + Cz + D = 0
    Or by: normal vector (A, B, C) and a point on the plane
    
    Attributes:
        normal: Normal vector (nx, ny, nz) - should be unit vector
        point: A point on the plane (x0, y0, z0)
        description: Human-readable description of the cut
    """
    normal: Tuple[float, float, float]
    point: Tuple[float, float, float]
    description: str = ""
    
    def __post_init__(self):
        """Normalize the normal vector."""
        nx, ny, nz = self.normal
        magnitude = np.sqrt(nx**2 + ny**2 + nz**2)
        
        if magnitude < 1e-10:
            raise ValueError("Normal vector cannot be zero")
        
        # Store normalized normal vector
        self.normal = (nx / magnitude, ny / magnitude, nz / magnitude)
    
    def get_equation(self) -> Tuple[float, float, float, float]:
        """
        Get plane equation coefficients: Ax + By + Cz + D = 0
        
        Returns:
            Tuple (A, B, C, D) representing plane equation
        """
        A, B, C = self.normal
        x0, y0, z0 = self.point
        D = -(A * x0 + B * y0 + C * z0)
        return (A, B, C, D)
    
    def get_normal_angles(self) -> Tuple[float, float, float]:
        """
        Calculate angles between normal vector and coordinate axes.
        
        Returns:
            Tuple (alpha_x, alpha_y, alpha_z) in degrees
            These are the angles the normal makes with X, Y, Z axes
        """
        nx, ny, nz = self.normal
        
        # Angle with X-axis
        alpha_x = np.degrees(np.arccos(np.clip(nx, -1.0, 1.0)))
        
        # Angle with Y-axis
        alpha_y = np.degrees(np.arccos(np.clip(ny, -1.0, 1.0)))
        
        # Angle with Z-axis
        alpha_z = np.degrees(np.arccos(np.clip(nz, -1.0, 1.0)))
        
        return (alpha_x, alpha_y, alpha_z)
    
    def distance_to_point(self, point: Tuple[float, float, float]) -> float:
        """
        Calculate signed distance from plane to a point.
        
        Positive distance means point is on the side of the normal vector.
        
        Args:
            point: Point coordinates (x, y, z)
            
        Returns:
            Signed distance from plane to point
        """
        A, B, C, D = self.get_equation()
        x, y, z = point
        return A * x + B * y + C * z + D
    
    def __repr__(self) -> str:
        A, B, C, D = self.get_equation()
        alpha_x, alpha_y, alpha_z = self.get_normal_angles()
        
        desc_str = f' "{self.description}"' if self.description else ""
        
        return (f"CutPlane(equation: {A:.3f}x + {B:.3f}y + {C:.3f}z + {D:.3f} = 0, "
                f"normal: [{self.normal[0]:.3f}, {self.normal[1]:.3f}, {self.normal[2]:.3f}], "
                f"angles: [α_x={alpha_x:.1f}°, α_y={alpha_y:.1f}°, α_z={alpha_z:.1f}°]"
                f"{desc_str})")


@dataclass
class PartPlacement:
    """
    Part placement specification: position and orientation.
    
    Attributes:
        part_id: Unique identifier for this part instance
        part_spec: Reference to TrapezoidalPrism specification
        position: Center point or reference point (x, y, z)
        orientation_index: Index into orientation library
        mirrored: Whether this part is mirrored/flipped
        geometry: Transformed CadQuery geometry (populated later)
    """
    part_id: int
    part_spec: TrapezoidalPrism
    position: Tuple[float, float, float]
    orientation_index: int = 0
    mirrored: bool = False
    geometry: Optional[Any] = field(default=None, repr=False)
    
    @property
    def x(self) -> float:
        """X coordinate of position."""
        return self.position[0]
    
    @property
    def y(self) -> float:
        """Y coordinate of position."""
        return self.position[1]
    
    @property
    def z(self) -> float:
        """Z coordinate of position."""
        return self.position[2]
    
    def __repr__(self) -> str:
        mirror_str = " (mirrored)" if self.mirrored else ""
        return (f"PartPlacement(id={self.part_id}, pos=({self.x:.1f}, {self.y:.1f}, {self.z:.1f}), "
                f"orientation={self.orientation_index}{mirror_str})")


# Validation functions
def validate_stock_block(block: StockBlock) -> bool:
    """
    Validate that stock block has positive dimensions.
    
    Args:
        block: StockBlock to validate
        
    Returns:
        True if valid
        
    Raises:
        ValueError: If dimensions are non-positive
    """
    if block.length <= 0 or block.width <= 0 or block.height <= 0:
        raise ValueError(f"Stock block dimensions must be positive: {block.dimensions}")
    return True


def validate_trapezoidal_prism(prism: TrapezoidalPrism) -> bool:
    """
    Validate that trapezoidal prism has valid dimensions.
    
    Args:
        prism: TrapezoidalPrism to validate
        
    Returns:
        True if valid
        
    Raises:
        ValueError: If dimensions are invalid
    """
    if prism.W1 <= 0 or prism.W2 <= 0:
        raise ValueError(f"Widths must be positive: W1={prism.W1}, W2={prism.W2}")
    
    if prism.D <= 0:
        raise ValueError(f"Length D must be positive: D={prism.D}")
    
    if prism.thickness <= 0:
        raise ValueError(f"Thickness must be positive: thickness={prism.thickness}")
    
    if prism.alpha < 0 or prism.alpha > 90:
        raise ValueError(f"Angle alpha must be between 0 and 90 degrees: α={prism.alpha}")
    
    return True


if __name__ == "__main__":
    """Test the data classes."""
    
    print("=" * 80)
    print("Testing Core Geometry Data Classes")
    print("=" * 80)
    
    # Test StockBlock
    print("\n1. Testing StockBlock:")
    print("-" * 80)
    
    stock1 = StockBlock(length=500, width=500, height=2000, material_type="Graphite")
    stock2 = StockBlock(length=800, width=500, height=2000, material_type="Graphite")
    
    print(stock1)
    print(stock2)
    print(f"\nStock 1 dimensions: {stock1.dimensions}")
    print(f"Stock 1 bounding box: {stock1.bounding_box}")
    
    validate_stock_block(stock1)
    validate_stock_block(stock2)
    print("✓ Stock blocks validated successfully")
    
    # Test TrapezoidalPrism
    print("\n2. Testing TrapezoidalPrism:")
    print("-" * 80)
    
    part = TrapezoidalPrism(
        W1=598.8,
        W2=566.3,
        D=444.5,
        thickness=67.8,
        alpha=2.1
    )
    
    print(part)
    print(f"\nPart dimensions: {part.dimensions}")
    print(f"Calculated offset C: {part.C:.2f} mm")
    print(f"Analytical volume: {part.volume_analytical:,.2f} mm³")
    
    validate_trapezoidal_prism(part)
    print("✓ Trapezoidal prism validated successfully")
    
    # Test CutPlane
    print("\n3. Testing CutPlane:")
    print("-" * 80)
    
    # Vertical cut plane at x = 250
    plane1 = CutPlane(
        normal=(1, 0, 0),
        point=(250, 0, 0),
        description="Vertical cut along X-axis at x=250"
    )
    
    # Angled cut plane
    plane2 = CutPlane(
        normal=(1, 1, 0),
        point=(0, 0, 0),
        description="45-degree angled cut in XY plane"
    )
    
    print(plane1)
    print(f"\nPlane 1 equation: {plane1.get_equation()}")
    print(f"Plane 1 angles: {plane1.get_normal_angles()}")
    
    print(f"\n{plane2}")
    print(f"Plane 2 equation: {plane2.get_equation()}")
    print(f"Plane 2 angles: {plane2.get_normal_angles()}")
    
    # Test distance calculation
    test_point = (300, 0, 0)
    dist = plane1.distance_to_point(test_point)
    print(f"\nDistance from plane1 to point {test_point}: {dist:.2f} mm")
    
    print("✓ Cut planes created and tested successfully")
    
    # Test PartPlacement
    print("\n4. Testing PartPlacement:")
    print("-" * 80)
    
    placement1 = PartPlacement(
        part_id=1,
        part_spec=part,
        position=(100, 100, 50),
        orientation_index=0,
        mirrored=False
    )
    
    placement2 = PartPlacement(
        part_id=2,
        part_spec=part,
        position=(100, 100, 150),
        orientation_index=5,
        mirrored=True
    )
    
    print(placement1)
    print(placement2)
    print(f"\nPlacement 1 coordinates: x={placement1.x}, y={placement1.y}, z={placement1.z}")
    
    print("✓ Part placements created successfully")
    
    print("\n" + "=" * 80)
    print("✓ All data classes tested successfully!")
    print("=" * 80)
