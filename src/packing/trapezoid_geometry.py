"""
Trapezoidal prism geometry calculations.
Vertex calculation using W1, W2, D, thickness (ignoring alpha angle).
"""

import numpy as np
import cadquery as cq
from typing import List, Tuple
from dataclasses import dataclass


@dataclass
class TrapezoidGeometry:
    """
    Trapezoidal prism geometry with vertex calculations.
    Uses W1, W2, D, thickness only (alpha ignored as per spec).

    Cross-section in XY plane:
    - W1: wider width at y=0
    - W2: narrower width at y=D
    - D: depth (length in Y direction)
    - thickness: height in Z direction

    Vertex layout (looking down Z-axis):

        (C, D) -------- (W1-C, D)     <- narrower side (W2)
           /              \\
          /                \\
         /                  \\
        (0, 0) -------- (W1, 0)       <- wider side (W1)

    Where C = (W1 - W2) / 2
    """
    W1: float
    W2: float
    D: float
    thickness: float

    def __post_init__(self):
        w1 = self.W1
        w2 = self.W2

        self.W1 = max(w1, w2)  # larger width
        self.W2 = min(w1, w2)  # smaller width
        self.C = (self.W1 - self.W2) / 2.0
        self.volume = ((self.W1 + self.W2) / 2.0) * self.D * self.thickness

    def get_base_vertices_2d(self) -> List[Tuple[float, float]]:
        """Get 2D vertices of trapezoid base (XY plane, counterclockwise)."""
        return [
            (0, 0),
            (self.W1, 0),
            (self.W1 - self.C, self.D),
            (self.C, self.D)
        ]

    def get_vertices_3d(self) -> np.ndarray:
        """
        Get all 8 vertices of the trapezoidal prism.

        Returns:
            ndarray of shape (8, 3) with vertex coordinates
        """
        base_2d = self.get_base_vertices_2d()
        vertices = []
        for x, y in base_2d:
            vertices.append([x, y, 0])
        for x, y in base_2d:
            vertices.append([x, y, self.thickness])
        return np.array(vertices)

    def get_bounding_box(self) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """Get axis-aligned bounding box: (min_corner, max_corner)."""
        return ((0, 0, 0), (self.W1, self.D, self.thickness))

    def get_bounding_dimensions(self) -> Tuple[float, float, float]:
        """Get bounding box dimensions (dx, dy, dz)."""
        return (self.W1, self.D, self.thickness)


def create_trapezoid_cq(geom: TrapezoidGeometry, mirrored: bool = False) -> cq.Workplane:
    """
    Create CadQuery geometry for trapezoidal prism.

    Args:
        geom: TrapezoidGeometry specification
        mirrored: If True, flip the trapezoid so narrow end is at y=0

    Returns:
        CadQuery Workplane with trapezoid at origin
    """
    if mirrored:
        pts = [
            (geom.C, 0),
            (geom.W1 - geom.C, 0),
            (geom.W1, geom.D),
            (0, geom.D)
        ]
    else:
        pts = geom.get_base_vertices_2d()

    prism = (cq.Workplane("XY")
             .polyline(pts)
             .close()
             .extrude(geom.thickness))

    return prism


def create_trapezoid_rotated(geom: TrapezoidGeometry,
                             rotation_axis: str = "Y",
                             mirrored: bool = False) -> cq.Workplane:
    """
    Create CadQuery geometry rotated 90 degrees around specified axis.
    Useful when part doesn't fit in stock in default orientation.

    Args:
        geom: TrapezoidGeometry specification
        rotation_axis: "X", "Y", or "Z"
        mirrored: If True, flip the trapezoid

    Returns:
        CadQuery Workplane with rotated trapezoid, origin at corner
    """
    prism = create_trapezoid_cq(geom, mirrored)

    axis_map = {"X": (1, 0, 0), "Y": (0, 1, 0), "Z": (0, 0, 1)}
    axis_vec = axis_map.get(rotation_axis.upper(), (0, 1, 0))

    prism = prism.rotate((0, 0, 0), axis_vec, 90)

    bbox = prism.val().BoundingBox()
    prism = prism.translate((-bbox.xmin, -bbox.ymin, -bbox.zmin))

    return prism


if __name__ == "__main__":
    from src.utils.config import PART_SPEC

    geom = TrapezoidGeometry(
        W1=PART_SPEC.W1,
        W2=PART_SPEC.W2,
        D=PART_SPEC.D,
        thickness=PART_SPEC.thickness
    )

    print("Trapezoid Geometry:")
    print(f"  W1={geom.W1}, W2={geom.W2}, D={geom.D}, thickness={geom.thickness}")
    print(f"  C (offset) = {geom.C:.2f}")
    print(f"  Volume = {geom.volume:,.2f} mm³")
    print(f"\n2D Base Vertices: {geom.get_base_vertices_2d()}")
    print(f"\n3D Vertices:\n{geom.get_vertices_3d()}")
    print(f"\nBounding box: {geom.get_bounding_box()}")
    print(f"Bounding dimensions: {geom.get_bounding_dimensions()}")
