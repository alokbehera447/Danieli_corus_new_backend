"""
Packing algorithms for trapezoidal prisms into stock blocks.

Implements:
1. BestFitPacker - Best-fit greedy algorithm
2. MirroredPairPacker - Alternating narrow/wide bases for space efficiency
3. Py3dbpPacker - Using py3dbp library for 3D bin packing
"""

import numpy as np
import cadquery as cq
from typing import List, Tuple, Optional
from dataclasses import dataclass, field
from py3dbp import Packer, Bin, Item

from .trapezoid_geometry import TrapezoidGeometry, create_trapezoid_cq


@dataclass
class PlacedPart:
    """A part placed in the stock block."""
    part_id: int
    position: Tuple[float, float, float]
    mirrored: bool
    rotation: str
    geometry: Optional[cq.Workplane] = field(default=None, repr=False)

    @property
    def x(self) -> float:
        return self.position[0]

    @property
    def y(self) -> float:
        return self.position[1]

    @property
    def z(self) -> float:
        return self.position[2]


@dataclass
class PackingResult:
    """Result of a packing algorithm."""
    algorithm_name: str
    stock_dimensions: Tuple[float, float, float]
    part_dimensions: Tuple[float, float, float]
    placed_parts: List[PlacedPart]
    stock_volume: float
    parts_volume: float
    num_parts: int

    @property
    def utilization(self) -> float:
        return (self.parts_volume / self.stock_volume) * 100 if self.stock_volume > 0 else 0

    @property
    def waste_percent(self) -> float:
        return 100 - self.utilization


class BestFitPacker:
    """
    Best-fit greedy packing algorithm.
    Places parts in a grid pattern, choosing positions that minimize wasted space.
    """

    def __init__(self, stock_dims: Tuple[float, float, float],
                 part_geom: TrapezoidGeometry,
                 margin: float = 0.0):
        self.stock_length, self.stock_width, self.stock_height = stock_dims
        self.part_geom = part_geom
        self.margin = margin
        self.part_dx, self.part_dy, self.part_dz = part_geom.get_bounding_dimensions()

    def pack(self) -> PackingResult:
        """
        Pack parts using best-fit greedy algorithm.
        Tries to fill Z layers first, then Y rows, then X columns.
        """
        placed = []
        part_id = 0

        dx = self.part_dx + self.margin
        dy = self.part_dy + self.margin
        dz = self.part_dz + self.margin

        nx = int(self.stock_length // dx)
        ny = int(self.stock_width // dy)
        nz = int(self.stock_height // dz)

        positions = []
        for iz in range(nz):
            for iy in range(ny):
                for ix in range(nx):
                    x = ix * dx
                    y = iy * dy
                    z = iz * dz
                    if (x + self.part_dx <= self.stock_length and
                        y + self.part_dy <= self.stock_width and
                        z + self.part_dz <= self.stock_height):
                        positions.append((x, y, z))

        positions.sort(key=lambda p: (p[2], p[1], p[0]))

        for pos in positions:
            part_cq = create_trapezoid_cq(self.part_geom, mirrored=False)
            part_cq = part_cq.translate(pos)

            placed.append(PlacedPart(
                part_id=part_id,
                position=pos,
                mirrored=False,
                rotation="none",
                geometry=part_cq
            ))
            part_id += 1

        stock_vol = self.stock_length * self.stock_width * self.stock_height
        parts_vol = len(placed) * self.part_geom.volume

        return PackingResult(
            algorithm_name="Best-Fit Greedy",
            stock_dimensions=(self.stock_length, self.stock_width, self.stock_height),
            part_dimensions=(self.part_dx, self.part_dy, self.part_dz),
            placed_parts=placed,
            stock_volume=stock_vol,
            parts_volume=parts_vol,
            num_parts=len(placed)
        )


class MirroredPairPacker:
    """
    Mirrored pair packing algorithm.
    Alternates part orientation so narrow base of one part
    is adjacent to wide base of the next, maximizing space usage.
    """

    def __init__(self, stock_dims: Tuple[float, float, float],
                 part_geom: TrapezoidGeometry,
                 margin: float = 0.0):
        self.stock_length, self.stock_width, self.stock_height = stock_dims
        self.part_geom = part_geom
        self.margin = margin
        self.part_dx, self.part_dy, self.part_dz = part_geom.get_bounding_dimensions()

    def pack(self) -> PackingResult:
        """
        Pack parts with alternating mirror orientation.
        Each pair of parts shares bounding box width more efficiently.
        """
        placed = []
        part_id = 0

        dx = self.part_dx + self.margin
        dy = self.part_dy + self.margin
        dz = self.part_dz + self.margin

        pair_width = self.part_geom.W2 + self.margin

        nx = int(self.stock_length // dx)
        nz = int(self.stock_height // dz)

        for iz in range(nz):
            for ix in range(nx):
                y = 0.0
                pair_idx = 0

                while y + self.part_dy <= self.stock_width:
                    x = ix * dx
                    z = iz * dz

                    if (x + self.part_dx <= self.stock_length and
                        z + self.part_dz <= self.stock_height):

                        mirrored = (pair_idx % 2 == 1)
                        part_cq = create_trapezoid_cq(self.part_geom, mirrored=mirrored)
                        part_cq = part_cq.translate((x, y, z))

                        placed.append(PlacedPart(
                            part_id=part_id,
                            position=(x, y, z),
                            mirrored=mirrored,
                            rotation="none",
                            geometry=part_cq
                        ))
                        part_id += 1

                    y += dy
                    pair_idx += 1

        stock_vol = self.stock_length * self.stock_width * self.stock_height
        parts_vol = len(placed) * self.part_geom.volume

        return PackingResult(
            algorithm_name="Mirrored Pair Packing",
            stock_dimensions=(self.stock_length, self.stock_width, self.stock_height),
            part_dimensions=(self.part_dx, self.part_dy, self.part_dz),
            placed_parts=placed,
            stock_volume=stock_vol,
            parts_volume=parts_vol,
            num_parts=len(placed)
        )


class Py3dbpPacker:
    """
    3D bin packing using py3dbp library.
    Uses bounding box dimensions for packing calculation.
    """

    def __init__(self, stock_dims: Tuple[float, float, float],
                 part_geom: TrapezoidGeometry,
                 num_parts: int = 100):
        self.stock_length, self.stock_width, self.stock_height = stock_dims
        self.part_geom = part_geom
        self.num_parts = num_parts
        self.part_dx, self.part_dy, self.part_dz = part_geom.get_bounding_dimensions()

    def pack(self) -> PackingResult:
        """
        Pack parts using py3dbp library.
        """
        packer = Packer()

        packer.add_bin(Bin(
            "Stock",
            self.stock_length,
            self.stock_height,
            self.stock_width,
            1000000
        ))

        for i in range(self.num_parts):
            packer.add_item(Item(
                f"Part_{i}",
                self.part_dx,
                self.part_dz,
                self.part_dy,
                1
            ))

        packer.pack()

        placed = []
        for b in packer.bins:
            for item in b.items:
                pos_x = float(item.position[0])
                pos_z = float(item.position[1])
                pos_y = float(item.position[2])

                part_cq = create_trapezoid_cq(self.part_geom, mirrored=False)
                part_cq = part_cq.translate((pos_x, pos_y, pos_z))

                placed.append(PlacedPart(
                    part_id=len(placed),
                    position=(pos_x, pos_y, pos_z),
                    mirrored=False,
                    rotation=str(item.rotation_type),
                    geometry=part_cq
                ))

        stock_vol = self.stock_length * self.stock_width * self.stock_height
        parts_vol = len(placed) * self.part_geom.volume

        return PackingResult(
            algorithm_name="py3dbp Library",
            stock_dimensions=(self.stock_length, self.stock_width, self.stock_height),
            part_dimensions=(self.part_dx, self.part_dy, self.part_dz),
            placed_parts=placed,
            stock_volume=stock_vol,
            parts_volume=parts_vol,
            num_parts=len(placed)
        )


class RotatedBestFitPacker:
    """
    Best-fit packing with part rotated 90 degrees around Y-axis.
    Useful when part W1 exceeds stock width in default orientation.
    """

    def __init__(self, stock_dims: Tuple[float, float, float],
                 part_geom: TrapezoidGeometry,
                 margin: float = 0.0):
        self.stock_length, self.stock_width, self.stock_height = stock_dims
        self.part_geom = part_geom
        self.margin = margin
        self.part_dx = part_geom.thickness
        self.part_dy = part_geom.D
        self.part_dz = part_geom.W1

    def pack(self) -> PackingResult:
        """Pack parts rotated 90 degrees around Y-axis."""
        placed = []
        part_id = 0

        dx = self.part_dx + self.margin
        dy = self.part_dy + self.margin
        dz = self.part_dz + self.margin

        nx = int(self.stock_length // dx)
        ny = int(self.stock_width // dy)
        nz = int(self.stock_height // dz)

        for iz in range(nz):
            for iy in range(ny):
                for ix in range(nx):
                    x = ix * dx
                    y = iy * dy
                    z = iz * dz

                    if (x + self.part_dx <= self.stock_length and
                        y + self.part_dy <= self.stock_width and
                        z + self.part_dz <= self.stock_height):

                        from .trapezoid_geometry import create_trapezoid_rotated
                        part_cq = create_trapezoid_rotated(
                            self.part_geom,
                            rotation_axis="Y",
                            mirrored=False
                        )
                        part_cq = part_cq.translate((x, y, z))

                        placed.append(PlacedPart(
                            part_id=part_id,
                            position=(x, y, z),
                            mirrored=False,
                            rotation="Y90",
                            geometry=part_cq
                        ))
                        part_id += 1

        stock_vol = self.stock_length * self.stock_width * self.stock_height
        parts_vol = len(placed) * self.part_geom.volume

        return PackingResult(
            algorithm_name="Rotated Best-Fit (Y90)",
            stock_dimensions=(self.stock_length, self.stock_width, self.stock_height),
            part_dimensions=(self.part_dx, self.part_dy, self.part_dz),
            placed_parts=placed,
            stock_volume=stock_vol,
            parts_volume=parts_vol,
            num_parts=len(placed)
        )


class RotatedMirroredPacker:
    def __init__(self, stock_dims: Tuple[float, float, float],
                 part_geom: TrapezoidGeometry,
                 margin: float = 0.0):
        self.stock_length, self.stock_width, self.stock_height = stock_dims
        self.part_geom = part_geom
        self.margin = margin
        self.part_dx = part_geom.thickness
        self.part_dy = part_geom.D
        self.part_dz = part_geom.W1

    def _create_rotated_flipped(self, flipped: bool) -> cq.Workplane:
        geom = self.part_geom
        pts = geom.get_base_vertices_2d()

        prism = cq.Workplane("XY").polyline(pts).close().extrude(geom.thickness)
        prism = prism.rotate((0, 0, 0), (0, 1, 0), 90)

        if flipped:
            bbox = prism.val().BoundingBox()
            cx = (bbox.xmin + bbox.xmax) / 2
            cy = (bbox.ymin + bbox.ymax) / 2
            cz = (bbox.zmin + bbox.zmax) / 2
            prism = prism.rotate((cx,cy,cz), (cx + 1, cy , cz), 180)

        bbox = prism.val().BoundingBox()
        prism = prism.translate((-bbox.xmin, -bbox.ymin, -bbox.zmin))
        return prism

    def pack(self) -> PackingResult:
        placed = []
        part_id = 0

        dx = self.part_dx + self.margin
        dy = self.part_dy + self.margin
        dz = self.part_dz + self.margin

        nx = int(self.stock_length // dx)
        nz = int(self.stock_height // dz)

        for iz in range(nz):
            for ix in range(nx):
                y = 0.0
                pair_idx = 0

                while y + self.part_dy <= self.stock_width:
                    x = ix * dx
                    z = iz * dz

                    if (x + self.part_dx <= self.stock_length and
                        z + self.part_dz <= self.stock_height):

                        flipped = (iz % 2 == 1)
                        part_cq = self._create_rotated_flipped(flipped)
                        part_cq = part_cq.translate((x, y, z))

                        placed.append(PlacedPart(
                            part_id=part_id,
                            position=(x, y, z),
                            mirrored=flipped,
                            rotation="Y90" + ("_flip" if flipped else ""),
                            geometry=part_cq
                        ))
                        part_id += 1

                    y += dy
                    pair_idx += 1

        stock_vol = self.stock_length * self.stock_width * self.stock_height
        parts_vol = len(placed) * self.part_geom.volume

        return PackingResult(
            algorithm_name="Rotated Mirrored Pair (Y90)",
            stock_dimensions=(self.stock_length, self.stock_width, self.stock_height),
            part_dimensions=(self.part_dx, self.part_dy, self.part_dz),
            placed_parts=placed,
            stock_volume=stock_vol,
            parts_volume=parts_vol,
            num_parts=len(placed)
        )


class AutoOrientPacker:
    """
    Automatically selects best orientation for the part to fit in stock.
    Tests multiple orientations and picks the one with most parts.
    """

    ORIENTATIONS = [
        # (rotation_axes, resulting dims order: dx, dy, dz)
        ("none", lambda g: (g.W1, g.D, g.thickness)),           # default
        ("Y90", lambda g: (g.thickness, g.D, g.W1)),            # rotate around Y
        ("X90", lambda g: (g.W1, g.thickness, g.D)),            # rotate around X
        ("Z90", lambda g: (g.D, g.W1, g.thickness)),            # rotate around Z
        ("Y90_X90", lambda g: (g.D, g.thickness, g.W1)),        # Y90 then X90
        ("X90_Z90", lambda g: (g.thickness, g.W1, g.D)),        # X90 then Z90
    ]

    def __init__(self, stock_dims: Tuple[float, float, float],
                 part_geom: TrapezoidGeometry,
                 margin: float = 0.0,
                 mirrored: bool = False):
        self.stock_length, self.stock_width, self.stock_height = stock_dims
        self.part_geom = part_geom
        self.margin = margin
        self.mirrored = mirrored

    def _create_oriented_part(self, orientation: str) -> cq.Workplane:
        """Create part with specified orientation."""
        geom = self.part_geom
        pts = geom.get_base_vertices_2d()

        prism = (cq.Workplane("XY")
                 .polyline(pts)
                 .close()
                 .extrude(geom.thickness))

        if "Y90" in orientation:
            prism = prism.rotate((0, 0, 0), (0, 1, 0), 90)
        if "X90" in orientation:
            prism = prism.rotate((0, 0, 0), (1, 0, 0), 90)
        if "Z90" in orientation:
            prism = prism.rotate((0, 0, 0), (0, 0, 1), 90)

        bbox = prism.val().BoundingBox()
        prism = prism.translate((-bbox.xmin, -bbox.ymin, -bbox.zmin))
        return prism

    def _count_parts(self, orientation: str) -> Tuple[int, Tuple[float, float, float]]:
        """Count how many parts fit with given orientation."""
        for name, dims_fn in self.ORIENTATIONS:
            if name == orientation:
                dx, dy, dz = dims_fn(self.part_geom)
                break
        else:
            return 0, (0, 0, 0)

        if (dx > self.stock_length or dy > self.stock_width or dz > self.stock_height):
            return 0, (dx, dy, dz)

        nx = int(self.stock_length // (dx + self.margin))
        ny = int(self.stock_width // (dy + self.margin))
        nz = int(self.stock_height // (dz + self.margin))
        return nx * ny * nz, (dx, dy, dz)

    def pack(self) -> PackingResult:
        """Find best orientation and pack parts."""
        best_count = 0
        best_orientation = "none"
        best_dims = (0, 0, 0)

        for name, _ in self.ORIENTATIONS:
            count, dims = self._count_parts(name)
            if count > best_count:
                best_count = count
                best_orientation = name
                best_dims = dims

        if best_count == 0:
            return PackingResult(
                algorithm_name=f"Auto-Orient ({best_orientation})",
                stock_dimensions=(self.stock_length, self.stock_width, self.stock_height),
                part_dimensions=best_dims,
                placed_parts=[],
                stock_volume=self.stock_length * self.stock_width * self.stock_height,
                parts_volume=0,
                num_parts=0
            )

        dx, dy, dz = best_dims
        placed = []
        part_id = 0

        nx = int(self.stock_length // (dx + self.margin))
        ny = int(self.stock_width // (dy + self.margin))
        nz = int(self.stock_height // (dz + self.margin))

        for iz in range(nz):
            for iy in range(ny):
                for ix in range(nx):
                    x = ix * (dx + self.margin)
                    y = iy * (dy + self.margin)
                    z = iz * (dz + self.margin)

                    if (x + dx <= self.stock_length and
                        y + dy <= self.stock_width and
                        z + dz <= self.stock_height):

                        part_cq = self._create_oriented_part(best_orientation)
                        part_cq = part_cq.translate((x, y, z))

                        placed.append(PlacedPart(
                            part_id=part_id,
                            position=(x, y, z),
                            mirrored=self.mirrored,
                            rotation=best_orientation,
                            geometry=part_cq
                        ))
                        part_id += 1

        stock_vol = self.stock_length * self.stock_width * self.stock_height
        parts_vol = len(placed) * self.part_geom.volume

        return PackingResult(
            algorithm_name=f"Auto-Orient ({best_orientation})",
            stock_dimensions=(self.stock_length, self.stock_width, self.stock_height),
            part_dimensions=best_dims,
            placed_parts=placed,
            stock_volume=stock_vol,
            parts_volume=parts_vol,
            num_parts=len(placed)
        )
