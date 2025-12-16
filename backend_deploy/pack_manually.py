"""
Step 5c: Mixed Parts Packing Optimization (G1-G5)

Packs DIFFERENT part types (G1, G2, G3, G4, G5) together in the same stock block.
Finds optimal combinations that minimize waste while ensuring all parts are extractable.

Strategy:
1. Try to fit largest parts first (by volume)
2. Fill remaining space with smaller parts
3. Use greedy placement with multiple orientations
"""

import os
import sys
import argparse
import itertools
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
import cadquery as cq
import numpy as np
import pandas as pd

from src.utils.config import STOCK_BLOCKS, PART_SPECS, TrapezoidalPrismSpec, get_parts_by_thickness, StockBlockSpec
from src.packing.trapezoid_geometry import TrapezoidGeometry, create_trapezoid_cq
from src.packing.orientation_explorer import OrientationExplorer, Orientation
from src.cutting.guillotine_validator import (
    GuillotineValidatorRecursive,
    Box3D,
    boxes_from_packing
)
from src.packing.packing_algorithms import (
    RotatedBestFitPacker,
    RotatedMirroredPacker,
    Py3dbpPacker,
    AutoOrientPacker,
    PackingResult
)
from src.visualization import GeometryExporter


OUTPUT_DIR = "outputs/visualizations/step5_mixed_parts"
DEFAULT_SAW_KERF = 0.0  # mm
Rotational_inc = 0


@dataclass
class MixedPlacedPart:
    """A part placed in the stock block."""
    part_spec_name: str  # G1, G2, etc.
    part_id: int
    position: Tuple[float, float, float]  # (x, y, z)
    orientation_idx: int
    mirrored: bool
    geometry: cq.Workplane
    bounding_box: Tuple[float, float, float, float, float, float]  # (min_x, min_y, min_z, max_x, max_y, max_z)
    volume: float


@dataclass
class MixedPackingResult:
    """Result of mixed parts packing."""
    placed_parts: List[MixedPlacedPart]
    parts_by_type: Dict[str, int]  # count of each part type
    total_parts: int
    total_volume: float
    waste_percentage: float
    is_extractable: bool
    stock_volume: float


def get_bounding_box(geom: cq.Workplane) -> Tuple[float, float, float, float, float, float]:
    """Get bounding box of a CadQuery geometry."""
    bb = geom.val().BoundingBox()
    return (bb.xmin, bb.ymin, bb.zmin, bb.xmax, bb.ymax, bb.zmax)


def boxes_overlap(bb1: Tuple, bb2: Tuple, tolerance: float = 0.01) -> bool:
    """Check if two bounding boxes overlap."""
    return not (bb1[3] <= bb2[0] + tolerance or bb2[3] <= bb1[0] + tolerance or
                bb1[4] <= bb2[1] + tolerance or bb2[4] <= bb1[1] + tolerance or
                bb1[5] <= bb2[2] + tolerance or bb2[5] <= bb1[2] + tolerance)


def is_within_stock(bb: Tuple, stock_dims: Tuple[float, float, float], tolerance: float = 0.01) -> bool:
    """Check if bounding box is within stock dimensions."""
    return (bb[0] >= -tolerance and bb[1] >= -tolerance and bb[2] >= -tolerance and
            bb[3] <= stock_dims[0] + tolerance and
            bb[4] <= stock_dims[1] + tolerance and
            bb[5] <= stock_dims[2] + tolerance)


class MixedPartsPacker:
    """Packs multiple different part types into a single stock block."""

    def __init__(self, stock_dims: Tuple[float, float, float],
                 part_specs: Dict[str, TrapezoidalPrismSpec],
                 saw_kerf: float = DEFAULT_SAW_KERF):
        """
        Args:
            stock_dims: (length, width, height) of stock block
            part_specs: Dict of part name -> TrapezoidalPrismSpec
            saw_kerf: Saw blade thickness
        """
        self.stock_dims = stock_dims
        self.part_specs = part_specs
        self.saw_kerf = saw_kerf
        self.stock_volume = stock_dims[0] * stock_dims[1] * stock_dims[2]

        # Pre-compute geometries and orientations for each part type
        self.part_geometries: Dict[str, TrapezoidGeometry] = {}
        self.part_explorers: Dict[str, OrientationExplorer] = {}
        self.part_orientations: Dict[str, List[Orientation]] = {}

        for name, spec in part_specs.items():
            geom = TrapezoidGeometry(
                W1=spec.W1, W2=spec.W2, D=spec.D, thickness=spec.thickness
            )
            self.part_geometries[name] = geom

            explorer = OrientationExplorer(geom, Rotational_inc)
            self.part_explorers[name] = explorer
            orientations = explorer.filter_fitting_orientations(stock_dims)
            self.part_orientations[name] = orientations

        # Sort parts by volume (largest first)
        self.parts_by_volume = sorted(
            part_specs.keys(),
            key=lambda n: self.part_geometries[n].volume,
            reverse=True
        )

    def _find_placement_position(self, part_name: str, orientation_idx: int,
                                  mirrored: bool, placed_parts: List[MixedPlacedPart]
                                  ) -> Optional[Tuple[Tuple[float, float, float], cq.Workplane, Tuple]]:
        """
        Find a valid position for a part using bottom-left-back placement.

        Returns:
            (position, geometry, bounding_box) or None if no valid position found
        """
        orientations = self.part_orientations.get(part_name, [])
        if orientation_idx >= len(orientations):
            return None

        orient = orientations[orientation_idx]
        explorer = self.part_explorers[part_name]

        # Get oriented bounding box dimensions (dims is a tuple: length, width, height)
        bbox_dims = orient.dims

        # Grid search for valid positions
        step = max(10.0, self.saw_kerf)  # Search step size

        # Generate candidate positions
        x_positions = np.arange(0, self.stock_dims[0] - bbox_dims[0] + step, step)
        y_positions = np.arange(0, self.stock_dims[1] - bbox_dims[1] + step, step)
        z_positions = np.arange(0, self.stock_dims[2] - bbox_dims[2] + step, step)

        for z in z_positions:
            for y in y_positions:
                for x in x_positions:
                    # Create geometry at this position using OrientationExplorer
                    geom = explorer.create_oriented_geometry(orient)

                    # Apply mirroring if requested (flip in Y direction)
                    if mirrored:
                        geom = geom.mirror("XZ")
                        # Re-normalize to origin
                        bb_temp = geom.val().BoundingBox()
                        geom = geom.translate((-bb_temp.xmin, -bb_temp.ymin, -bb_temp.zmin))

                    # Translate to position
                    geom = geom.translate((x, y, z))

                    bb = get_bounding_box(geom)

                    # Check within stock
                    if not is_within_stock(bb, self.stock_dims):
                        continue

                    # Check overlap with existing parts
                    overlaps = False
                    for placed in placed_parts:
                        if boxes_overlap(bb, placed.bounding_box, self.saw_kerf):
                            overlaps = True
                            break

                    if not overlaps:
                        return ((x, y, z), geom, bb)

        return None

    def _iterative_fill(self, placed_parts: List[MixedPlacedPart],
                       parts_count: Dict[str, int],
                       total_volume: float,
                       part_id: int,
                       max_iterations: int = 5) -> Tuple[List[MixedPlacedPart], Dict[str, int], float, int]:
        """
        Iteratively try to fill remaining space with any available parts.

        After initial packing, this method repeatedly tries to fit any part (any type, any orientation)
        into the remaining empty space until no more parts can be placed.

        Args:
            placed_parts: Currently placed parts
            parts_count: Count of each part type placed
            total_volume: Total volume of placed parts
            part_id: Next part ID to use
            max_iterations: Maximum number of filling iterations

        Returns:
            Tuple of (updated_placed_parts, updated_parts_count, updated_total_volume, updated_part_id)
        """
        print(f"    Iterative filling: Starting with {len(placed_parts)} parts...")

        for iteration in range(max_iterations):
            parts_added_this_iteration = 0
            initial_count = len(placed_parts)

            # Try all part types (sorted by volume, smallest first for better gap filling)
            for part_name in reversed(self.parts_by_volume):
                orientations = self.part_orientations.get(part_name, [])
                if not orientations:
                    continue

                # Try to place as many as possible
                while True:
                    placed = False

                    # Try each orientation
                    for orient_idx in range(len(orientations)):
                        # Try normal and mirrored
                        for mirrored in [False, True]:
                            result = self._find_placement_position(
                                part_name, orient_idx, mirrored, placed_parts
                            )

                            if result:
                                pos, geom, bb = result
                                part = MixedPlacedPart(
                                    part_spec_name=part_name,
                                    part_id=part_id,
                                    position=pos,
                                    orientation_idx=orient_idx,
                                    mirrored=mirrored,
                                    geometry=geom,
                                    bounding_box=bb,
                                    volume=self.part_geometries[part_name].volume
                                )
                                placed_parts.append(part)
                                parts_count[part_name] = parts_count.get(part_name, 0) + 1
                                total_volume += part.volume
                                part_id += 1
                                parts_added_this_iteration += 1
                                placed = True
                                break

                        if placed:
                            break

                    if not placed:
                        break  # No more of this part type fits

            print(f"      Iteration {iteration + 1}: Added {parts_added_this_iteration} parts "
                  f"(total: {len(placed_parts)})")

            # Stop if no new parts were added
            if parts_added_this_iteration == 0:
                break

        final_count = len(placed_parts)
        print(f"    Iterative filling complete: {final_count - initial_count} additional parts added")

        return placed_parts, parts_count, total_volume, part_id

    def pack_greedy_largest_first(self, max_parts_per_type: int = 10,
                                   iterative_fill: bool = True) -> MixedPackingResult:
        """
        Greedy packing: try to place largest parts first, then fill with smaller.

        Args:
            max_parts_per_type: Maximum number of each part type to try
            iterative_fill: If True, iteratively try to fill remaining space

        Returns:
            MixedPackingResult
        """
        placed_parts: List[MixedPlacedPart] = []
        parts_count: Dict[str, int] = {name: 0 for name in self.part_specs}
        part_id = 0
        total_volume = 0.0

        # Try each part type in order of decreasing volume
        for part_name in self.parts_by_volume:
            orientations = self.part_orientations.get(part_name, [])
            if not orientations:
                continue

            # Try to place multiple instances of this part
            for _ in range(max_parts_per_type):
                placed = False

                # Try each orientation
                for orient_idx in range(len(orientations)):
                    # Try normal and mirrored
                    for mirrored in [False, True]:
                        result = self._find_placement_position(
                            part_name, orient_idx, mirrored, placed_parts
                        )

                        if result:
                            pos, geom, bb = result
                            part = MixedPlacedPart(
                                part_spec_name=part_name,
                                part_id=part_id,
                                position=pos,
                                orientation_idx=orient_idx,
                                mirrored=mirrored,
                                geometry=geom,
                                bounding_box=bb,
                                volume=self.part_geometries[part_name].volume
                            )
                            placed_parts.append(part)
                            parts_count[part_name] += 1
                            total_volume += part.volume
                            part_id += 1
                            placed = True
                            break

                    if placed:
                        break

                if not placed:
                    break  # No more of this part type fits

        # Iterative space filling: try to fit more parts into remaining space
        if iterative_fill:
            placed_parts, parts_count, total_volume, part_id = self._iterative_fill(
                placed_parts, parts_count, total_volume, part_id, max_iterations=5
            )

        # Check extractability
        is_extractable = self._check_extractability(placed_parts)

        waste_percentage = (1 - total_volume / self.stock_volume) * 100

        return MixedPackingResult(
            placed_parts=placed_parts,
            parts_by_type={k: v for k, v in parts_count.items() if v > 0},
            total_parts=len(placed_parts),
            total_volume=total_volume,
            waste_percentage=waste_percentage,
            is_extractable=is_extractable,
            stock_volume=self.stock_volume
        )

    def pack_by_thickness_layers(self, max_parts_per_type: int = 10,
                                  iterative_fill: bool = True) -> MixedPackingResult:
        """
        Pack parts in layers by thickness - parts with same thickness in same layer.

        Args:
            max_parts_per_type: Maximum number of each part type to try
            iterative_fill: If True, iteratively try to fill remaining space

        Returns:
            MixedPackingResult
        """
        placed_parts: List[MixedPlacedPart] = []
        parts_count: Dict[str, int] = {name: 0 for name in self.part_specs}
        part_id = 0
        total_volume = 0.0

        # Group parts by thickness
        thickness_groups: Dict[float, List[str]] = {}
        for name, spec in self.part_specs.items():
            t = round(spec.thickness, 1)
            if t not in thickness_groups:
                thickness_groups[t] = []
            thickness_groups[t].append(name)

        # Sort thicknesses (largest first for better packing)
        sorted_thicknesses = sorted(thickness_groups.keys(), reverse=True)

        current_z = 0.0

        for thickness in sorted_thicknesses:
            part_names = thickness_groups[thickness]

            # Sort by volume within thickness group
            part_names_sorted = sorted(
                part_names,
                key=lambda n: self.part_geometries[n].volume,
                reverse=True
            )

            layer_placed = []

            for part_name in part_names_sorted:
                orientations = self.part_orientations.get(part_name, [])
                if not orientations:
                    continue

                for _ in range(max_parts_per_type):
                    placed = False

                    for orient_idx in range(len(orientations)):
                        orient = orientations[orient_idx]

                        # Only use orientations that keep part within current layer
                        if abs(orient.dims[2] - thickness) > 1.0:
                            continue

                        for mirrored in [False, True]:
                            # Modified placement search for current layer
                            explorer = self.part_explorers[part_name]
                            bbox_dims = orient.dims
                            step = max(10.0, self.saw_kerf)

                            x_positions = np.arange(0, self.stock_dims[0] - bbox_dims[0] + step, step)
                            y_positions = np.arange(0, self.stock_dims[1] - bbox_dims[1] + step, step)

                            for y in y_positions:
                                for x in x_positions:
                                    z = current_z

                                    # Create geometry using OrientationExplorer
                                    geom = explorer.create_oriented_geometry(orient)

                                    # Apply mirroring if requested
                                    if mirrored:
                                        geom = geom.mirror("XZ")
                                        bb_temp = geom.val().BoundingBox()
                                        geom = geom.translate((-bb_temp.xmin, -bb_temp.ymin, -bb_temp.zmin))

                                    # Translate to position
                                    geom = geom.translate((x, y, z))

                                    bb = get_bounding_box(geom)

                                    if not is_within_stock(bb, self.stock_dims):
                                        continue

                                    overlaps = False
                                    for p in placed_parts + layer_placed:
                                        if boxes_overlap(bb, p.bounding_box, self.saw_kerf):
                                            overlaps = True
                                            break

                                    if not overlaps:
                                        part = MixedPlacedPart(
                                            part_spec_name=part_name,
                                            part_id=part_id,
                                            position=(x, y, z),
                                            orientation_idx=orient_idx,
                                            mirrored=mirrored,
                                            geometry=geom,
                                            bounding_box=bb,
                                            volume=self.part_geometries[part_name].volume
                                        )
                                        layer_placed.append(part)
                                        parts_count[part_name] += 1
                                        total_volume += part.volume
                                        part_id += 1
                                        placed = True
                                        break

                                if placed:
                                    break
                            if placed:
                                break
                        if placed:
                            break

                    if not placed:
                        break

            placed_parts.extend(layer_placed)

            if layer_placed:
                # Move to next layer
                current_z += thickness + self.saw_kerf

        # Iterative space filling: try to fit more parts into remaining space
        if iterative_fill:
            placed_parts, parts_count, total_volume, part_id = self._iterative_fill(
                placed_parts, parts_count, total_volume, part_id, max_iterations=5
            )

        is_extractable = self._check_extractability(placed_parts)
        waste_percentage = (1 - total_volume / self.stock_volume) * 100

        return MixedPackingResult(
            placed_parts=placed_parts,
            parts_by_type={k: v for k, v in parts_count.items() if v > 0},
            total_parts=len(placed_parts),
            total_volume=total_volume,
            waste_percentage=waste_percentage,
            is_extractable=is_extractable,
            stock_volume=self.stock_volume
        )

    def _check_extractability(self, placed_parts: List[MixedPlacedPart]) -> bool:
        """Check if all placed parts can be extracted using guillotine cuts."""
        if not placed_parts:
            return True

        # Convert to Box3D for validator
        boxes = []
        for p in placed_parts:
            bb = p.bounding_box
            boxes.append(Box3D(
                x_min=bb[0], y_min=bb[1], z_min=bb[2],
                x_max=bb[3], y_max=bb[4], z_max=bb[5],
                part_id=p.part_id
            ))

        validator = GuillotineValidatorRecursive(self.stock_dims)
        result = validator.validate(boxes)
        return result.is_valid


def create_stock_geometry(length: float, width: float, height: float) -> cq.Workplane:
    return (cq.Workplane("XY")
            .box(length, width, height)
            .translate((length/2, width/2, height/2)))


def visualize_mixed_packing(result: MixedPackingResult, stock_dims: Tuple,
                            exporter: GeometryExporter, filename: str, title: str):
    """Visualize mixed parts packing."""
    stock_geom = create_stock_geometry(*stock_dims)
    geometries = [(stock_geom, "Stock", "lightgray", 0.15)]

    # Color map for different part types
    part_colors = {
        "G1": "#e74c3c",  # Red
        "G2": "#3498db",  # Blue
        "G3": "#2ecc71",  # Green
        "G4": "#9b59b6",  # Purple
        "G5": "#f39c12",  # Orange
    }

    for part in result.placed_parts:
        color = part_colors.get(part.part_spec_name, "#95a5a6")
        label = f"{part.part_spec_name}_{part.part_id}"
        if part.mirrored:
            label += "_M"
        geometries.append((part.geometry, label, color, 0.8))

    exporter.export_combined(geometries, filename, title)


@dataclass
class SubBlock:
    """A sub-block region created by cutting."""
    origin: Tuple[float, float, float]  # (x0, y0, z0)
    dimensions: Tuple[float, float, float]  # (dx, dy, dz)

    @property
    def x0(self) -> float:
        return self.origin[0]

    @property
    def y0(self) -> float:
        return self.origin[1]

    @property
    def z0(self) -> float:
        return self.origin[2]

    @property
    def x_max(self) -> float:
        return self.origin[0] + self.dimensions[0]

    @property
    def y_max(self) -> float:
        return self.origin[1] + self.dimensions[1]

    @property
    def z_max(self) -> float:
        return self.origin[2] + self.dimensions[2]


def hierarchical_packing(stock_dims: Tuple[float, float, float],
                         primary_part_name: str = "G14",
                         merging_plane_order: str = "XY-X",
                         saw_kerf: float = 0.0,
                         available_parts: Optional[Dict[str, TrapezoidalPrismSpec]] = None,
                         verbose: int = 1):
    """
    Hierarchical packing: Fill block with one part type, then fill sub-blocks with another.

    Args:
        stock_dims: Parent stock block dimensions
        primary_part_name: Part type to fill parent block (e.g., "G14")
        merging_plane_order: Order of merging planes
        saw_kerf: Saw kerf thickness (set to 0 as requested)
        available_parts: Optional dictionary of available parts to use for sub-blocks
        verbose: Print verbosity level (0=minimal, 1=detailed)

    Returns:
        Tuple of (primary_result, sub_blocks, sub_block_results, bounded_region)
    """
    # Use all PART_SPECS if available_parts not provided
    if available_parts is None:
        available_parts = PART_SPECS

    if verbose:
        print(f"\n{'='*80}")
        print("HIERARCHICAL PACKING")
        print(f"{'='*80}")
        print(f"Primary part: {primary_part_name}, Saw kerf: {saw_kerf} mm")

    # PHASE 1: Fill parent block with primary part using tight grid packing
    if verbose:
        print(f"\nPhase 1: Filling parent block with {primary_part_name}...")

    # Use available_parts if provided, otherwise fall back to PART_SPECS
    primary_spec = available_parts.get(primary_part_name, PART_SPECS.get(primary_part_name))
    if primary_spec is None:
        if verbose:
            print(f"  ERROR: Part '{primary_part_name}' not found in available parts!")
        return None, [], []

    primary_geom = TrapezoidGeometry(
        W1=primary_spec.W1, W2=primary_spec.W2,
        D=primary_spec.D, thickness=primary_spec.thickness
    )

    # Get best orientation for primary part
    explorer = OrientationExplorer(primary_geom, Rotational_inc)
    orientations = explorer.filter_fitting_orientations(stock_dims)

    if not orientations:
        if verbose:
            print(f"  ERROR: {primary_part_name} doesn't fit in stock!")
        return None, [], []

    orient = orientations[0] # New use orient which give max packing volume
    # Try to use orientation in which slanting face is inward (XZ plane) otherwise use someother orienttation
    # for oreints in orientations:
    #     if oreints.face == 1 and oreints.rotation_angle == 0:
    #         orient = oreints
    #         break

    if verbose:
        print(f"  Using orientation: dims=({orient.dims[0]:.1f}, {orient.dims[1]:.1f}, {orient.dims[2]:.1f})")

    # Tight grid packing (no gaps, saw_kerf=0)
    dx, dy, dz = orient.dims
    sx, sy, sz = stock_dims

    face_id = orient.face
    # modified due to mirror flippng 
    dx = dx - primary_geom.C * (face_id -1)*(face_id - 2) / 2
    dy = dy - primary_geom.C * face_id*(face_id - 2)
    dz = dz - primary_geom.C * face_id*(face_id - 1) / 2
    
    # Calculate grid
    nx = int(sx / dx)
    ny = int(sy / dy)
    nz = int(sz / dz)

    if verbose:
        print(f"  Grid: {nx} x {ny} x {nz} = {nx*ny*nz} parts")

    # Place parts in tight grid
    primary_parts = []
    part_id = 0

    for iz in range(nz):
        z = iz * dz
        for ix in range(nx):
            x = ix * dx

            # Alternate mirroring along X direction
            # Even ix: normal orientation
            # Odd ix: flipped orientation (180° rotation around vertical axis through center)
            flipped = (ix % 2 == 1)

            for iy in range(ny):
                y = iy * dy
                
                face_map = [ix,iy,iz]
                flipped = (face_map[orient.face] % 2 == 1)

                to_rot_x = 0
                to_rot_z = 1
                if orient.face == 0: # If face is XY plane then rotate along X otherwise rotate along Z
                    to_rot_x = 1
                    to_rot_z = 0

                # Create geometry at origin
                geom = explorer.create_oriented_geometry(orient)

                # Apply mirroring if needed (flip 180° around Z axis to swap wide/narrow sides)
                if flipped:
                    # Get bounding box before translation
                    bbox_temp = geom.val().BoundingBox()
                    cx = (bbox_temp.xmin + bbox_temp.xmax) / 2
                    cy = (bbox_temp.ymin + bbox_temp.ymax) / 2
                    cz = (bbox_temp.zmin + bbox_temp.zmax) / 2

                    # Rotate 180° around Z axis (vertical) to flip wide/narrow orientation
                    geom = geom.rotate((cx, cy, cz), (cx + to_rot_x, cy, cz + to_rot_z), 180)

                    # Re-normalize to origin after rotation
                    bbox_temp = geom.val().BoundingBox()
                    geom = geom.translate((-bbox_temp.xmin, -bbox_temp.ymin, -bbox_temp.zmin))

                # Now translate to final position
                geom = geom.translate((x, y, z))
                bb = get_bounding_box(geom)

                # Check within stock
                if bb[3] <= sx + 0.01 and bb[4] <= sy + 0.01 and bb[5] <= sz + 0.01:
                    part = MixedPlacedPart(
                        part_spec_name=primary_part_name,
                        part_id=part_id,
                        position=(x, y, z),
                        orientation_idx=0,
                        mirrored=flipped,
                        geometry=geom,
                        bounding_box=bb,
                        volume=primary_geom.volume
                    )
                    primary_parts.append(part)
                    part_id += 1


    stock_geom = create_stock_geometry(*stock_dims)
    # packer = RotatedMirroredPacker(stock_dims=stock_dims, part_geom=primary_geom)
    # result = packer.pack()
    # primary_parts = result.placed_parts

    if verbose:
        print(f"  Placed {len(primary_parts)} parts")

    # Create bounding box around primary packing
    if not primary_parts:
        if verbose:
            print("  ERROR: No parts placed!")
        return None, [], []

    # Find tight bounding box
    all_bbs = [p.bounding_box for p in primary_parts]
    bbox_min_x = min(bb[0] for bb in all_bbs)
    bbox_min_y = min(bb[1] for bb in all_bbs)
    bbox_min_z = min(bb[2] for bb in all_bbs)
    bbox_max_x = max(bb[3] for bb in all_bbs)
    bbox_max_y = max(bb[4] for bb in all_bbs)
    bbox_max_z = max(bb[5] for bb in all_bbs)

    bounded_region = (bbox_min_x, bbox_min_y, bbox_min_z, bbox_max_x, bbox_max_y, bbox_max_z)

    if verbose:
        print(f"  Bounding box: ({bbox_min_x:.1f}, {bbox_min_y:.1f}, {bbox_min_z:.1f}) to "
              f"({bbox_max_x:.1f}, {bbox_max_y:.1f}, {bbox_max_z:.1f})")
        print(f"  Dimensions: {bbox_max_x - bbox_min_x:.1f} x {bbox_max_y - bbox_min_y:.1f} x {bbox_max_z - bbox_min_z:.1f}")

    # PHASE 2: Cut bounding box from parent to create 3 merged regions based on merging_plane_order
    if verbose:
        print(f"\nPhase 2: Creating 3 merged regions (merging_plane_order={merging_plane_order})...")

    x0, y0, z0 = 0, 0, 0
    x1, y1, z1 = stock_dims
    cx, cy, cz = bbox_max_x, bbox_max_y, bbox_max_z

    EPSILON = 0.01

    sub_blocks = []

    # Create merged regions based on merging_plane_order
    if merging_plane_order == "XY-X":
        # Region 1: {X+ Y+ Z+, X- Y+ Z+, X+ Y- Z+, X- Y- Z+} - All Z+ (top half)
        if cz < z1 - EPSILON:
            sub_blocks.append(SubBlock(
                origin=(x0, y0, cz),
                dimensions=(x1 - x0, y1 - y0, z1 - cz)
            ))
        # Region 2: {X+ Y+ Z-, X- Y+ Z-} - Y+ (back) in Z- (bottom)
        if cy < y1 - EPSILON:
            sub_blocks.append(SubBlock(
                origin=(x0, cy, z0),
                dimensions=(x1 - x0, y1 - cy, cz)
            ))
        # Region 3: {X+ Y- Z-} - Y- (front) in Z- (bottom)
        sub_blocks.append(SubBlock(
            origin=(cx, y0, z0),
            dimensions=(x1 - cx, cy, cz)
        ))

    elif merging_plane_order == "XY-Y":
        # Region 1: {X+ Y+ Z+, X- Y+ Z+, X+ Y- Z+, X- Y- Z+} - All Z+ (top half)
        if cz < z1 - EPSILON:
            sub_blocks.append(SubBlock(
                origin=(x0, y0, cz),
                dimensions=(x1 - x0, y1 - y0, z1 - cz)
            ))
        # Region 2: {X+ Y- Z-, X+ Y+ Z-} - X+ (right) in Z- (bottom)
        if cx < x1 - EPSILON:
            sub_blocks.append(SubBlock(
                origin=(cx, y0, z0),
                dimensions=(x1 - cx, y1 - y0, cz)
            ))
        # Region 3: {X- Y+ Z-} - X- (left) in Z- (bottom)
        sub_blocks.append(SubBlock(
            origin=(x0, cy, z0),
            dimensions=(cx, y1 - cy, cz)
        ))

    elif merging_plane_order == "XZ-X":
        # Region 1: {X+ Z+ Y+, X- Z+ Y+, X+ Z- Y+, X- Z- Y+} - All Y+ (back half)
        if cy < y1 - EPSILON:
            sub_blocks.append(SubBlock(
                origin=(x0, cy, z0),
                dimensions=(x1 - x0, y1 - cy, z1 - z0)
            ))
        # Region 2: {X+ Z+ Y-, X- Z+ Y-} - Z+ (top) in Y- (front)
        if cz < z1 - EPSILON:
            sub_blocks.append(SubBlock(
                origin=(x0, y0, cz),
                dimensions=(x1 - x0, cy, z1 - cz)
            ))
        # Region 3: {X+ Z- Y-} - Z- (bottom) in Y- (front)
        sub_blocks.append(SubBlock(
            origin=(cx, y0, z0),
            dimensions=(x1 - cx, cy, cz)
        ))

    elif merging_plane_order == "XZ-Z":
        # Region 1: {X+ Z+ Y+, X- Z+ Y+, X+ Z- Y+, X- Z- Y+} - All Y+ (back half)
        if cy < y1 - EPSILON:
            sub_blocks.append(SubBlock(
                origin=(x0, cy, z0),
                dimensions=(x1 - x0, y1 - cy, z1 - z0)
            ))
        # Region 2: {X+ Z- Y-, X+ Z+ Y-} - X+ (right) in Y- (front)
        if cx < x1 - EPSILON:
            sub_blocks.append(SubBlock(
                origin=(cx, y0, z0),
                dimensions=(x1 - cx, cy, z1 - z0)
            ))
        # Region 3: {X- Z+ Y-} - X- (left) in Y- (front)
        sub_blocks.append(SubBlock(
            origin=(x0, y0, cz),
            dimensions=(cx, cy, z1 - cz)
        ))

    elif merging_plane_order == "ZY-Z":
        # Region 1: {Z+ Y+ X+, Z- Y+ X+, Z+ Y- X+, Z- Y- X+} - All X+ (right half)
        if cx < x1 - EPSILON:
            sub_blocks.append(SubBlock(
                origin=(cx, y0, z0),
                dimensions=(x1 - cx, y1 - y0, z1 - z0)
            ))
        # Region 2: {Z+ Y+ X-, Z- Y+ X-} - Y+ (back) in X- (left)
        if cy < y1 - EPSILON:
            sub_blocks.append(SubBlock(
                origin=(x0, cy, z0),
                dimensions=(cx, y1 - cy, z1 - z0)
            ))
        # Region 3: {Z+ Y- X-} - Y- (front) in X- (left)
        sub_blocks.append(SubBlock(
            origin=(x0, y0, cz),
            dimensions=(cx, cy, z1 - cz)
        ))

    elif merging_plane_order == "ZY-Y":
        # Region 1: {Z+ Y+ X+, Z- Y+ X+, Z+ Y- X+, Z- Y- X+} - All X+ (right half)
        if cx < x1 - EPSILON:
            sub_blocks.append(SubBlock(
                origin=(cx, y0, z0),
                dimensions=(x1 - cx, y1 - y0, z1 - z0)
            ))
        # Region 2: {Z+ Y- X-, Z+ Y+ X-} - Z+ (top) in X- (left)
        if cz < z1 - EPSILON:
            sub_blocks.append(SubBlock(
                origin=(x0, y0, cz),
                dimensions=(cx, y1 - y0, z1 - cz)
            ))
        # Region 3: {Z- Y+ X-} - Z- (bottom) in X- (left)
        sub_blocks.append(SubBlock(
            origin=(x0, cy, z0),
            dimensions=(cx, y1 - cy, cz)
        ))

    else:
        if verbose:
            print(f"  WARNING: Unknown merging_plane_order '{merging_plane_order}', using default XY-X")
        # Default to XY-X
        if cz < z1 - EPSILON:
            sub_blocks.append(SubBlock(origin=(x0, y0, cz), dimensions=(x1 - x0, y1 - y0, z1 - cz)))
        if cy < y1 - EPSILON:
            sub_blocks.append(SubBlock(origin=(x0, cy, z0), dimensions=(cx, y1 - cy, cz)))
        if cx < x1 - EPSILON:
            sub_blocks.append(SubBlock(origin=(cx, y0, z0), dimensions=(x1 - cx, cy, cz)))
        exit()

    # Filter valid sub-blocks (positive dimensions)
    sub_blocks = [sb for sb in sub_blocks
                  if sb.dimensions[0] > EPSILON and sb.dimensions[1] > EPSILON and sb.dimensions[2] > EPSILON]

    if verbose:
        print(f"  Created {len(sub_blocks)} merged regions:")
        for i, sb in enumerate(sub_blocks):
            print(f"    Region {i+1}: origin=({sb.x0:.1f}, {sb.y0:.1f}, {sb.z0:.1f}), "
                  f"dims=({sb.dimensions[0]:.1f}, {sb.dimensions[1]:.1f}, {sb.dimensions[2]:.1f})")

    # PHASE 3: Fill each sub-block with best fitting part from G1-G56
    if verbose:
        print(f"\nPhase 3: Filling sub-blocks with parts from G1-G56...")

    sub_block_results = []

    for i, sb in enumerate(sub_blocks):
        if verbose:
            print(f"\n  Sub-block {i+1}: {sb.dimensions[0]:.1f} x {sb.dimensions[1]:.1f} x {sb.dimensions[2]:.1f}")

        # Search for first fitting part type
        fitting_part = None
        max_volume_occupied = 0
        for part_name in available_parts:
            if part_name not in available_parts:
                continue

            spec = available_parts[part_name]
            geom = TrapezoidGeometry(W1=spec.W1, W2=spec.W2, D=spec.D, thickness=spec.thickness)
            exp = OrientationExplorer(geom, Rotational_inc)
            orients = exp.filter_fitting_orientations(sb.dimensions)
            if orients:
                orient = orients[0]
                face_id = orient.face
                        
                fdx, fdy, fdz = orient.dims
                # modified due to mirror flippng 
                #print(f"{merging_plane_order} - {fitting_part} - {face_id}")
                fdx = fdx - geom.C * (face_id -1)*(face_id - 2) / 2
                fdy = fdy - geom.C * face_id*(face_id - 2)
                fdz = fdz - geom.C * face_id*(face_id - 1) / 2
                fnx = int(sb.dimensions[0] / fdx)
                fny = int(sb.dimensions[1] / fdy)
                fnz = int(sb.dimensions[2] / fdz)
                volume_occupied = fnx*fny*fnz*geom.volume

                if orients and volume_occupied > max_volume_occupied:
                    fitting_part = part_name
                    if verbose:
                        print(f"    First fitting part: {part_name}")
                    max_volume_occupied = volume_occupied

        if not fitting_part:
            if verbose:
                print(f"    No fitting part found - skipping")
            sub_block_results.append(None)
            continue
        else:
            if verbose:
                print(f"    Best fitting part: {fitting_part}")
               

        # Pack this part into sub-block
        fit_spec = available_parts[fitting_part]
        fit_geom = TrapezoidGeometry(W1=fit_spec.W1, W2=fit_spec.W2, D=fit_spec.D, thickness=fit_spec.thickness)
        fit_explorer = OrientationExplorer(fit_geom, Rotational_inc)
        fit_orients = fit_explorer.filter_fitting_orientations(sb.dimensions)
        fit_orient = fit_orients[0]

        face_id = fit_orient.face
                    
        fdx, fdy, fdz = fit_orient.dims
        # modified due to mirror flippng 
        #print(f"{merging_plane_order} - {fitting_part} - {face_id}")
        fdx = fdx - fit_geom.C * (face_id -1)*(face_id - 2) / 2
        fdy = fdy - fit_geom.C * face_id*(face_id - 2)
        fdz = fdz - fit_geom.C * face_id*(face_id - 1) / 2
        fnx = int(sb.dimensions[0] / fdx)
        fny = int(sb.dimensions[1] / fdy)
        fnz = int(sb.dimensions[2] / fdz)

        if verbose:
            print(f"    Packing: {fnx} x {fny} x {fnz} = {fnx*fny*fnz} parts")

        sub_parts = []
        sub_part_id = 0

        for iz in range(fnz):
            z = sb.z0 + iz * fdz
            for ix in range(fnx):
                x = sb.x0 + ix * fdx

                for iy in range(fny):
                    y = sb.y0 + iy * fdy

                    # Create geometry at origin
                    geom = fit_explorer.create_oriented_geometry(fit_orient)

                    face_map = [ix,iy,iz]
                    
                    to_rot_x = 0
                    to_rot_z = 1
                    if face_id == 0: # If face is XY plane then rotate along X otherwise rotate along Z
                        to_rot_x = 1
                        to_rot_z = 0
                    flipped = (face_map[face_id] % 2 == 1)

                    # Apply mirroring if needed (flip 180° around Z axis)
                    if flipped:
                        bbox_temp = geom.val().BoundingBox()
                        cx = (bbox_temp.xmin + bbox_temp.xmax) / 2
                        cy = (bbox_temp.ymin + bbox_temp.ymax) / 2
                        cz = (bbox_temp.zmin + bbox_temp.zmax) / 2

                        # Rotate 180° around Z axis to flip wide/narrow orientation
                        geom = geom.rotate((cx, cy, cz), (cx + to_rot_x, cy, cz + to_rot_z), 180)

                        # Re-normalize to origin
                        bbox_temp = geom.val().BoundingBox()
                        #geom = geom.translate((-bbox_temp.xmin + ix * fit_geom.C * (face_id -1)*(face_id - 2) / 2, -bbox_temp.ymin + iy * fit_geom.C * face_id*(face_id - 2), -bbox_temp.zmin + iz * fit_geom.C * face_id*(face_id - 1) / 2))
                        geom = geom.translate((-bbox_temp.xmin , -bbox_temp.ymin , -bbox_temp.zmin ))

                    # Translate to final position
                    geom = geom.translate((x, y, z))
                    bb = get_bounding_box(geom)

                    if bb[3] <= sb.x_max + 0.01 and bb[4] <= sb.y_max + 0.01 and bb[5] <= sb.z_max + 0.01:
                        part = MixedPlacedPart(
                            part_spec_name=fitting_part,
                            part_id=sub_part_id,
                            position=(x, y, z),
                            orientation_idx=0,
                            mirrored=flipped,
                            geometry=geom,
                            bounding_box=bb,
                            volume=fit_geom.volume
                        )
                        sub_parts.append(part)
                        sub_part_id += 1

        if verbose:
            print(f"    Placed {len(sub_parts)} parts")
        sub_block_results.append((fitting_part, sub_parts))

    # Create results
    primary_result = MixedPackingResult(
        placed_parts=primary_parts,
        parts_by_type={primary_part_name: len(primary_parts)},
        total_parts=len(primary_parts),
        total_volume=len(primary_parts) * primary_geom.volume,
        waste_percentage=(1 - (len(primary_parts) * primary_geom.volume) / (stock_dims[0] * stock_dims[1] * stock_dims[2])) * 100,
        is_extractable=True,
        stock_volume=stock_dims[0] * stock_dims[1] * stock_dims[2]
    )

    return primary_result, sub_blocks, sub_block_results, bounded_region


def read_parts_from_excel(excel_file: str) -> Dict[str, TrapezoidalPrismSpec]:
    """
    Read trapezoid specifications from Excel file.

    Args:
        excel_file: Path to Excel file

    Returns:
        Dictionary of part_name -> TrapezoidalPrismSpec
    """
    print(f"\nReading parts from Excel file: {excel_file}")

    try:
        # First, try to read all sheets to find one with required columns
        xl = pd.ExcelFile(excel_file)
        df = None

        # Try each sheet to find one with MARK column
        for sheet_name in xl.sheet_names:
            try:
                temp_df = pd.read_excel(excel_file, sheet_name=sheet_name)
                if 'MARK' in temp_df.columns:
                    df = temp_df
                    print(f"  Using sheet: {sheet_name}")
                    break
            except:
                continue

        # If no sheet found with MARK column, try first sheet
        if df is None:
            df = pd.read_excel(excel_file, sheet_name=0)

        # Expected columns: MARK, A(W1), B(W2), C, D(length), Thickness, α
        # We need: MARK, A, B, D, Thickness

        parts = {}
        for idx, row in df.iterrows():
            try:
                mark = str(row['MARK']).strip()

                # Try different column name variations
                w1 = row.get('A(W1)', row.get('A', row.get('W1', None)))
                w2 = row.get('B(W2)', row.get('B', row.get('W2', None)))
                d = row.get('D(length)', row.get('D', row.get('Length', None)))
                thickness = row.get('Thickness', row.get('thickness', row.get('THICKNESS', None)))

                # Alpha is optional, default to 2.168 if not provided
                alpha = row.get('(α)', row.get('α', row.get('alpha', row.get('Alpha', 2.168))))

                # Skip rows with missing critical data
                if w1 is None or w2 is None or d is None or thickness is None:
                    continue

                # Skip NaN values
                if pd.isna(w1) or pd.isna(w2) or pd.isna(d) or pd.isna(thickness):
                    continue

                parts[mark] = TrapezoidalPrismSpec(
                    name=mark,
                    W1=float(w1),
                    W2=float(w2),
                    D=float(d),
                    thickness=float(thickness),
                    alpha=float(alpha) if not pd.isna(alpha) else 2.168
                )
            except Exception as e:
                # Skip rows that can't be parsed
                continue

        if not parts:
            raise ValueError("No valid parts found in Excel file. Please ensure it has columns: MARK, A(W1), B(W2), D(length), Thickness")

        print(f"  Successfully read {len(parts)} parts from Excel")
        return parts

    except Exception as e:
        print(f"  ERROR: Failed to read Excel file: {e}")
        sys.exit(1)


def get_user_configuration():
    """
    Get configuration from user interactively.

    Returns:
        Tuple of (stock_dims, parts_to_use, stock_name, output_dir, verbose)
    """
    print("\n" + "=" * 80)
    print("INTERACTIVE CONFIGURATION")
    print("=" * 80)

    # Ask for output directory
    output_dir_input = input("\nEnter output folder name (press Enter for default 'step5_mixed_parts'): ").strip()
    if output_dir_input:
        output_dir = os.path.join("outputs/visualizations/step5_mixed_parts", output_dir_input)
    else:
        output_dir = "outputs/visualizations/step5_mixed_parts"
    print(f"  Output directory: {output_dir}")

    # Set verbose to 0 for interactive mode
    verbose = 0

    # Ask if using default configuration
    use_default = input("\nUse default configuration? (y/n): ").strip().lower()

    if use_default in ['y', 'yes']:
        # Select stock block
        print("\nAvailable Stock Blocks:")
        print("-" * 80)
        for idx, (key, spec) in enumerate(STOCK_BLOCKS.items(), 1):
            print(f"{idx}. {key}: {spec.name}")
            print(f"   Dimensions: {spec.length} x {spec.width} x {spec.height} mm")
            print(f"   Volume: {spec.volume:,.0f} mm³")

        stock_choice = input(f"\nSelect stock block (1-{len(STOCK_BLOCKS)}): ").strip()
        try:
            stock_idx = int(stock_choice) - 1
            stock_key = list(STOCK_BLOCKS.keys())[stock_idx]
            stock = STOCK_BLOCKS[stock_key]
            stock_dims = (stock.length, stock.width, stock.height)
            stock_name = stock_key
            print(f"  Selected: {stock.name}")
        except (ValueError, IndexError):
            print("  Invalid selection. Using default 'size_2'")
            stock = STOCK_BLOCKS['size_2']
            stock_dims = (stock.length, stock.width, stock.height)
            stock_name = 'size_2'

        # Select parts
        print("\nAvailable Parts: G1 to G56")
        print("-" * 80)
        print("Enter part selection:")
        print("  - For range: 'G1-G15' (uses G1 through G15)")
        print("  - For specific parts: 'G1 G10 G20 G48 G51'")
        print("  - For all parts: 'all' or 'G1-G56'")

        parts_input = input("\nPart selection: ").strip()

        if parts_input.lower() == 'all':
            parts_to_use = {name: PART_SPECS[name] for name in PART_SPECS if name in PART_SPECS}
        elif '-' in parts_input:
            # Range format: G1-G15
            try:
                start, end = parts_input.split('-')
                start_num = int(start.replace('G', '').strip())
                end_num = int(end.replace('G', '').strip())
                parts_to_use = {}
                for i in range(start_num, end_num + 1):
                    part_name = f"G{i}"
                    if part_name in PART_SPECS:
                        parts_to_use[part_name] = PART_SPECS[part_name]
                print(f"  Selected {len(parts_to_use)} parts: G{start_num} to G{end_num}")
            except Exception as e:
                print(f"  ERROR parsing range: {e}. Using G1-G56")
                parts_to_use = {name: PART_SPECS[name] for name in PART_SPECS}
        else:
            # Specific parts: G1 G10 G20
            part_names = parts_input.split()
            parts_to_use = {}
            for name in part_names:
                name = name.strip().upper()
                if name in PART_SPECS:
                    parts_to_use[name] = PART_SPECS[name]
                else:
                    print(f"  WARNING: Part '{name}' not found, skipping")

            if not parts_to_use:
                print("  No valid parts selected. Using all parts G1-G56")
                parts_to_use = {name: PART_SPECS[name] for name in PART_SPECS}
            else:
                print(f"  Selected {len(parts_to_use)} parts: {', '.join(sorted(parts_to_use.keys()))}")

    else:
        # Manual configuration
        print("\nManual Configuration")
        print("-" * 80)

        # Get stock dimensions
        print("\nEnter parent block dimensions (in mm):")
        try:
            length = float(input("  Length: ").strip())
            width = float(input("  Width: ").strip())
            height = float(input("  Height: ").strip())
            stock_dims = (length, width, height)
            stock_name = "custom"
            print(f"  Stock block: {length} x {width} x {height} mm")
        except ValueError:
            print("  ERROR: Invalid dimensions. Using default size_2")
            stock = STOCK_BLOCKS['size_2']
            stock_dims = (stock.length, stock.width, stock.height)
            stock_name = 'size_2'

        # Get Excel file for parts
        excel_file = input("\nEnter Excel file path for trapezoid dimensions: ").strip()

        if not os.path.exists(excel_file):
            print(f"  ERROR: File '{excel_file}' not found. Using default parts G1-G56")
            parts_to_use = {name: PART_SPECS[name] for name in PART_SPECS if name == "G14"}
        else:
            parts_to_use = read_parts_from_excel(excel_file)

    return stock_dims, parts_to_use, stock_name, output_dir, verbose


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Mixed Parts Packing Optimization')
    parser.add_argument('-i', '--interactive', action='store_true',
                       help='Run in interactive mode to configure stock and parts')
    args = parser.parse_args()

    # Get configuration (interactive or default)
    if args.interactive:
        stock_dims, parts_to_pack, stock_name, output_dir, verbose = get_user_configuration()
    else:
        # Default configuration
        output_dir = OUTPUT_DIR
        verbose = 1
        stock_name = "size_2"
        stock = STOCK_BLOCKS[stock_name]
        stock_dims = (stock.length, stock.width, stock.height)
        parts_to_pack = {name: PART_SPECS[name] for name in PART_SPECS}  # Use all parts G1-G56

    os.makedirs(output_dir, exist_ok=True)
    exporter = GeometryExporter(output_dir)

    if verbose:
        print("=" * 80)
        print("STEP 5c: MIXED PARTS PACKING (G1-G5)")
        print("=" * 80)

    # Display configuration
    if verbose:
        if stock_name in STOCK_BLOCKS:
            stock = STOCK_BLOCKS[stock_name]
            print(f"\nStock Block: {stock.name}")
            print(f"  Dimensions: {stock.length} x {stock.width} x {stock.height} mm")
            print(f"  Volume: {stock.length * stock.width * stock.height:,.0f} mm³")
        else:
            print(f"\nStock Block: Custom")
            print(f"  Dimensions: {stock_dims[0]} x {stock_dims[1]} x {stock_dims[2]} mm")
            print(f"  Volume: {stock_dims[0] * stock_dims[1] * stock_dims[2]:,.0f} mm³")

        print(f"\nSaw kerf: {DEFAULT_SAW_KERF} mm")

        print(f"\n{'='*80}")
        print("PARTS TO PACK:")
        print(f"{'='*80}")
        print(f"{'Name':<6} {'W1':>8} {'W2':>8} {'D':>8} {'Thick':>8} {'Volume':>12}")
        print("-" * 60)

    total_part_volume = 0
    for name, spec in parts_to_pack.items():
        geom = TrapezoidGeometry(W1=spec.W1, W2=spec.W2, D=spec.D, thickness=spec.thickness)
        total_part_volume += geom.volume
        if verbose:
            print(f"{name:<6} {spec.W1:>8.1f} {spec.W2:>8.1f} {spec.D:>8.1f} {spec.thickness:>8.1f} {geom.volume:>12,.0f}")

    if verbose:
        print("-" * 60)
        print(f"{'Total':<6} {'':<8} {'':<8} {'':<8} {'':<8} {total_part_volume:>12,.0f}")
    else:
        print(f"Packing {len(parts_to_pack)} parts")

    # Create packer
    if verbose:
        print(f"\n{'='*80}")
        print("RUNNING PACKING ALGORITHMS...")
        print(f"{'='*80}")

    # packer = MixedPartsPacker(
    #     stock_dims=stock_dims,
    #     part_specs=parts_to_pack,
    #     saw_kerf=DEFAULT_SAW_KERF
    # )

    # # Show available orientations per part
    # print("\nAvailable orientations per part:")
    # for name in parts_to_pack:
    #     num_orient = len(packer.part_orientations.get(name, []))
    #     print(f"  {name}: {num_orient} orientations fit in stock")

    # results: List[Tuple[str, MixedPackingResult]] = []

    # # Algorithm 1: Greedy largest-first
    # print("\n1. Greedy Largest-First Algorithm...")
    # result1 = packer.pack_greedy_largest_first(max_parts_per_type=5)
    # results.append(("Greedy_LargestFirst", result1))
    # print(f"   Parts: {result1.total_parts}, Waste: {result1.waste_percentage:.2f}%, "
    #       f"Extractable: {result1.is_extractable}")
    # print(f"   Parts by type: {result1.parts_by_type}")

    # # Algorithm 2: Layer-based by thickness
    # print("\n2. Layer-Based by Thickness Algorithm...")
    # result2 = packer.pack_by_thickness_layers(max_parts_per_type=5)
    # results.append(("LayerBased_Thickness", result2))
    # print(f"   Parts: {result2.total_parts}, Waste: {result2.waste_percentage:.2f}%, "
    #       f"Extractable: {result2.is_extractable}")
    # print(f"   Parts by type: {result2.parts_by_type}")

    # # Find best result
    # print(f"\n{'='*80}")
    # print("RESULTS SUMMARY")
    # print(f"{'='*80}")

    # extractable_results = [(name, r) for name, r in results if r.is_extractable]

    # if extractable_results:
    #     best_name, best_result = min(extractable_results, key=lambda x: x[1].waste_percentage)

    #     print(f"\nBest Extractable Packing: {best_name}")
    #     print(f"  Total parts: {best_result.total_parts}")
    #     print(f"  Parts volume: {best_result.total_volume:,.0f} mm³")
    #     print(f"  Waste: {best_result.waste_percentage:.2f}%")
    #     print(f"  Parts by type:")
    #     for part_name, count in sorted(best_result.parts_by_type.items()):
    #         print(f"    {part_name}: {count}")

    #     # Detailed part placement
    #     print(f"\n  Part placements:")
    #     for part in best_result.placed_parts:
    #         print(f"    {part.part_spec_name}_{part.part_id}: "
    #               f"pos=({part.position[0]:.1f}, {part.position[1]:.1f}, {part.position[2]:.1f}), "
    #               f"orient={part.orientation_idx}, mirror={part.mirrored}")
    # else:
    #     print("\nNo extractable packing found!")
    #     best_name, best_result = min(results, key=lambda x: x[1].waste_percentage)
    #     print(f"\nBest non-extractable: {best_name}")
    #     print(f"  Total parts: {best_result.total_parts}")
    #     print(f"  Waste: {best_result.waste_percentage:.2f}%")

    # # Generate visualizations
    # print(f"\n{'='*80}")
    # print("GENERATING VISUALIZATIONS")
    # print(f"{'='*80}")

    # for algo_name, result in results:
    #     if result.total_parts > 0:
    #         filename = f"mixed_{algo_name}_{result.total_parts}parts"
    #         title = (f"{algo_name}: {result.total_parts} parts, "
    #                  f"{result.waste_percentage:.1f}% waste, "
    #                  f"Extract: {result.is_extractable}")
    #         visualize_mixed_packing(result, stock_dims, exporter, filename, title)
    #         print(f"  {algo_name}: {filename}.html")

    # # Save report
    # report_file = os.path.join(OUTPUT_DIR, "mixed_parts_report.txt")
    # with open(report_file, 'w') as f:
    #     f.write("=" * 80 + "\n")
    #     f.write("MIXED PARTS PACKING REPORT (G1-G5)\n")
    #     f.write("=" * 80 + "\n\n")

    #     f.write(f"Stock: {stock.length} x {stock.width} x {stock.height} mm\n")
    #     f.write(f"Stock Volume: {stock.length * stock.width * stock.height:,.0f} mm³\n\n")

    #     f.write("PARTS SPECIFICATIONS:\n")
    #     f.write("-" * 60 + "\n")
    #     for name, spec in parts_to_pack.items():
    #         geom = TrapezoidGeometry(W1=spec.W1, W2=spec.W2, D=spec.D, thickness=spec.thickness)
    #         f.write(f"{name}: W1={spec.W1:.1f}, W2={spec.W2:.1f}, D={spec.D:.1f}, "
    #                 f"t={spec.thickness:.1f}, vol={geom.volume:,.0f}\n")

    #     f.write("\nRESULTS:\n")
    #     f.write("-" * 60 + "\n")
    #     for algo_name, result in results:
    #         f.write(f"\n{algo_name}:\n")
    #         f.write(f"  Parts: {result.total_parts}\n")
    #         f.write(f"  Volume: {result.total_volume:,.0f} mm³\n")
    #         f.write(f"  Waste: {result.waste_percentage:.2f}%\n")
    #         f.write(f"  Extractable: {result.is_extractable}\n")
    #         f.write(f"  Parts by type: {result.parts_by_type}\n")

    # print(f"\nReport saved to: {report_file}")
    # print(f"All files saved to: {OUTPUT_DIR}/")

    # # Run hierarchical packing
    # print(f"\n{'='*80}")
    # print("RUNNING HIERARCHICAL PACKING")
    # print(f"{'='*80}")

# This file contains the loop implementation to add to step5_mixed_parts.py main() function
# Replace the section from "hier_result = hierarchical_packing..." to the end

    # Loop over all part types G1-G56 as primary part
    if verbose:
        print(f"\n{'='*80}")
        print("RUNNING HIERARCHICAL PACKING FOR ALL PART TYPES")
        print(f"{'='*80}")

    all_configurations = []
    merging_plane_order_set = ["XY-X","XY-Y","XZ-Z","XZ-X","ZY-Y","ZY-Z"]

    # Get list of parts to test as primary parts
    parts_to_test = sorted(parts_to_pack.keys())

    for idx, primary_part_name in enumerate(parts_to_test, 1):
        if primary_part_name not in parts_to_pack:
            continue

        if verbose:
            print(f"\n{'='*80}")
            print(f"TESTING PRIMARY PART: {primary_part_name} ({idx}/{len(parts_to_test)})")
            print(f"{'='*80}")
        else:
            print(f"Testing {primary_part_name} ({idx}/{len(parts_to_test)})")

        for merging_plane_order in merging_plane_order_set:

            try:
                # stock_dims already defined earlier
                hier_result = hierarchical_packing(
                    stock_dims=stock_dims,
                    primary_part_name=primary_part_name,
                    merging_plane_order=merging_plane_order,
                    saw_kerf=0.0,
                    available_parts=parts_to_pack,
                    verbose=verbose
                )

                if hier_result[0] is None:
                    if verbose:
                        print(f"  Skipping {primary_part_name} - no valid packing")
                    continue

                primary_result, sub_blocks, sub_block_results, bounded_region = hier_result

                # Calculate parts by type
                parts_by_type = {k: v for k, v in primary_result.parts_by_type.items()}
                total_sub_parts = 0

                for result in sub_block_results:
                    if result is not None:
                        part_name, parts = result
                        parts_by_type[part_name] = parts_by_type.get(part_name, 0) + len(parts)
                        total_sub_parts += len(parts)

                # Calculate total volume properly
                total_volume = 0.0
                for pname, count in parts_by_type.items():
                    if pname in parts_to_pack:
                        spec = parts_to_pack[pname]
                        part_vol = ((spec.W1 + spec.W2) / 2.0) * spec.D * spec.thickness
                        total_volume += part_vol * count

                total_all_parts = len(primary_result.placed_parts) + total_sub_parts
                waste_percentage = (1 - total_volume / (stock_dims[0] * stock_dims[1] * stock_dims[2])) * 100

                # Store configuration
                all_configurations.append({
                    'primary_part': primary_part_name + merging_plane_order,
                    'parts_by_type': parts_by_type.copy(),
                    'total_parts': total_all_parts,
                    'total_volume': total_volume,
                    'waste_percentage': waste_percentage,
                    'primary_result': primary_result,
                    'sub_blocks': sub_blocks,
                    'sub_block_results': sub_block_results,
                    'bounded_region': bounded_region
                })
                config_name = primary_part_name + merging_plane_order
                # Visualization 1: Sub-blocks and bounded primary packing
                if verbose:
                    print(f"\n  Generating visualizations for {config_name}...")

                stock_geom = create_stock_geometry(*stock_dims)
                geometries = [(stock_geom, "Stock", "lightgray", 0.05)]

                # Add primary parts
                for part in primary_result.placed_parts:
                    label = f"{config_name}_{part.part_id}"
                    geometries.append((part.geometry, label, "#2ecc71", 0.6))

                # Add bounding box wireframe
                bbox_min_x, bbox_min_y, bbox_min_z, bbox_max_x, bbox_max_y, bbox_max_z = bounded_region
                bbox_geom = (cq.Workplane("XY")
                            .box(bbox_max_x - bbox_min_x, bbox_max_y - bbox_min_y, bbox_max_z - bbox_min_z)
                            .translate((bbox_min_x + (bbox_max_x - bbox_min_x)/2,
                                        bbox_min_y + (bbox_max_y - bbox_min_y)/2,
                                        bbox_min_z + (bbox_max_z - bbox_min_z)/2)))
                geometries.append((bbox_geom, "BoundingBox", "#e74c3c", 0.1))

                # Add sub-block wireframes
                sub_colors = ["#3498db", "#9b59b6", "#f39c12", "#1abc9c", "#e67e22", "#8e44ad", "#16a085"]
                for i, sb in enumerate(sub_blocks):
                    sb_geom = (cq.Workplane("XY")
                            .box(sb.dimensions[0], sb.dimensions[1], sb.dimensions[2])
                            .translate((sb.x0 + sb.dimensions[0]/2,
                                        sb.y0 + sb.dimensions[1]/2,
                                        sb.z0 + sb.dimensions[2]/2)))
                    color = sub_colors[i % len(sub_colors)]
                    geometries.append((sb_geom, f"SubBlock{i+1}", color, 0.15))

                filename = f"/mnt/data_drive/cutting_blocks/backend/outputs/visualizations/hierarchical_{config_name}_step1_subblocks"
                title = f"Hierarchical Step 1 ({config_name}): {len(primary_result.placed_parts)} {primary_part_name} parts + {len(sub_blocks)} sub-blocks"
                exporter.export_combined(geometries, filename, title)
                if verbose:
                    print(f"    Saved: {filename}.html")

                # Visualization 2: Complete hierarchical packing with filled sub-blocks
                geometries2 = [(stock_geom, "Stock", "lightgray", 0.05)]

                # Add primary parts
                for part in primary_result.placed_parts:
                    label = f"{config_name}_{part.part_id}"
                    geometries2.append((part.geometry, label, "#2ecc71", 0.7))

                # Sub-block colors - each sub-block gets a unique color
                sub_block_colors = ["#e74c3c", "#3498db", "#9b59b6", "#f39c12", "#1abc9c", "#e67e22", "#16a085"]

                for i, result in enumerate(sub_block_results):
                    if result is None:
                        continue
                    part_name, parts = result
                    color = sub_block_colors[i % len(sub_block_colors)]

                    for part in parts:
                        label = f"{part_name}_{part.part_id}_SB{i+1}"
                        geometries2.append((part.geometry, label, color, 0.7))

                # Create title in format "20 G14 + 4 G13 + 1 G7 and 45% Waste"
                parts_summary = " + ".join([f"{count} {pname}" for pname, count in sorted(parts_by_type.items(), key=lambda x: -x[1])]) + str(f" and {waste_percentage:.1f} % Waste") 

                filename2 = f"hierarchical_{config_name}_step2_complete"
                title2 = f"Hierarchical Step 2 ({config_name}): {parts_summary}"
                exporter.export_combined(geometries2, filename2, title2)
                if verbose:
                    print(f"    Saved: {filename2}.html")

                    # Print summary for this configuration
                    print(f"\n  Summary for {primary_part_name}:")
                    print(f"    Total parts: {total_all_parts}")
                    print(f"    Waste: {waste_percentage:.2f}%")
                    print(f"    Parts: {parts_summary}")

            except Exception as e:
                if verbose:
                    print(f"  ERROR processing {config_name}: {e}")
                import traceback
                traceback.print_exc()
                continue

    # Sort all configurations by waste percentage (descending as requested by user)
    all_configurations.sort(key=lambda x: x['waste_percentage'], reverse=True)

    # Save summary report
    if verbose:
        print(f"\n{'='*80}")
        print("SAVING HIERARCHICAL PACKING SUMMARY REPORT")
        print(f"{'='*80}")

    report_file = os.path.join(output_dir, "hierarchical_packing_summary.txt")
    with open(report_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("HIERARCHICAL PACKING SUMMARY - ALL CONFIGURATIONS\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Stock: {stock_dims[0]} x {stock_dims[1]} x {stock_dims[2]} mm\n")
        f.write(f"Stock Volume: {stock_dims[0] * stock_dims[1] * stock_dims[2]:,.0f} mm³\n\n")
        f.write("Configurations sorted by total waste percentage (descending):\n")
        f.write("-" * 80 + "\n\n")

        for rank, config in enumerate(all_configurations, 1):
            parts_summary = ", ".join([f"{count} {pname}" for pname, count in sorted(config['parts_by_type'].items(), key=lambda x: -x[1])])
            f.write(f"Rank {rank}: Primary Part = {config['primary_part']}\n")
            f.write(f"  Total Parts: {config['total_parts']} ({parts_summary})\n")
            f.write(f"  Total Volume: {config['total_volume']:,.0f} mm³\n")
            f.write(f"  Waste: {config['waste_percentage']:.2f}%\n")
            f.write(f"  Parts breakdown:\n")
            for part_name, count in sorted(config['parts_by_type'].items(), key=lambda x: -x[1]):
                f.write(f"    {part_name}: {count}\n")
            f.write("\n")

    if verbose:
        print(f"  Report saved to: {report_file}")
    else:
        print(f"\nCompleted: Report saved to {report_file}")

    # Print top 10 configurations
    if verbose:
        print(f"\n{'='*80}")
        print("TOP 10 CONFIGURATIONS (lowest waste first)")
        print(f"{'='*80}")

        # Sort by ascending waste for display (best first)
        display_configs = sorted(all_configurations, key=lambda x: x['waste_percentage'])

        for rank, config in enumerate(display_configs[:10], 1):
            parts_summary = ", ".join([f"{count} {pname}" for pname, count in sorted(config['parts_by_type'].items(), key=lambda x: -x[1])])
            print(f"\n{rank}. Primary: {config['primary_part']} | Waste: {config['waste_percentage']:.2f}%")
            print(f"   Parts: {parts_summary}")
    else:
        # For minimal output, just show the best configuration
        if all_configurations:
            best = min(all_configurations, key=lambda x: x['waste_percentage'])
            parts_summary = ", ".join([f"{count} {pname}" for pname, count in sorted(best['parts_by_type'].items(), key=lambda x: -x[1])])
            print(f"Best: {best['primary_part']} | Waste: {best['waste_percentage']:.2f}% | Parts: {parts_summary}")


if __name__ == "__main__":
    main()
