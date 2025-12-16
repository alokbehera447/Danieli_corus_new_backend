"""
Enhanced Greedy Packing with Multiple Algorithms.

Implements multiple placement strategies for generating high-quality baseline packings:
1. Bottom-Left (BL) Heuristic - Place at lowest-leftmost valid position
2. Best-Fit (BF) Heuristic - Score positions and choose best
3. Skyline Heuristic - Maintain and fill skyline gaps
4. Layer-Based Packing - Pack in horizontal layers

All strategies support:
- Multiple orientations from OrientationExplorer
- Mirrored pair detection and placement
- Extractability validation via guillotine check
- Saw kerf spacing (0.1mm default)
"""

import cadquery as cq
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field
from enum import Enum

from .trapezoid_geometry import TrapezoidGeometry, create_trapezoid_cq
from .orientation_explorer import OrientationExplorer, Orientation
from .packing_algorithms import PlacedPart, PackingResult
from ..cutting.guillotine_validator import (
    GuillotineValidatorRecursive,
    GuillotineValidatorGraph,
    Box3D
)


# =============================================================================
# CONSTANTS
# =============================================================================

DEFAULT_SAW_KERF = 0.0  # mm - spacing between parts for saw blade


class PackingAlgorithm(Enum):
    """Available packing algorithms."""
    BOTTOM_LEFT = "bottom_left"
    BEST_FIT = "best_fit"
    SKYLINE = "skyline"
    LAYER_BASED = "layer_based"
    MIRRORED_PAIRS = "mirrored_pairs"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class PackingCandidate:
    """A candidate packing solution for comparison."""
    algorithm: str
    orientation: Orientation
    placed_parts: List[PlacedPart]
    num_parts: int
    waste_percentage: float
    is_extractable: bool
    stock_dims: Tuple[float, float, float]
    part_volume: float
    mirrored_pairs: int = 0
    total_cuts: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def stock_volume(self) -> float:
        return self.stock_dims[0] * self.stock_dims[1] * self.stock_dims[2]

    @property
    def parts_volume(self) -> float:
        return self.num_parts * self.part_volume

    @property
    def leftover_volume(self) -> float:
        return self.stock_volume - self.parts_volume

    def to_packing_result(self) -> PackingResult:
        """Convert to PackingResult for compatibility."""
        return PackingResult(
            algorithm_name=f"{self.algorithm} (Ori {self.orientation.index})",
            stock_dimensions=self.stock_dims,
            part_dimensions=self.orientation.dims,
            placed_parts=self.placed_parts,
            stock_volume=self.stock_volume,
            parts_volume=self.parts_volume,
            num_parts=self.num_parts
        )


@dataclass
class MirroredPairInfo:
    """Information about a mirrored pair placement."""
    part1_pos: Tuple[float, float, float]
    part2_pos: Tuple[float, float, float]
    combined_bbox: Tuple[float, float, float]
    space_saved: float


# =============================================================================
# COLLISION DETECTION
# =============================================================================

def boxes_overlap(box1: Tuple[float, float, float, float, float, float],
                  box2: Tuple[float, float, float, float, float, float],
                  eps: float = 1e-6) -> bool:
    """Check if two axis-aligned boxes overlap."""
    x1_min, y1_min, z1_min, x1_max, y1_max, z1_max = box1
    x2_min, y2_min, z2_min, x2_max, y2_max, z2_max = box2

    return not (
        x1_max <= x2_min + eps or x2_max <= x1_min + eps or
        y1_max <= y2_min + eps or y2_max <= y1_min + eps or
        z1_max <= z2_min + eps or z2_max <= z1_min + eps
    )


def is_within_bounds(box: Tuple[float, float, float, float, float, float],
                     stock_dims: Tuple[float, float, float],
                     eps: float = 1e-6) -> bool:
    """Check if box is within stock bounds."""
    x_min, y_min, z_min, x_max, y_max, z_max = box
    sx, sy, sz = stock_dims

    return (
        x_min >= -eps and y_min >= -eps and z_min >= -eps and
        x_max <= sx + eps and y_max <= sy + eps and z_max <= sz + eps
    )


def get_placed_bbox(position: Tuple[float, float, float],
                    dims: Tuple[float, float, float]) -> Tuple[float, float, float, float, float, float]:
    """Get bounding box for a part at given position."""
    x, y, z = position
    dx, dy, dz = dims
    return (x, y, z, x + dx, y + dy, z + dz)


def check_collision(position: Tuple[float, float, float],
                    dims: Tuple[float, float, float],
                    placed_boxes: List[Tuple[float, float, float, float, float, float]],
                    stock_dims: Tuple[float, float, float]) -> bool:
    """Check if placing a part at position would cause collision or go out of bounds."""
    new_box = get_placed_bbox(position, dims)

    # Check bounds
    if not is_within_bounds(new_box, stock_dims):
        return True

    # Check collisions with existing parts
    for box in placed_boxes:
        if boxes_overlap(new_box, box):
            return True

    return False


# =============================================================================
# MIRRORED PAIR DETECTION
# =============================================================================

def can_use_mirrored_pair(part_geom: TrapezoidGeometry) -> Tuple[bool, MirroredPairInfo]:
    """
    Check if part benefits from mirroring and calculate space savings.

    Returns:
        Tuple of (can_use: bool, info: MirroredPairInfo)
    """
    # Check if part is symmetric (no benefit from mirroring)
    if abs(part_geom.W1 - part_geom.W2) < 0.01:
        return False, None

    # Calculate bounding boxes
    # Normal orientation: W1 at bottom, W2 at top
    normal_dims = (part_geom.W1, part_geom.D, part_geom.thickness)

    # Two separate parts stacked
    two_separate_height = 2 * part_geom.thickness

    # Mirrored pair: parts nested with angled faces touching
    # When mirrored, the narrow ends interlock
    # Combined height is still 2 * thickness (no height savings in Z)
    # But we save space in the X direction due to the taper
    pair_width = part_geom.W1  # Width stays the same
    pair_depth = part_geom.D * 2  # Depth doubles for side-by-side
    pair_height = part_geom.thickness

    # Alternative: stack vertically with Z-flip
    # Height savings from nested trapezoids
    # The offset C allows some nesting
    nested_height = 2 * part_geom.thickness - (part_geom.C * 0.5)  # Approximate

    # Calculate space saved
    normal_volume_for_two = part_geom.W1 * part_geom.D * two_separate_height
    pair_volume = pair_width * part_geom.D * nested_height

    space_saved = normal_volume_for_two - pair_volume

    if space_saved > 0:
        info = MirroredPairInfo(
            part1_pos=(0, 0, 0),
            part2_pos=(0, 0, part_geom.thickness),
            combined_bbox=(pair_width, part_geom.D, nested_height),
            space_saved=space_saved
        )
        return True, info

    return False, None


def validate_mirrored_pair_extractable(part1_bbox: Tuple, part2_bbox: Tuple,
                                        stock_dims: Tuple[float, float, float]) -> bool:
    """Check if a mirrored pair can be extracted with guillotine cuts."""
    boxes = [
        Box3D(part1_bbox[0], part1_bbox[1], part1_bbox[2],
              part1_bbox[3], part1_bbox[4], part1_bbox[5], 0),
        Box3D(part2_bbox[0], part2_bbox[1], part2_bbox[2],
              part2_bbox[3], part2_bbox[4], part2_bbox[5], 1)
    ]

    validator = GuillotineValidatorRecursive(stock_dims)
    result = validator.validate(boxes)
    return result.is_valid


# =============================================================================
# PACKING ALGORITHMS
# =============================================================================

class BottomLeftPacker:
    """
    Bottom-Left (BL) Heuristic Packer.

    Places each part at the lowest-leftmost valid position.
    Simple but effective for regular grids.
    """

    def __init__(self, stock_dims: Tuple[float, float, float],
                 part_geom: TrapezoidGeometry,
                 orientation: Orientation,
                 explorer: OrientationExplorer,
                 saw_kerf: float = DEFAULT_SAW_KERF):
        self.stock_dims = stock_dims
        self.part_geom = part_geom
        self.orientation = orientation
        self.explorer = explorer
        self.saw_kerf = saw_kerf
        self.part_dims = orientation.dims

    def pack(self) -> List[PlacedPart]:
        """Pack parts using bottom-left heuristic."""
        sx, sy, sz = self.stock_dims
        dx, dy, dz = self.part_dims

        if dx > sx or dy > sy or dz > sz:
            return []

        placed = []
        placed_boxes = []
        part_id = 0

        # Grid step with saw kerf
        step_x = dx + self.saw_kerf
        step_y = dy + self.saw_kerf
        step_z = dz + self.saw_kerf

        # Try positions in bottom-left order (Z first, then Y, then X)
        z = 0
        while z + dz <= sz:
            y = 0
            while y + dy <= sy:
                x = 0
                while x + dx <= sx:
                    position = (x, y, z)

                    if not check_collision(position, self.part_dims, placed_boxes, self.stock_dims):
                        # Place part
                        part_cq = self.explorer.create_oriented_geometry(self.orientation)
                        part_cq = part_cq.translate(position)

                        placed.append(PlacedPart(
                            part_id=part_id,
                            position=position,
                            mirrored=False,
                            rotation=f"ori_{self.orientation.index}",
                            geometry=part_cq
                        ))

                        placed_boxes.append(get_placed_bbox(position, self.part_dims))
                        part_id += 1

                    x += step_x
                y += step_y
            z += step_z

        return placed


class BestFitPacker:
    """
    Best-Fit (BF) Heuristic Packer.

    Evaluates multiple candidate positions and chooses the one with best score.
    Score considers: contact with existing parts, distance from origin, waste.
    """

    def __init__(self, stock_dims: Tuple[float, float, float],
                 part_geom: TrapezoidGeometry,
                 orientation: Orientation,
                 explorer: OrientationExplorer,
                 saw_kerf: float = DEFAULT_SAW_KERF):
        self.stock_dims = stock_dims
        self.part_geom = part_geom
        self.orientation = orientation
        self.explorer = explorer
        self.saw_kerf = saw_kerf
        self.part_dims = orientation.dims

    def _score_position(self, position: Tuple[float, float, float],
                        placed_boxes: List[Tuple]) -> float:
        """Score a position - higher is better."""
        x, y, z = position
        score = 0.0

        # Prefer lower Z positions (gravity/stability)
        score -= z * 0.1

        # Prefer positions closer to origin
        distance = np.sqrt(x**2 + y**2 + z**2)
        score -= distance * 0.01

        # Prefer positions that touch existing parts (reduce gaps)
        new_box = get_placed_bbox(position, self.part_dims)
        for box in placed_boxes:
            if self._boxes_touch(new_box, box):
                score += 50

        # Prefer corner/edge positions
        if x < self.saw_kerf * 2:
            score += 10
        if y < self.saw_kerf * 2:
            score += 10
        if z < self.saw_kerf * 2:
            score += 10

        return score

    def _boxes_touch(self, box1: Tuple, box2: Tuple, tol: float = 0.5) -> bool:
        """Check if two boxes touch (share a face within tolerance)."""
        x1_min, y1_min, z1_min, x1_max, y1_max, z1_max = box1
        x2_min, y2_min, z2_min, x2_max, y2_max, z2_max = box2

        # Check X faces
        if abs(x1_max - x2_min) < tol or abs(x2_max - x1_min) < tol:
            if (y1_min < y2_max and y1_max > y2_min and
                z1_min < z2_max and z1_max > z2_min):
                return True

        # Check Y faces
        if abs(y1_max - y2_min) < tol or abs(y2_max - y1_min) < tol:
            if (x1_min < x2_max and x1_max > x2_min and
                z1_min < z2_max and z1_max > z2_min):
                return True

        # Check Z faces
        if abs(z1_max - z2_min) < tol or abs(z2_max - z1_min) < tol:
            if (x1_min < x2_max and x1_max > x2_min and
                y1_min < y2_max and y1_max > y2_min):
                return True

        return False

    def _get_candidate_positions(self, placed_boxes: List[Tuple]) -> List[Tuple[float, float, float]]:
        """Generate candidate positions for next part."""
        sx, sy, sz = self.stock_dims
        dx, dy, dz = self.part_dims
        step = self.saw_kerf

        candidates = []

        # Always include origin
        candidates.append((0, 0, 0))

        # Grid positions
        num_x = int((sx - dx) / (dx + step)) + 1
        num_y = int((sy - dy) / (dy + step)) + 1
        num_z = int((sz - dz) / (dz + step)) + 1

        for iz in range(num_z):
            for iy in range(num_y):
                for ix in range(num_x):
                    x = ix * (dx + step)
                    y = iy * (dy + step)
                    z = iz * (dz + step)
                    if x + dx <= sx and y + dy <= sy and z + dz <= sz:
                        candidates.append((x, y, z))

        # Positions adjacent to existing boxes
        for box in placed_boxes:
            x_min, y_min, z_min, x_max, y_max, z_max = box

            # Right of box
            if x_max + step + dx <= sx:
                candidates.append((x_max + step, y_min, z_min))

            # Behind box
            if y_max + step + dy <= sy:
                candidates.append((x_min, y_max + step, z_min))

            # Above box
            if z_max + step + dz <= sz:
                candidates.append((x_min, y_min, z_max + step))

        return list(set(candidates))

    def pack(self) -> List[PlacedPart]:
        """Pack parts using best-fit heuristic."""
        sx, sy, sz = self.stock_dims
        dx, dy, dz = self.part_dims

        if dx > sx or dy > sy or dz > sz:
            return []

        placed = []
        placed_boxes = []
        part_id = 0

        # Maximum parts that could fit
        max_parts = int((sx / dx) * (sy / dy) * (sz / dz)) + 10

        for _ in range(max_parts):
            candidates = self._get_candidate_positions(placed_boxes)

            # Score and filter valid candidates
            valid_candidates = []
            for pos in candidates:
                if not check_collision(pos, self.part_dims, placed_boxes, self.stock_dims):
                    score = self._score_position(pos, placed_boxes)
                    valid_candidates.append((pos, score))

            if not valid_candidates:
                break

            # Choose best position
            best_pos, _ = max(valid_candidates, key=lambda x: x[1])

            # Place part
            part_cq = self.explorer.create_oriented_geometry(self.orientation)
            part_cq = part_cq.translate(best_pos)

            placed.append(PlacedPart(
                part_id=part_id,
                position=best_pos,
                mirrored=False,
                rotation=f"ori_{self.orientation.index}",
                geometry=part_cq
            ))

            placed_boxes.append(get_placed_bbox(best_pos, self.part_dims))
            part_id += 1

        return placed


class SkylinePacker:
    """
    Skyline Heuristic Packer.

    Maintains a 2D skyline (height map) and places parts to fill gaps.
    Works layer by layer in the Y direction.
    """

    def __init__(self, stock_dims: Tuple[float, float, float],
                 part_geom: TrapezoidGeometry,
                 orientation: Orientation,
                 explorer: OrientationExplorer,
                 saw_kerf: float = DEFAULT_SAW_KERF):
        self.stock_dims = stock_dims
        self.part_geom = part_geom
        self.orientation = orientation
        self.explorer = explorer
        self.saw_kerf = saw_kerf
        self.part_dims = orientation.dims

    def pack(self) -> List[PlacedPart]:
        """Pack parts using skyline heuristic."""
        sx, sy, sz = self.stock_dims
        dx, dy, dz = self.part_dims
        step = self.saw_kerf

        if dx > sx or dy > sy or dz > sz:
            return []

        placed = []
        placed_boxes = []
        part_id = 0

        # Initialize skyline: array of heights at each X position
        # We discretize X into bins
        resolution = max(1.0, dx / 10)
        num_bins = int(sx / resolution) + 1
        skyline = np.zeros(num_bins)

        # Process Y layers
        y = 0
        while y + dy <= sy:
            # Reset skyline for each Y layer
            skyline = np.zeros(num_bins)

            # Fill this Y layer
            placed_in_layer = True
            while placed_in_layer:
                placed_in_layer = False

                # Find lowest point in skyline where part fits
                best_x = None
                best_z = float('inf')

                for i in range(num_bins):
                    x = i * resolution
                    if x + dx > sx:
                        continue

                    # Get max height in the range [x, x+dx]
                    i_end = min(int((x + dx) / resolution) + 1, num_bins)
                    z = skyline[i:i_end].max()

                    if z + dz <= sz and z < best_z:
                        # Check collision with existing parts
                        pos = (x, y, z + step if z > 0 else z)
                        if not check_collision(pos, self.part_dims, placed_boxes, self.stock_dims):
                            best_z = z
                            best_x = x

                if best_x is not None:
                    # Place part
                    z = best_z + step if best_z > 0 else best_z
                    position = (best_x, y, z)

                    part_cq = self.explorer.create_oriented_geometry(self.orientation)
                    part_cq = part_cq.translate(position)

                    placed.append(PlacedPart(
                        part_id=part_id,
                        position=position,
                        mirrored=False,
                        rotation=f"ori_{self.orientation.index}",
                        geometry=part_cq
                    ))

                    placed_boxes.append(get_placed_bbox(position, self.part_dims))
                    part_id += 1

                    # Update skyline
                    i_start = int(best_x / resolution)
                    i_end = min(int((best_x + dx) / resolution) + 1, num_bins)
                    skyline[i_start:i_end] = z + dz

                    placed_in_layer = True

            y += dy + step

        return placed


class LayerBasedPacker:
    """
    Layer-Based Packer.

    Packs parts in regular horizontal layers (XY planes at different Z levels).
    Most efficient for identical parts.
    """

    def __init__(self, stock_dims: Tuple[float, float, float],
                 part_geom: TrapezoidGeometry,
                 orientation: Orientation,
                 explorer: OrientationExplorer,
                 saw_kerf: float = DEFAULT_SAW_KERF):
        self.stock_dims = stock_dims
        self.part_geom = part_geom
        self.orientation = orientation
        self.explorer = explorer
        self.saw_kerf = saw_kerf
        self.part_dims = orientation.dims

    def pack(self) -> List[PlacedPart]:
        """Pack parts in regular grid layers."""
        sx, sy, sz = self.stock_dims
        dx, dy, dz = self.part_dims
        step = self.saw_kerf

        if dx > sx or dy > sy or dz > sz:
            return []

        placed = []
        part_id = 0

        # Calculate how many fit in each direction
        nx = int(sx / (dx + step))
        ny = int(sy / (dy + step))
        nz = int(sz / (dz + step))

        # Adjust for last row/column that might fit without kerf
        if nx * (dx + step) + dx <= sx + step:
            nx += 1
        if ny * (dy + step) + dy <= sy + step:
            ny += 1
        if nz * (dz + step) + dz <= sz + step:
            nz += 1

        # Place parts layer by layer
        for iz in range(nz):
            z = iz * (dz + step)
            if z + dz > sz:
                continue

            for iy in range(ny):
                y = iy * (dy + step)
                if y + dy > sy:
                    continue

                for ix in range(nx):
                    x = ix * (dx + step)
                    if x + dx > sx:
                        continue

                    position = (x, y, z)

                    part_cq = self.explorer.create_oriented_geometry(self.orientation)
                    part_cq = part_cq.translate(position)

                    placed.append(PlacedPart(
                        part_id=part_id,
                        position=position,
                        mirrored=False,
                        rotation=f"ori_{self.orientation.index}",
                        geometry=part_cq
                    ))
                    part_id += 1

        return placed


class MirroredPairsPacker:
    """
    Mirrored Pairs Packer.

    Places parts in mirrored pairs where possible to reduce waste.
    Validates extractability of pairs.
    """

    def __init__(self, stock_dims: Tuple[float, float, float],
                 part_geom: TrapezoidGeometry,
                 orientation: Orientation,
                 explorer: OrientationExplorer,
                 saw_kerf: float = DEFAULT_SAW_KERF):
        self.stock_dims = stock_dims
        self.part_geom = part_geom
        self.orientation = orientation
        self.explorer = explorer
        self.saw_kerf = saw_kerf
        self.part_dims = orientation.dims

    def _create_mirrored_part(self) -> cq.Workplane:
        """Create a mirrored version of the part at current orientation."""
        # Get normal part
        part = self.explorer.create_oriented_geometry(self.orientation)

        # Flip around X axis (through center)
        bbox = part.val().BoundingBox()
        cx = (bbox.xmin + bbox.xmax) / 2
        cy = (bbox.ymin + bbox.ymax) / 2
        cz = (bbox.zmin + bbox.zmax) / 2

        part = part.rotate((cx, cy, cz), (cx + 1, cy, cz), 180)

        # Translate back to origin
        bbox = part.val().BoundingBox()
        part = part.translate((-bbox.xmin, -bbox.ymin, -bbox.zmin))

        return part

    def pack(self) -> List[PlacedPart]:
        """Pack parts in mirrored pairs."""
        sx, sy, sz = self.stock_dims
        dx, dy, dz = self.part_dims
        step = self.saw_kerf

        if dx > sx or dy > sy or dz > sz:
            return []

        placed = []
        part_id = 0

        # Calculate grid for pairs (two parts per Z position)
        nx = int(sx / (dx + step))
        ny = int(sy / (dy + step))
        nz = int(sz / (2 * dz + step))  # Pairs take 2*dz height

        # Remaining single layer
        remaining_z = sz - nz * (2 * dz + step)
        extra_z_layer = remaining_z >= dz

        # Place pairs
        for iz in range(nz):
            z_base = iz * (2 * dz + step)

            for iy in range(ny):
                y = iy * (dy + step)
                if y + dy > sy:
                    continue

                for ix in range(nx):
                    x = ix * (dx + step)
                    if x + dx > sx:
                        continue

                    # Place normal part at bottom
                    position1 = (x, y, z_base)
                    part1 = self.explorer.create_oriented_geometry(self.orientation)
                    part1 = part1.translate(position1)

                    placed.append(PlacedPart(
                        part_id=part_id,
                        position=position1,
                        mirrored=False,
                        rotation=f"ori_{self.orientation.index}",
                        geometry=part1
                    ))
                    part_id += 1

                    # Place mirrored part on top
                    position2 = (x, y, z_base + dz + step)
                    part2 = self._create_mirrored_part()
                    part2 = part2.translate(position2)

                    placed.append(PlacedPart(
                        part_id=part_id,
                        position=position2,
                        mirrored=True,
                        rotation=f"ori_{self.orientation.index}_flip",
                        geometry=part2
                    ))
                    part_id += 1

        # Place extra single layer if space
        if extra_z_layer:
            z = nz * (2 * dz + step)
            for iy in range(ny):
                y = iy * (dy + step)
                if y + dy > sy:
                    continue

                for ix in range(nx):
                    x = ix * (dx + step)
                    if x + dx > sx:
                        continue

                    if z + dz <= sz:
                        position = (x, y, z)
                        part = self.explorer.create_oriented_geometry(self.orientation)
                        part = part.translate(position)

                        placed.append(PlacedPart(
                            part_id=part_id,
                            position=position,
                            mirrored=False,
                            rotation=f"ori_{self.orientation.index}",
                            geometry=part
                        ))
                        part_id += 1

        return placed


# =============================================================================
# VALIDATION
# =============================================================================

def validate_packing_extractable(placed_parts: List[PlacedPart],
                                  stock_dims: Tuple[float, float, float]) -> bool:
    """Check if packing is extractable using guillotine cuts."""
    if len(placed_parts) <= 1:
        return True

    boxes = []
    for part in placed_parts:
        bbox = part.geometry.val().BoundingBox()
        boxes.append(Box3D(
            x_min=bbox.xmin, y_min=bbox.ymin, z_min=bbox.zmin,
            x_max=bbox.xmax, y_max=bbox.ymax, z_max=bbox.zmax,
            part_id=part.part_id
        ))

    validator = GuillotineValidatorRecursive(stock_dims)
    result = validator.validate(boxes)
    return result.is_valid


# =============================================================================
# MAIN PACKER CLASS
# =============================================================================

class EnhancedGreedyPacker:
    """
    Enhanced Greedy Packer with multiple algorithms and orientations.

    Generates multiple packing candidates and ranks them by waste percentage.
    """

    def __init__(self, stock_dims: Tuple[float, float, float],
                 part_geom: TrapezoidGeometry,
                 saw_kerf: float = DEFAULT_SAW_KERF,
                 top_n_orientations: int = 20):
        """
        Initialize the enhanced greedy packer.

        Args:
            stock_dims: Stock block dimensions (length, width, height)
            part_geom: Part geometry specification
            saw_kerf: Spacing between parts for saw blade
            top_n_orientations: Number of top orientations to test
        """
        self.stock_dims = stock_dims
        self.part_geom = part_geom
        self.saw_kerf = saw_kerf
        self.top_n_orientations = top_n_orientations

        # Create orientation explorer
        self.explorer = OrientationExplorer(part_geom)

        # Get top orientations by potential parts
        self.orientations = self._get_top_orientations()

    def _get_top_orientations(self) -> List[Orientation]:
        """Get top N orientations by potential part count."""
        ranked = self.explorer.rank_orientations(self.stock_dims, self.saw_kerf)

        # Take top N
        top = [ori for ori, count in ranked[:self.top_n_orientations]]
        return top

    def generate_all_packings(self,
                               algorithms: List[PackingAlgorithm] = None) -> List[PackingCandidate]:
        """
        Generate packings using all algorithms and top orientations.

        Args:
            algorithms: List of algorithms to use (default: all)

        Returns:
            List of PackingCandidate sorted by waste percentage
        """
        if algorithms is None:
            algorithms = list(PackingAlgorithm)

        candidates = []

        for orientation in self.orientations:
            for algo in algorithms:
                try:
                    placed = self._run_algorithm(algo, orientation)

                    if not placed:
                        continue

                    # Validate extractability
                    is_extractable = validate_packing_extractable(placed, self.stock_dims)

                    # Calculate metrics
                    num_parts = len(placed)
                    parts_volume = num_parts * self.part_geom.volume
                    stock_volume = self.stock_dims[0] * self.stock_dims[1] * self.stock_dims[2]
                    waste_percentage = ((stock_volume - parts_volume) / stock_volume) * 100

                    # Count mirrored pairs
                    mirrored_count = sum(1 for p in placed if p.mirrored)

                    candidate = PackingCandidate(
                        algorithm=algo.value,
                        orientation=orientation,
                        placed_parts=placed,
                        num_parts=num_parts,
                        waste_percentage=waste_percentage,
                        is_extractable=is_extractable,
                        stock_dims=self.stock_dims,
                        part_volume=self.part_geom.volume,
                        mirrored_pairs=mirrored_count // 2,
                        metadata={"saw_kerf": self.saw_kerf}
                    )
                    candidates.append(candidate)

                except Exception as e:
                    # Log but continue with other algorithms
                    print(f"  Warning: {algo.value} failed for orientation {orientation.index}: {e}")
                    continue

        # Sort by waste percentage (lowest first), with extractable preferred
        candidates.sort(key=lambda c: (not c.is_extractable, c.waste_percentage))

        return candidates

    def _run_algorithm(self, algo: PackingAlgorithm, orientation: Orientation) -> List[PlacedPart]:
        """Run a specific packing algorithm."""
        if algo == PackingAlgorithm.BOTTOM_LEFT:
            packer = BottomLeftPacker(
                self.stock_dims, self.part_geom, orientation,
                self.explorer, self.saw_kerf
            )
        elif algo == PackingAlgorithm.BEST_FIT:
            packer = BestFitPacker(
                self.stock_dims, self.part_geom, orientation,
                self.explorer, self.saw_kerf
            )
        elif algo == PackingAlgorithm.SKYLINE:
            packer = SkylinePacker(
                self.stock_dims, self.part_geom, orientation,
                self.explorer, self.saw_kerf
            )
        elif algo == PackingAlgorithm.LAYER_BASED:
            packer = LayerBasedPacker(
                self.stock_dims, self.part_geom, orientation,
                self.explorer, self.saw_kerf
            )
        elif algo == PackingAlgorithm.MIRRORED_PAIRS:
            packer = MirroredPairsPacker(
                self.stock_dims, self.part_geom, orientation,
                self.explorer, self.saw_kerf
            )
        else:
            raise ValueError(f"Unknown algorithm: {algo}")

        return packer.pack()

    def get_best_extractable(self, n: int = 5) -> List[PackingCandidate]:
        """Get the top N extractable packings."""
        all_packings = self.generate_all_packings()
        extractable = [p for p in all_packings if p.is_extractable]
        return extractable[:n]

    def get_comparison_table(self, candidates: List[PackingCandidate]) -> str:
        """Generate comparison table for candidates."""
        lines = [
            "=" * 100,
            "PACKING COMPARISON",
            "=" * 100,
            f"{'Algorithm':<20} {'Orientation':<15} {'Parts':<8} {'Waste%':<10} {'Mirrored':<10} {'Extractable':<12}",
            "-" * 100,
        ]

        for c in candidates:
            ori_str = f"Ori {c.orientation.index}"
            extract_str = "YES" if c.is_extractable else "NO"
            lines.append(
                f"{c.algorithm:<20} {ori_str:<15} {c.num_parts:<8} {c.waste_percentage:<10.2f} "
                f"{c.mirrored_pairs:<10} {extract_str:<12}"
            )

        lines.append("=" * 100)
        return "\n".join(lines)
