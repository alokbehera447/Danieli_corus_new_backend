"""
Guillotine cut validation algorithms.

Two independent implementations to verify packing can be extracted using guillotine cuts:
1. GuillotineValidatorRecursive - Recursive decomposition approach
2. GuillotineValidatorGraph - Graph-based separation approach

A guillotine cut is a planar cut that goes completely through the stock block,
dividing it into two pieces. All parts must be extractable using only such cuts.
"""

import numpy as np
from typing import List, Tuple, Optional, Set
from dataclasses import dataclass
from itertools import permutations


@dataclass
class Box3D:
    """Axis-aligned 3D bounding box."""
    x_min: float
    y_min: float
    z_min: float
    x_max: float
    y_max: float
    z_max: float
    part_id: int = -1

    @property
    def center(self) -> Tuple[float, float, float]:
        return (
            (self.x_min + self.x_max) / 2,
            (self.y_min + self.y_max) / 2,
            (self.z_min + self.z_max) / 2
        )

    def overlaps(self, other: 'Box3D', eps: float = 1e-6) -> bool:
        return not (
            self.x_max <= other.x_min + eps or other.x_max <= self.x_min + eps or
            self.y_max <= other.y_min + eps or other.y_max <= self.y_min + eps or
            self.z_max <= other.z_min + eps or other.z_max <= self.z_min + eps
        )

    def contains_point(self, x: float, y: float, z: float, eps: float = 1e-6) -> bool:
        return (
            self.x_min - eps <= x <= self.x_max + eps and
            self.y_min - eps <= y <= self.y_max + eps and
            self.z_min - eps <= z <= self.z_max + eps
        )


@dataclass
class GuillotineCut:
    """A guillotine cut specification."""
    axis: str
    position: float
    boxes_below: List[int]
    boxes_above: List[int]


@dataclass
class ValidationResult:
    """Result of guillotine validation."""
    is_valid: bool
    cut_sequence: List[GuillotineCut]
    message: str


class GuillotineValidatorRecursive:
    """
    Validates guillotine extractability using recursive decomposition.

    Algorithm:
    1. Start with all boxes in the stock region
    2. Try to find a guillotine cut that separates boxes into two groups
    3. Recursively validate each group
    4. Base case: single box or empty region is always valid
    """

    def __init__(self, stock_dims: Tuple[float, float, float], eps: float = 1e-6):
        self.stock_length, self.stock_width, self.stock_height = stock_dims
        self.eps = eps
        self.cut_sequence = []

    def validate(self, boxes: List[Box3D]) -> ValidationResult:
        self.cut_sequence = []

        if len(boxes) == 0:
            return ValidationResult(True, [], "Empty packing is trivially valid")

        if len(boxes) == 1:
            return ValidationResult(True, [], "Single part is trivially extractable")

        for i, b1 in enumerate(boxes):
            for j, b2 in enumerate(boxes):
                if i < j and b1.overlaps(b2, self.eps):
                    return ValidationResult(
                        False, [],
                        f"Parts {b1.part_id} and {b2.part_id} overlap"
                    )

        stock_box = Box3D(0, 0, 0, self.stock_length, self.stock_width, self.stock_height)

        success = self._recursive_validate(boxes, stock_box)

        if success:
            return ValidationResult(
                True, self.cut_sequence,
                f"Valid guillotine packing with {len(self.cut_sequence)} cuts"
            )
        else:
            return ValidationResult(
                False, [],
                "Cannot find valid guillotine cut sequence"
            )

    def _recursive_validate(self, boxes: List[Box3D], region: Box3D) -> bool:
        if len(boxes) <= 1:
            return True

        for axis in ['x', 'y', 'z']:
            cut_positions = self._find_cut_positions(boxes, axis)

            for pos in cut_positions:
                below, above = self._split_boxes(boxes, axis, pos)

                if len(below) > 0 and len(above) > 0:
                    if len(below) < len(boxes) and len(above) < len(boxes):
                        cut = GuillotineCut(
                            axis=axis,
                            position=pos,
                            boxes_below=[b.part_id for b in below],
                            boxes_above=[b.part_id for b in above]
                        )

                        region_below, region_above = self._split_region(region, axis, pos)

                        if (self._recursive_validate(below, region_below) and
                            self._recursive_validate(above, region_above)):
                            self.cut_sequence.append(cut)
                            return True

        return False

    def _find_cut_positions(self, boxes: List[Box3D], axis: str) -> List[float]:
        positions = set()

        for box in boxes:
            if axis == 'x':
                positions.add(box.x_min)
                positions.add(box.x_max)
            elif axis == 'y':
                positions.add(box.y_min)
                positions.add(box.y_max)
            else:
                positions.add(box.z_min)
                positions.add(box.z_max)

        sorted_pos = sorted(positions)
        cut_positions = []

        for i in range(len(sorted_pos) - 1):
            mid = (sorted_pos[i] + sorted_pos[i + 1]) / 2
            cut_positions.append(sorted_pos[i])
            cut_positions.append(mid)
        if sorted_pos:
            cut_positions.append(sorted_pos[-1])

        valid_positions = []
        for pos in cut_positions:
            is_valid = True
            for box in boxes:
                if axis == 'x':
                    if box.x_min + self.eps < pos < box.x_max - self.eps:
                        is_valid = False
                        break
                elif axis == 'y':
                    if box.y_min + self.eps < pos < box.y_max - self.eps:
                        is_valid = False
                        break
                else:
                    if box.z_min + self.eps < pos < box.z_max - self.eps:
                        is_valid = False
                        break
            if is_valid:
                valid_positions.append(pos)

        return sorted(set(valid_positions))

    def _split_boxes(self, boxes: List[Box3D], axis: str, pos: float) -> Tuple[List[Box3D], List[Box3D]]:
        below = []
        above = []

        for box in boxes:
            if axis == 'x':
                box_max = box.x_max
            elif axis == 'y':
                box_max = box.y_max
            else:
                box_max = box.z_max

            if box_max <= pos + self.eps:
                below.append(box)
            else:
                above.append(box)

        return below, above

    def _split_region(self, region: Box3D, axis: str, pos: float) -> Tuple[Box3D, Box3D]:
        if axis == 'x':
            below = Box3D(region.x_min, region.y_min, region.z_min, pos, region.y_max, region.z_max)
            above = Box3D(pos, region.y_min, region.z_min, region.x_max, region.y_max, region.z_max)
        elif axis == 'y':
            below = Box3D(region.x_min, region.y_min, region.z_min, region.x_max, pos, region.z_max)
            above = Box3D(region.x_min, pos, region.z_min, region.x_max, region.y_max, region.z_max)
        else:
            below = Box3D(region.x_min, region.y_min, region.z_min, region.x_max, region.y_max, pos)
            above = Box3D(region.x_min, region.y_min, pos, region.x_max, region.y_max, region.z_max)

        return below, above


class GuillotineValidatorGraph:
    """
    Validates guillotine extractability using graph-based approach.

    Algorithm:
    1. Build a separation graph where edges represent possible cuts between boxes
    2. A valid guillotine packing exists if we can find a sequence of cuts
       that separates all boxes using only through-cuts
    3. Uses BFS/DFS to find valid cut orderings
    """

    def __init__(self, stock_dims: Tuple[float, float, float], eps: float = 1e-6):
        self.stock_length, self.stock_width, self.stock_height = stock_dims
        self.eps = eps

    def validate(self, boxes: List[Box3D]) -> ValidationResult:
        if len(boxes) == 0:
            return ValidationResult(True, [], "Empty packing is trivially valid")

        if len(boxes) == 1:
            return ValidationResult(True, [], "Single part is trivially extractable")

        for i, b1 in enumerate(boxes):
            for j, b2 in enumerate(boxes):
                if i < j and b1.overlaps(b2, self.eps):
                    return ValidationResult(
                        False, [],
                        f"Parts {b1.part_id} and {b2.part_id} overlap"
                    )

        box_indices = set(range(len(boxes)))
        cut_sequence = []

        success = self._find_cut_sequence(boxes, box_indices, cut_sequence)

        if success:
            return ValidationResult(
                True, cut_sequence,
                f"Valid guillotine packing with {len(cut_sequence)} cuts"
            )
        else:
            return ValidationResult(
                False, [],
                "Cannot find valid guillotine cut sequence"
            )

    def _find_cut_sequence(self, boxes: List[Box3D], remaining: Set[int],
                           cut_sequence: List[GuillotineCut]) -> bool:
        if len(remaining) <= 1:
            return True

        remaining_boxes = [boxes[i] for i in remaining]

        for axis in ['x', 'y', 'z']:
            separating_cuts = self._find_separating_cuts(remaining_boxes, axis)

            for pos in separating_cuts:
                group1_indices = set()
                group2_indices = set()

                for idx in remaining:
                    box = boxes[idx]
                    if axis == 'x':
                        box_max = box.x_max
                    elif axis == 'y':
                        box_max = box.y_max
                    else:
                        box_max = box.z_max

                    if box_max <= pos + self.eps:
                        group1_indices.add(idx)
                    else:
                        group2_indices.add(idx)

                if len(group1_indices) > 0 and len(group2_indices) > 0:
                    cut = GuillotineCut(
                        axis=axis,
                        position=pos,
                        boxes_below=[boxes[i].part_id for i in group1_indices],
                        boxes_above=[boxes[i].part_id for i in group2_indices]
                    )

                    if (self._find_cut_sequence(boxes, group1_indices, cut_sequence) and
                        self._find_cut_sequence(boxes, group2_indices, cut_sequence)):
                        cut_sequence.append(cut)
                        return True

        return False

    def _find_separating_cuts(self, boxes: List[Box3D], axis: str) -> List[float]:
        boundaries = set()

        for box in boxes:
            if axis == 'x':
                boundaries.add(box.x_min)
                boundaries.add(box.x_max)
            elif axis == 'y':
                boundaries.add(box.y_min)
                boundaries.add(box.y_max)
            else:
                boundaries.add(box.z_min)
                boundaries.add(box.z_max)

        valid_positions = []
        for pos in sorted(boundaries):
            cuts_through_box = False
            for box in boxes:
                if axis == 'x':
                    if box.x_min + self.eps < pos < box.x_max - self.eps:
                        cuts_through_box = True
                        break
                elif axis == 'y':
                    if box.y_min + self.eps < pos < box.y_max - self.eps:
                        cuts_through_box = True
                        break
                else:
                    if box.z_min + self.eps < pos < box.z_max - self.eps:
                        cuts_through_box = True
                        break

            if not cuts_through_box:
                valid_positions.append(pos)

        return valid_positions


def boxes_from_packing(placed_parts) -> List[Box3D]:
    """Convert placed parts to Box3D list for validation."""
    boxes = []
    for part in placed_parts:
        bbox = part.geometry.val().BoundingBox()
        boxes.append(Box3D(
            x_min=bbox.xmin,
            y_min=bbox.ymin,
            z_min=bbox.zmin,
            x_max=bbox.xmax,
            y_max=bbox.ymax,
            z_max=bbox.zmax,
            part_id=part.part_id
        ))
    return boxes


def validate_guillotine_packing(placed_parts, stock_dims: Tuple[float, float, float],
                                 method: str = "both") -> dict:
    """
    Validate if a packing can be extracted using guillotine cuts.

    Args:
        placed_parts: List of PlacedPart objects with geometry
        stock_dims: (length, width, height) of stock block
        method: "recursive", "graph", or "both"

    Returns:
        dict with validation results from requested methods
    """
    boxes = boxes_from_packing(placed_parts)
    results = {}

    if method in ["recursive", "both"]:
        validator_r = GuillotineValidatorRecursive(stock_dims)
        results["recursive"] = validator_r.validate(boxes)

    if method in ["graph", "both"]:
        validator_g = GuillotineValidatorGraph(stock_dims)
        results["graph"] = validator_g.validate(boxes)

    return results
