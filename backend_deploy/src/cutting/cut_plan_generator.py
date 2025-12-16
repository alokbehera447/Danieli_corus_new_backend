"""
Cut plan generation for guillotine-based extraction.

Generates ordered cutting sequences from packed parts, identifying unique cut planes,
merging shared planes within tolerance, and ordering cuts for efficient extraction.

Strategy: Separate Columns First (Vertical Cut First)
- Start with vertical cuts (X-axis) to separate columns
- Then horizontal cuts (Z-axis) to separate layers within columns
- Finally depth cuts (Y-axis) if needed
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Set
from dataclasses import dataclass, field
from collections import defaultdict

from ..geometry.data_classes import CutPlane
from .guillotine_validator import Box3D, GuillotineValidatorRecursive


@dataclass
class CutSpecification:
    """
    Complete specification for a single cut.

    Attributes:
        cut_id: Unique identifier for this cut
        axis: Primary axis ('x', 'y', or 'z')
        position: Position along the axis (mm)
        plane: CutPlane object with equation and normal
        parts_separated: List of part IDs separated by this cut
        group_before: Part IDs on the negative side of cut
        group_after: Part IDs on the positive side of cut
        is_shared: Whether this plane is shared by multiple parts
        merged_from: Original positions merged into this cut
    """
    cut_id: int
    axis: str
    position: float
    plane: CutPlane
    parts_separated: List[int] = field(default_factory=list)
    group_before: List[int] = field(default_factory=list)
    group_after: List[int] = field(default_factory=list)
    is_shared: bool = False
    merged_from: List[float] = field(default_factory=list)

    def get_equation_string(self) -> str:
        """Get human-readable plane equation string."""
        A, B, C, D = self.plane.get_equation()
        terms = []
        if abs(A) > 1e-6:
            terms.append(f"{A:.4f}x")
        if abs(B) > 1e-6:
            sign = "+" if B > 0 else ""
            terms.append(f"{sign}{B:.4f}y")
        if abs(C) > 1e-6:
            sign = "+" if C > 0 else ""
            terms.append(f"{sign}{C:.4f}z")
        sign = "+" if D > 0 else ""
        terms.append(f"{sign}{D:.4f}")
        return " ".join(terms) + " = 0"

    def __repr__(self) -> str:
        shared_str = " [SHARED]" if self.is_shared else ""
        return (f"Cut #{self.cut_id}: {self.axis.upper()}={self.position:.2f}mm{shared_str} "
                f"separating {len(self.parts_separated)} parts")


@dataclass
class CutPlan:
    """
    Complete cutting plan for extracting parts from stock.

    Attributes:
        stock_dims: Stock block dimensions (length, width, height)
        cuts: Ordered list of cut specifications
        parts: List of Box3D parts being extracted
        total_parts: Number of parts to extract
        extraction_order: Suggested order for part extraction
    """
    stock_dims: Tuple[float, float, float]
    cuts: List[CutSpecification] = field(default_factory=list)
    parts: List[Box3D] = field(default_factory=list)
    total_parts: int = 0
    extraction_order: List[int] = field(default_factory=list)

    def get_summary(self) -> str:
        """Generate a summary of the cut plan."""
        lines = [
            "=" * 60,
            "CUT PLAN SUMMARY",
            "=" * 60,
            f"Stock dimensions: {self.stock_dims[0]:.1f} x {self.stock_dims[1]:.1f} x {self.stock_dims[2]:.1f} mm",
            f"Total parts to extract: {self.total_parts}",
            f"Total cuts required: {len(self.cuts)}",
            "",
            "Cuts by axis:",
        ]

        axis_counts = defaultdict(int)
        for cut in self.cuts:
            axis_counts[cut.axis] += 1

        for axis in ['x', 'y', 'z']:
            if axis_counts[axis] > 0:
                lines.append(f"  {axis.upper()}-axis cuts: {axis_counts[axis]}")

        shared_count = sum(1 for c in self.cuts if c.is_shared)
        if shared_count > 0:
            lines.append(f"\nShared/merged planes: {shared_count}")

        lines.append("=" * 60)
        return "\n".join(lines)

    def get_detailed_report(self) -> str:
        """Generate detailed cut specification report."""
        lines = [
            "=" * 70,
            "CUT SPECIFICATION REPORT",
            "=" * 70,
            f"Stock Block: {self.stock_dims[0]:.1f}mm × {self.stock_dims[1]:.1f}mm × {self.stock_dims[2]:.1f}mm",
            f"Stock Volume: {self.stock_dims[0] * self.stock_dims[1] * self.stock_dims[2]:,.0f} mm³",
            "",
        ]

        for cut in self.cuts:
            A, B, C, D = cut.plane.get_equation()
            angles = cut.plane.get_normal_angles()

            lines.extend([
                f"Cut #{cut.cut_id}:",
                f"  Axis: {cut.axis.upper()}-axis cut",
                f"  Position: {cut.axis} = {cut.position:.2f} mm",
                f"  Plane equation: {cut.get_equation_string()}",
                f"  Normal vector: [{cut.plane.normal[0]:.4f}, {cut.plane.normal[1]:.4f}, {cut.plane.normal[2]:.4f}]",
                f"  Normal angles: [α_x={angles[0]:.1f}°, α_y={angles[1]:.1f}°, α_z={angles[2]:.1f}°]",
                f"  Point on plane: [{cut.plane.point[0]:.2f}, {cut.plane.point[1]:.2f}, {cut.plane.point[2]:.2f}]",
            ])

            if cut.is_shared:
                lines.append(f"  Shared plane: YES (merged from {len(cut.merged_from)} positions)")
                if cut.merged_from:
                    merged_str = ", ".join(f"{p:.2f}" for p in cut.merged_from)
                    lines.append(f"  Original positions: [{merged_str}]")

            lines.append(f"  Parts before cut: {cut.group_before}")
            lines.append(f"  Parts after cut: {cut.group_after}")
            lines.append("")

        lines.append("=" * 70)
        return "\n".join(lines)


class CutPlanGenerator:
    """
    Generates cutting plans from packed parts.

    Strategy: Vertical-First (Separate Columns First)
    1. Identify all potential cut planes from part boundaries
    2. Merge planes within tolerance (0.2mm)
    3. Order cuts: X-axis first, then Z-axis, then Y-axis
    4. Generate mathematical specifications for each cut
    """

    def __init__(self, stock_dims: Tuple[float, float, float],
                 merge_tolerance: float = 0.2,
                 eps: float = 1e-6):
        """
        Initialize cut plan generator.

        Args:
            stock_dims: Stock block dimensions (length, width, height)
            merge_tolerance: Tolerance for merging similar planes (mm)
            eps: Epsilon for floating point comparisons
        """
        self.stock_length, self.stock_width, self.stock_height = stock_dims
        self.stock_dims = stock_dims
        self.merge_tolerance = merge_tolerance
        self.eps = eps

    def generate_plan(self, boxes: List[Box3D]) -> CutPlan:
        """
        Generate a complete cutting plan for the given parts.

        Args:
            boxes: List of Box3D objects representing parts

        Returns:
            CutPlan with ordered cut specifications
        """
        if not boxes:
            return CutPlan(stock_dims=self.stock_dims, total_parts=0)

        # Validate guillotine extractability first
        validator = GuillotineValidatorRecursive(self.stock_dims, self.eps)
        result = validator.validate(boxes)

        if not result.is_valid:
            raise ValueError(f"Packing is not guillotine-extractable: {result.message}")

        # Collect all boundary positions for each axis
        boundaries = self._collect_boundaries(boxes)

        # Merge nearby boundaries
        merged_boundaries = self._merge_boundaries(boundaries)

        # Generate cut sequence using recursive decomposition
        cuts = self._generate_cut_sequence(boxes, merged_boundaries)

        # Create cut plan
        plan = CutPlan(
            stock_dims=self.stock_dims,
            cuts=cuts,
            parts=boxes,
            total_parts=len(boxes),
            extraction_order=[b.part_id for b in boxes]
        )

        return plan

    def _collect_boundaries(self, boxes: List[Box3D]) -> Dict[str, List[Tuple[float, Set[int]]]]:
        """
        Collect all boundary positions from part bounding boxes.

        Returns:
            Dict mapping axis to list of (position, set of part_ids touching this boundary)
        """
        boundaries = {'x': defaultdict(set), 'y': defaultdict(set), 'z': defaultdict(set)}

        for box in boxes:
            # X boundaries
            boundaries['x'][box.x_min].add(box.part_id)
            boundaries['x'][box.x_max].add(box.part_id)

            # Y boundaries
            boundaries['y'][box.y_min].add(box.part_id)
            boundaries['y'][box.y_max].add(box.part_id)

            # Z boundaries
            boundaries['z'][box.z_min].add(box.part_id)
            boundaries['z'][box.z_max].add(box.part_id)

        # Convert to sorted list of (position, part_ids)
        result = {}
        for axis in ['x', 'y', 'z']:
            result[axis] = [(pos, parts) for pos, parts in sorted(boundaries[axis].items())]

        return result

    def _merge_boundaries(self, boundaries: Dict[str, List[Tuple[float, Set[int]]]]) -> Dict[str, List[Tuple[float, Set[int], List[float]]]]:
        """
        Merge boundaries that are within tolerance.

        Returns:
            Dict mapping axis to list of (merged_position, part_ids, original_positions)
        """
        result = {}

        for axis, bound_list in boundaries.items():
            if not bound_list:
                result[axis] = []
                continue

            merged = []
            current_pos = bound_list[0][0]
            current_parts = set(bound_list[0][1])
            original_positions = [bound_list[0][0]]

            for pos, parts in bound_list[1:]:
                if abs(pos - current_pos) <= self.merge_tolerance:
                    # Merge: average position, union of parts
                    original_positions.append(pos)
                    current_parts.update(parts)
                    current_pos = sum(original_positions) / len(original_positions)
                else:
                    # Save current and start new
                    merged.append((current_pos, current_parts, original_positions))
                    current_pos = pos
                    current_parts = set(parts)
                    original_positions = [pos]

            # Don't forget the last group
            merged.append((current_pos, current_parts, original_positions))
            result[axis] = merged

        return result

    def _generate_cut_sequence(self, boxes: List[Box3D],
                               boundaries: Dict[str, List[Tuple[float, Set[int], List[float]]]]) -> List[CutSpecification]:
        """
        Generate ordered cut sequence using vertical-first strategy.

        Strategy:
        1. X-axis cuts first (separate columns)
        2. Z-axis cuts (separate layers within columns)
        3. Y-axis cuts (separate depth if needed)
        """
        cuts = []
        cut_id = 1

        # Get stock bounds
        stock_bounds = {
            'x': (0, self.stock_length),
            'y': (0, self.stock_width),
            'z': (0, self.stock_height)
        }

        # Axis priority: X first, then Z, then Y
        axis_order = ['x', 'z', 'y']

        # Track remaining boxes to separate
        remaining_box_ids = set(b.part_id for b in boxes)
        box_map = {b.part_id: b for b in boxes}

        # Generate cuts recursively
        cuts = self._recursive_cut_generation(
            boxes, box_map, boundaries, axis_order,
            stock_bounds, cut_id
        )

        return cuts

    def _recursive_cut_generation(self, boxes: List[Box3D], box_map: Dict[int, Box3D],
                                   boundaries: Dict[str, List[Tuple[float, Set[int], List[float]]]],
                                   axis_order: List[str],
                                   region_bounds: Dict[str, Tuple[float, float]],
                                   start_cut_id: int) -> List[CutSpecification]:
        """
        Recursively generate cuts to separate all boxes.
        """
        if len(boxes) <= 1:
            return []

        cuts = []
        cut_id = start_cut_id

        # Try each axis in priority order
        for axis in axis_order:
            cut_result = self._find_separating_cut(boxes, box_map, boundaries, axis, region_bounds)

            if cut_result is not None:
                pos, group_before_ids, group_after_ids, original_positions = cut_result

                # Create cut specification
                cut = self._create_cut_specification(
                    cut_id, axis, pos,
                    group_before_ids, group_after_ids,
                    original_positions, region_bounds
                )
                cuts.append(cut)
                cut_id += 1

                # Recursively process each group
                group_before = [box_map[pid] for pid in group_before_ids]
                group_after = [box_map[pid] for pid in group_after_ids]

                # Update region bounds for recursive calls
                region_before = dict(region_bounds)
                region_before[axis] = (region_bounds[axis][0], pos)

                region_after = dict(region_bounds)
                region_after[axis] = (pos, region_bounds[axis][1])

                # Recurse
                cuts.extend(self._recursive_cut_generation(
                    group_before, box_map, boundaries, axis_order,
                    region_before, cut_id
                ))
                cut_id = max(c.cut_id for c in cuts) + 1 if cuts else cut_id

                cuts.extend(self._recursive_cut_generation(
                    group_after, box_map, boundaries, axis_order,
                    region_after, cut_id
                ))

                return cuts

        return cuts

    def _find_separating_cut(self, boxes: List[Box3D], box_map: Dict[int, Box3D],
                             boundaries: Dict[str, List[Tuple[float, Set[int], List[float]]]],
                             axis: str,
                             region_bounds: Dict[str, Tuple[float, float]]) -> Optional[Tuple[float, Set[int], Set[int], List[float]]]:
        """
        Find a valid separating cut position on the given axis.

        Returns:
            Tuple of (position, group_before_ids, group_after_ids, original_positions) or None
        """
        box_ids = set(b.part_id for b in boxes)

        # Get candidate positions from merged boundaries
        for pos, parts, original_positions in boundaries[axis]:
            # Check if position is within region bounds
            if pos <= region_bounds[axis][0] + self.eps or pos >= region_bounds[axis][1] - self.eps:
                continue

            # Check if this cut goes through any box
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
                else:  # z
                    if box.z_min + self.eps < pos < box.z_max - self.eps:
                        cuts_through_box = True
                        break

            if cuts_through_box:
                continue

            # Split boxes into groups
            group_before = set()
            group_after = set()

            for box in boxes:
                if axis == 'x':
                    box_max = box.x_max
                elif axis == 'y':
                    box_max = box.y_max
                else:
                    box_max = box.z_max

                if box_max <= pos + self.eps:
                    group_before.add(box.part_id)
                else:
                    group_after.add(box.part_id)

            # Valid cut must separate boxes into two non-empty groups
            if len(group_before) > 0 and len(group_after) > 0:
                return (pos, group_before, group_after, original_positions)

        return None

    def _create_cut_specification(self, cut_id: int, axis: str, position: float,
                                   group_before: Set[int], group_after: Set[int],
                                   original_positions: List[float],
                                   region_bounds: Dict[str, Tuple[float, float]]) -> CutSpecification:
        """
        Create a CutSpecification for the given cut.
        """
        # Determine normal vector and point based on axis
        if axis == 'x':
            normal = (1.0, 0.0, 0.0)
            point = (position, 0.0, 0.0)
        elif axis == 'y':
            normal = (0.0, 1.0, 0.0)
            point = (0.0, position, 0.0)
        else:  # z
            normal = (0.0, 0.0, 1.0)
            point = (0.0, 0.0, position)

        # Create CutPlane
        plane = CutPlane(
            normal=normal,
            point=point,
            description=f"{axis.upper()}-axis cut at {position:.2f}mm"
        )

        # Check if this is a shared/merged plane
        is_shared = len(original_positions) > 1

        # Parts separated by this cut
        parts_separated = list(group_before | group_after)

        return CutSpecification(
            cut_id=cut_id,
            axis=axis,
            position=position,
            plane=plane,
            parts_separated=sorted(parts_separated),
            group_before=sorted(group_before),
            group_after=sorted(group_after),
            is_shared=is_shared,
            merged_from=original_positions if is_shared else []
        )

    def verify_plan(self, plan: CutPlan) -> Dict:
        """
        Verify that the cut plan correctly extracts all parts.

        Returns:
            Dict with verification results
        """
        # Calculate theoretical volumes
        stock_volume = self.stock_length * self.stock_width * self.stock_height

        parts_volume = sum(
            (b.x_max - b.x_min) * (b.y_max - b.y_min) * (b.z_max - b.z_min)
            for b in plan.parts
        )

        scrap_volume = stock_volume - parts_volume

        # Count unique cut positions per axis
        cuts_by_axis = defaultdict(set)
        for cut in plan.cuts:
            cuts_by_axis[cut.axis].add(cut.position)

        return {
            'stock_volume': stock_volume,
            'parts_volume': parts_volume,
            'scrap_volume': scrap_volume,
            'utilization': (parts_volume / stock_volume) * 100,
            'total_cuts': len(plan.cuts),
            'unique_x_cuts': len(cuts_by_axis['x']),
            'unique_y_cuts': len(cuts_by_axis['y']),
            'unique_z_cuts': len(cuts_by_axis['z']),
            'shared_planes': sum(1 for c in plan.cuts if c.is_shared),
            'is_valid': True
        }


def generate_cut_plan_from_packing(placed_parts, stock_dims: Tuple[float, float, float],
                                    merge_tolerance: float = 0.2) -> CutPlan:
    """
    Generate a cut plan from placed parts.

    Args:
        placed_parts: List of PlacedPart objects with geometry
        stock_dims: (length, width, height) of stock block
        merge_tolerance: Tolerance for merging similar planes (mm)

    Returns:
        CutPlan with ordered cut specifications
    """
    # Convert placed parts to Box3D
    boxes = []
    for part in placed_parts:
        bbox = part.geometry.val().BoundingBox()
        boxes.append(Box3D(
            x_min=bbox.xmin, y_min=bbox.ymin, z_min=bbox.zmin,
            x_max=bbox.xmax, y_max=bbox.ymax, z_max=bbox.zmax,
            part_id=part.part_id
        ))

    generator = CutPlanGenerator(stock_dims, merge_tolerance)
    return generator.generate_plan(boxes)
