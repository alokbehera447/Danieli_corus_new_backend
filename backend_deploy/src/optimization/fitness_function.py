"""
Fitness Function for Genetic Algorithm Optimization.

Evaluates packing solutions based on:
- Waste percentage (primary objective)
- Overlap penalties
- Out-of-bounds penalties
- Extractability penalties
- Number of cuts (efficiency)

Lower fitness = better solution.

Fitness formula:
    fitness = waste_percentage + penalties + cut_penalty

Where:
    waste_percentage = (stock_volume - parts_volume) / stock_volume * 100
    penalties = overlaps * 50 + out_of_bounds * 10 + (not extractable) * 1000
    cut_penalty = num_cuts / (num_parts + 1) * 10
"""

import numpy as np
import cadquery as cq
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field

from ..packing.trapezoid_geometry import TrapezoidGeometry
from ..packing.orientation_explorer import OrientationExplorer, Orientation
from ..packing.packing_algorithms import PlacedPart
from ..cutting.guillotine_validator import (
    GuillotineValidatorRecursive,
    Box3D
)


# =============================================================================
# PENALTY WEIGHTS (Configurable)
# =============================================================================

@dataclass
class PenaltyWeights:
    """Configurable penalty weights for fitness function."""
    overlap: float = 50.0           # Per overlapping pair
    out_of_bounds: float = 10.0     # Per out-of-bounds part
    non_extractable: float = 1000.0 # If packing is not guillotine-extractable
    cut_factor: float = 10.0        # Multiplier for cut penalty term


# Default weights
DEFAULT_WEIGHTS = PenaltyWeights()


# =============================================================================
# INDIVIDUAL REPRESENTATION
# =============================================================================

@dataclass
class Gene:
    """
    Single gene representing one part placement.

    Attributes:
        x, y, z: Position coordinates
        orientation_idx: Index into orientation library
        mirrored: Whether part is mirrored/flipped
    """
    x: float
    y: float
    z: float
    orientation_idx: int
    mirrored: bool = False

    def to_tuple(self) -> Tuple:
        return (self.x, self.y, self.z, self.orientation_idx, self.mirrored)

    @classmethod
    def from_tuple(cls, t: Tuple) -> 'Gene':
        return cls(x=t[0], y=t[1], z=t[2], orientation_idx=t[3], mirrored=t[4] if len(t) > 4 else False)


@dataclass
class Individual:
    """
    Individual in the GA population (variable-length chromosome).

    Chromosome = List of genes, each representing a part placement.
    """
    genes: List[Gene] = field(default_factory=list)

    def __len__(self):
        return len(self.genes)

    def to_list(self) -> List[Tuple]:
        return [g.to_tuple() for g in self.genes]

    @classmethod
    def from_list(cls, lst: List[Tuple]) -> 'Individual':
        return cls(genes=[Gene.from_tuple(t) for t in lst])


# =============================================================================
# FITNESS EVALUATION RESULT
# =============================================================================

@dataclass
class FitnessResult:
    """
    Detailed fitness evaluation result.

    Attributes:
        fitness: Final fitness score (lower = better)
        waste_percentage: Percentage of stock wasted
        num_parts: Number of valid parts placed
        num_overlaps: Number of overlapping part pairs
        num_out_of_bounds: Number of parts outside stock
        is_extractable: Whether packing is guillotine-extractable
        num_cuts: Estimated number of cuts needed
        penalties: Dictionary of penalty components
        valid: Whether solution is valid (no overlaps, in bounds, extractable)
    """
    fitness: float
    waste_percentage: float
    num_parts: int
    num_overlaps: int
    num_out_of_bounds: int
    is_extractable: bool
    num_cuts: int
    penalties: Dict[str, float]
    valid: bool

    def __repr__(self):
        return (f"FitnessResult(fitness={self.fitness:.2f}, "
                f"parts={self.num_parts}, waste={self.waste_percentage:.2f}%, "
                f"valid={self.valid})")


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_part_bbox(position: Tuple[float, float, float],
                  dims: Tuple[float, float, float]) -> Tuple[float, float, float, float, float, float]:
    """Get bounding box (xmin, ymin, zmin, xmax, ymax, zmax) for a part."""
    x, y, z = position
    dx, dy, dz = dims
    return (x, y, z, x + dx, y + dy, z + dz)


def boxes_overlap(box1: Tuple, box2: Tuple, eps: float = 1e-6) -> bool:
    """Check if two bounding boxes overlap."""
    x1_min, y1_min, z1_min, x1_max, y1_max, z1_max = box1
    x2_min, y2_min, z2_min, x2_max, y2_max, z2_max = box2

    return not (
        x1_max <= x2_min + eps or x2_max <= x1_min + eps or
        y1_max <= y2_min + eps or y2_max <= y1_min + eps or
        z1_max <= z2_min + eps or z2_max <= z1_min + eps
    )


def is_in_bounds(box: Tuple, stock_dims: Tuple[float, float, float], eps: float = 1e-6) -> bool:
    """Check if bounding box is within stock dimensions."""
    x_min, y_min, z_min, x_max, y_max, z_max = box
    sx, sy, sz = stock_dims

    return (
        x_min >= -eps and y_min >= -eps and z_min >= -eps and
        x_max <= sx + eps and y_max <= sy + eps and z_max <= sz + eps
    )


def count_overlapping_pairs(boxes: List[Tuple]) -> int:
    """Count number of overlapping box pairs."""
    count = 0
    n = len(boxes)
    for i in range(n):
        for j in range(i + 1, n):
            if boxes_overlap(boxes[i], boxes[j]):
                count += 1
    return count


def count_out_of_bounds(boxes: List[Tuple], stock_dims: Tuple[float, float, float]) -> int:
    """Count number of boxes outside stock bounds."""
    count = 0
    for box in boxes:
        if not is_in_bounds(box, stock_dims):
            count += 1
    return count


def estimate_num_cuts(num_parts: int) -> int:
    """
    Estimate number of guillotine cuts needed.

    For a regular grid of n parts, approximately n-1 cuts are needed
    in the simplest case. For 3D, it's roughly 3*(n^(1/3) - 1) * n^(2/3).

    Simplified: Use n-1 as lower bound.
    """
    if num_parts <= 1:
        return 0
    return num_parts - 1


# =============================================================================
# MAIN FITNESS FUNCTION
# =============================================================================

class FitnessEvaluator:
    """
    Fitness function evaluator for GA optimization.

    Evaluates individuals (packing configurations) and returns fitness scores.
    Lower fitness = better solution.
    """

    def __init__(self,
                 stock_dims: Tuple[float, float, float],
                 part_geom: TrapezoidGeometry,
                 explorer: OrientationExplorer,
                 weights: PenaltyWeights = None):
        """
        Initialize fitness evaluator.

        Args:
            stock_dims: Stock block dimensions (length, width, height)
            part_geom: Part geometry specification
            explorer: OrientationExplorer with valid orientations
            weights: Penalty weights (default: DEFAULT_WEIGHTS)
        """
        self.stock_dims = stock_dims
        self.part_geom = part_geom
        self.explorer = explorer
        self.weights = weights or DEFAULT_WEIGHTS

        self.stock_volume = stock_dims[0] * stock_dims[1] * stock_dims[2]
        self.part_volume = part_geom.volume

    def decode_individual(self, individual: Individual) -> List[Tuple[Tuple, Tuple]]:
        """
        Decode individual genes into part placements.

        Returns:
            List of (position, dims) tuples
        """
        placements = []
        for gene in individual.genes:
            # Get orientation
            orientation = self.explorer.get_orientation(gene.orientation_idx)
            if orientation is None:
                continue

            position = (gene.x, gene.y, gene.z)
            dims = orientation.dims
            placements.append((position, dims, gene.orientation_idx, gene.mirrored))

        return placements

    def evaluate(self, individual: Individual) -> FitnessResult:
        """
        Evaluate fitness of an individual.

        Args:
            individual: Individual to evaluate

        Returns:
            FitnessResult with detailed breakdown
        """
        # Decode individual
        placements = self.decode_individual(individual)

        if not placements:
            # Empty solution - maximum penalty
            return FitnessResult(
                fitness=float('inf'),
                waste_percentage=100.0,
                num_parts=0,
                num_overlaps=0,
                num_out_of_bounds=0,
                is_extractable=True,
                num_cuts=0,
                penalties={"empty": float('inf')},
                valid=False
            )

        # Calculate bounding boxes
        boxes = []
        for pos, dims, ori_idx, mirrored in placements:
            boxes.append(get_part_bbox(pos, dims))

        # Count overlaps
        num_overlaps = count_overlapping_pairs(boxes)

        # Count out-of-bounds
        num_out_of_bounds = count_out_of_bounds(boxes, self.stock_dims)

        # Filter valid parts (in bounds, not overlapping)
        valid_boxes = [b for b in boxes if is_in_bounds(b, self.stock_dims)]

        # Check extractability using guillotine validator
        # Only check if no overlaps and all in bounds
        is_extractable = True
        if num_overlaps == 0 and num_out_of_bounds == 0 and len(valid_boxes) > 1:
            box3d_list = []
            for i, box in enumerate(valid_boxes):
                box3d_list.append(Box3D(
                    x_min=box[0], y_min=box[1], z_min=box[2],
                    x_max=box[3], y_max=box[4], z_max=box[5],
                    part_id=i
                ))
            validator = GuillotineValidatorRecursive(self.stock_dims)
            result = validator.validate(box3d_list)
            is_extractable = result.is_valid

        # Short-circuit if not extractable
        if not is_extractable:
            return FitnessResult(
                fitness=self.weights.non_extractable,
                waste_percentage=100.0,
                num_parts=len(placements),
                num_overlaps=num_overlaps,
                num_out_of_bounds=num_out_of_bounds,
                is_extractable=False,
                num_cuts=0,
                penalties={"non_extractable": self.weights.non_extractable},
                valid=False
            )

        # Calculate waste
        num_valid_parts = len(valid_boxes)
        parts_volume = num_valid_parts * self.part_volume
        waste_volume = self.stock_volume - parts_volume
        waste_percentage = (waste_volume / self.stock_volume) * 100

        # Estimate cuts
        num_cuts = estimate_num_cuts(num_valid_parts)

        # Calculate penalties
        penalties = {}
        penalty_total = 0.0

        if num_overlaps > 0:
            penalties["overlap"] = num_overlaps * self.weights.overlap
            penalty_total += penalties["overlap"]

        if num_out_of_bounds > 0:
            penalties["out_of_bounds"] = num_out_of_bounds * self.weights.out_of_bounds
            penalty_total += penalties["out_of_bounds"]

        # Cut penalty term: num_cuts / (num_parts + 1) * factor
        cut_penalty = (num_cuts / (num_valid_parts + 1)) * self.weights.cut_factor
        penalties["cuts"] = cut_penalty

        # Final fitness
        fitness = waste_percentage + penalty_total + cut_penalty

        # Determine validity
        valid = (num_overlaps == 0 and num_out_of_bounds == 0 and is_extractable)

        return FitnessResult(
            fitness=fitness,
            waste_percentage=waste_percentage,
            num_parts=num_valid_parts,
            num_overlaps=num_overlaps,
            num_out_of_bounds=num_out_of_bounds,
            is_extractable=is_extractable,
            num_cuts=num_cuts,
            penalties=penalties,
            valid=valid
        )

    def evaluate_tuple(self, individual_list: List[Tuple]) -> Tuple[float]:
        """
        Evaluate fitness from tuple representation (DEAP compatible).

        Args:
            individual_list: List of (x, y, z, orientation_idx, mirrored) tuples

        Returns:
            Tuple containing single fitness value (for DEAP)
        """
        individual = Individual.from_list(individual_list)
        result = self.evaluate(individual)
        return (result.fitness,)

    def create_individual_from_packing(self, placed_parts: List[PlacedPart],
                                        orientation_idx: int = 0) -> Individual:
        """
        Create an Individual from a list of PlacedPart objects.

        Args:
            placed_parts: List of placed parts from greedy packing
            orientation_idx: Orientation index to use

        Returns:
            Individual representing the packing
        """
        genes = []
        for part in placed_parts:
            gene = Gene(
                x=part.position[0],
                y=part.position[1],
                z=part.position[2],
                orientation_idx=orientation_idx,
                mirrored=part.mirrored
            )
            genes.append(gene)
        return Individual(genes=genes)


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def evaluate_packing_fitness(placed_parts: List[PlacedPart],
                              stock_dims: Tuple[float, float, float],
                              part_geom: TrapezoidGeometry,
                              weights: PenaltyWeights = None) -> FitnessResult:
    """
    Convenience function to evaluate fitness of a packing solution.

    Args:
        placed_parts: List of placed parts
        stock_dims: Stock dimensions
        part_geom: Part geometry
        weights: Optional penalty weights

    Returns:
        FitnessResult with detailed breakdown
    """
    explorer = OrientationExplorer(part_geom)
    evaluator = FitnessEvaluator(stock_dims, part_geom, explorer, weights)

    # Convert placed parts to boxes
    boxes = []
    for part in placed_parts:
        bbox = part.geometry.val().BoundingBox()
        boxes.append((bbox.xmin, bbox.ymin, bbox.zmin, bbox.xmax, bbox.ymax, bbox.zmax))

    # Count metrics
    num_overlaps = count_overlapping_pairs(boxes)
    num_out_of_bounds = count_out_of_bounds(boxes, stock_dims)

    # Check extractability
    is_extractable = True
    if num_overlaps == 0 and num_out_of_bounds == 0 and len(boxes) > 1:
        box3d_list = []
        for i, box in enumerate(boxes):
            box3d_list.append(Box3D(
                x_min=box[0], y_min=box[1], z_min=box[2],
                x_max=box[3], y_max=box[4], z_max=box[5],
                part_id=i
            ))
        validator = GuillotineValidatorRecursive(stock_dims)
        result = validator.validate(box3d_list)
        is_extractable = result.is_valid

    # Calculate waste
    stock_volume = stock_dims[0] * stock_dims[1] * stock_dims[2]
    parts_volume = len(boxes) * part_geom.volume
    waste_percentage = ((stock_volume - parts_volume) / stock_volume) * 100

    # Estimate cuts
    num_cuts = estimate_num_cuts(len(boxes))

    # Calculate penalties
    w = weights or DEFAULT_WEIGHTS
    penalties = {}
    penalty_total = 0.0

    if num_overlaps > 0:
        penalties["overlap"] = num_overlaps * w.overlap
        penalty_total += penalties["overlap"]

    if num_out_of_bounds > 0:
        penalties["out_of_bounds"] = num_out_of_bounds * w.out_of_bounds
        penalty_total += penalties["out_of_bounds"]

    if not is_extractable:
        penalties["non_extractable"] = w.non_extractable
        penalty_total += penalties["non_extractable"]

    cut_penalty = (num_cuts / (len(boxes) + 1)) * w.cut_factor
    penalties["cuts"] = cut_penalty

    fitness = waste_percentage + penalty_total + cut_penalty
    valid = (num_overlaps == 0 and num_out_of_bounds == 0 and is_extractable)

    return FitnessResult(
        fitness=fitness,
        waste_percentage=waste_percentage,
        num_parts=len(boxes),
        num_overlaps=num_overlaps,
        num_out_of_bounds=num_out_of_bounds,
        is_extractable=is_extractable,
        num_cuts=num_cuts,
        penalties=penalties,
        valid=valid
    )
