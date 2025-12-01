"""
Boolean cutting operations for extracting parts from stock blocks.

Provides validated cutting operations using CadQuery's boolean subtraction,
with comprehensive pre/post validation and error handling.

Tolerance Settings:
- Length: 0.2 mm
- Area: 4 mm²
- Volume: 8 mm³
"""

import cadquery as cq
import numpy as np
from typing import Tuple, Optional, Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime
import json
import logging

from ..geometry.data_classes import CutPlane


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# TOLERANCE CONSTANTS
# =============================================================================

class Tolerances:
    """Tolerance settings for cutting operations."""
    LENGTH = 0.2       # mm - tolerance for length/position comparisons
    AREA = 4.0         # mm² - tolerance for area comparisons
    VOLUME = 8.0       # mm³ - tolerance for volume comparisons
    RELATIVE = 0.001   # 0.1% - relative tolerance for volume conservation


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class CutResult:
    """
    Result of a cutting operation.

    Attributes:
        success: Whether the cut was successful
        piece_negative: Geometry on negative side of plane (below/left/front)
        piece_positive: Geometry on positive side of plane (above/right/back)
        original_volume: Volume of original geometry
        negative_volume: Volume of negative side piece
        positive_volume: Volume of positive side piece
        volume_error: Absolute volume conservation error
        relative_error: Relative volume conservation error
        message: Status message
        diagnostics: Detailed diagnostic information
    """
    success: bool
    piece_negative: Optional[cq.Workplane] = None
    piece_positive: Optional[cq.Workplane] = None
    original_volume: float = 0.0
    negative_volume: float = 0.0
    positive_volume: float = 0.0
    volume_error: float = 0.0
    relative_error: float = 0.0
    message: str = ""
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def is_volume_conserved(self) -> bool:
        """Check if volume is conserved within tolerance."""
        return (self.volume_error <= Tolerances.VOLUME or
                self.relative_error <= Tolerances.RELATIVE)


@dataclass
class ValidationResult:
    """Result of geometry validation."""
    is_valid: bool
    volume: float = 0.0
    is_manifold: bool = True
    has_positive_volume: bool = True
    bbox: Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = None
    errors: List[str] = field(default_factory=list)


@dataclass
class CutError:
    """Detailed error report for failed cut operations."""
    error_type: str
    error_message: str
    geometry_volume: float
    geometry_bbox: Optional[Tuple] = None
    plane_equation: Optional[Tuple[float, float, float, float]] = None
    plane_normal: Optional[Tuple[float, float, float]] = None
    plane_point: Optional[Tuple[float, float, float]] = None
    timestamp: str = ""
    additional_info: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> Dict:
        return {
            "error_type": self.error_type,
            "error_message": self.error_message,
            "geometry_volume": self.geometry_volume,
            "geometry_bbox": self.geometry_bbox,
            "plane_equation": self.plane_equation,
            "plane_normal": self.plane_normal,
            "plane_point": self.plane_point,
            "timestamp": self.timestamp,
            "additional_info": self.additional_info
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, default=str)


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

def validate_geometry(geometry: cq.Workplane) -> ValidationResult:
    """
    Validate that a geometry is valid for cutting operations.

    Checks:
    - Positive volume
    - Valid bounding box
    - Geometry exists

    Args:
        geometry: CadQuery Workplane object

    Returns:
        ValidationResult with validation status and details
    """
    errors = []
    volume = 0.0
    bbox = None
    has_positive_volume = False
    is_manifold = True

    try:
        # Get the solid
        solid = geometry.val()

        # Check volume
        volume = solid.Volume()
        has_positive_volume = volume > Tolerances.VOLUME

        if not has_positive_volume:
            errors.append(f"Volume too small or zero: {volume:.6f} mm³")

        # Get bounding box
        bb = solid.BoundingBox()
        bbox = (
            (bb.xmin, bb.ymin, bb.zmin),
            (bb.xmax, bb.ymax, bb.zmax)
        )

        # Check for degenerate dimensions
        dims = (bb.xmax - bb.xmin, bb.ymax - bb.ymin, bb.zmax - bb.zmin)
        for i, dim in enumerate(dims):
            if dim < Tolerances.LENGTH:
                axis = ['X', 'Y', 'Z'][i]
                errors.append(f"Degenerate dimension on {axis}-axis: {dim:.6f} mm")

    except Exception as e:
        errors.append(f"Geometry validation error: {str(e)}")
        is_manifold = False

    return ValidationResult(
        is_valid=len(errors) == 0,
        volume=volume,
        is_manifold=is_manifold,
        has_positive_volume=has_positive_volume,
        bbox=bbox,
        errors=errors
    )


def plane_intersects_geometry(geometry: cq.Workplane, plane: CutPlane) -> Tuple[bool, Dict]:
    """
    Check if a cutting plane intersects the geometry.

    Args:
        geometry: CadQuery Workplane object
        plane: CutPlane defining the cutting plane

    Returns:
        Tuple of (intersects: bool, details: dict)
    """
    try:
        solid = geometry.val()
        bb = solid.BoundingBox()

        # Get plane equation Ax + By + Cz + D = 0
        A, B, C, D = plane.get_equation()

        # Check all 8 corners of bounding box
        corners = [
            (bb.xmin, bb.ymin, bb.zmin),
            (bb.xmax, bb.ymin, bb.zmin),
            (bb.xmin, bb.ymax, bb.zmin),
            (bb.xmax, bb.ymax, bb.zmin),
            (bb.xmin, bb.ymin, bb.zmax),
            (bb.xmax, bb.ymin, bb.zmax),
            (bb.xmin, bb.ymax, bb.zmax),
            (bb.xmax, bb.ymax, bb.zmax),
        ]

        # Calculate signed distances from plane
        distances = []
        for x, y, z in corners:
            dist = A * x + B * y + C * z + D
            distances.append(dist)

        min_dist = min(distances)
        max_dist = max(distances)

        # Check if plane passes through bounding box
        # (corners on both sides of plane)
        passes_through = (min_dist < -Tolerances.LENGTH and max_dist > Tolerances.LENGTH)

        # Check if plane is at exact boundary (within tolerance)
        at_boundary = (
            (abs(min_dist) < Tolerances.LENGTH and max_dist > Tolerances.LENGTH) or
            (min_dist < -Tolerances.LENGTH and abs(max_dist) < Tolerances.LENGTH)
        )

        # Check if plane misses entirely
        misses = (min_dist > Tolerances.LENGTH or max_dist < -Tolerances.LENGTH)

        details = {
            "min_distance": min_dist,
            "max_distance": max_dist,
            "passes_through": passes_through,
            "at_boundary": at_boundary,
            "misses_geometry": misses,
            "corner_distances": list(zip(corners, distances))
        }

        return passes_through, details

    except Exception as e:
        return False, {"error": str(e)}


def volumes_match(v1: float, v2: float,
                  absolute_tolerance: float = Tolerances.VOLUME,
                  relative_tolerance: float = Tolerances.RELATIVE) -> Tuple[bool, float, float]:
    """
    Check if two volumes match within tolerance.

    Args:
        v1: First volume
        v2: Second volume
        absolute_tolerance: Absolute tolerance in mm³
        relative_tolerance: Relative tolerance (0.001 = 0.1%)

    Returns:
        Tuple of (match: bool, absolute_error: float, relative_error: float)
    """
    absolute_error = abs(v1 - v2)
    relative_error = absolute_error / max(v1, v2, 1e-10)

    match = (absolute_error <= absolute_tolerance or
             relative_error <= relative_tolerance)

    return match, absolute_error, relative_error


# =============================================================================
# CUTTING OPERATIONS
# =============================================================================

def create_cutting_halfspace(plane: CutPlane,
                              bounds: Tuple[float, float, float, float, float, float],
                              extension: float = 100.0) -> cq.Workplane:
    """
    Create a half-space solid for cutting.

    Creates a large box on the positive side of the plane that can be
    subtracted from geometry to produce a cut.

    Args:
        plane: CutPlane defining the cutting plane
        bounds: Bounding box (xmin, ymin, zmin, xmax, ymax, zmax)
        extension: How far to extend beyond bounds

    Returns:
        CadQuery Workplane representing the half-space
    """
    xmin, ymin, zmin, xmax, ymax, zmax = bounds

    # Extend bounds
    xmin -= extension
    ymin -= extension
    zmin -= extension
    xmax += extension
    ymax += extension
    zmax += extension

    # Create a large box
    size_x = xmax - xmin
    size_y = ymax - ymin
    size_z = zmax - zmin

    # Get plane normal and point
    nx, ny, nz = plane.normal
    px, py, pz = plane.point

    # Create box and position it on positive side of plane
    # We'll create a box and then cut it with the plane
    center_x = (xmin + xmax) / 2
    center_y = (ymin + ymax) / 2
    center_z = (zmin + zmax) / 2

    # Create the box
    box = (cq.Workplane("XY")
           .box(size_x, size_y, size_z)
           .translate((center_x, center_y, center_z)))

    # Create a workplane on the cutting plane
    # Direction is the normal vector
    origin = plane.point

    # Create the cutting face using a sketch on the plane
    # Then extrude in the negative normal direction to create the "keep" side
    max_dim = max(size_x, size_y, size_z) * 2

    # Create a face on the plane and extrude
    halfspace = (cq.Workplane(cq.Plane(origin=origin,
                                        xDir=_perpendicular_vector(plane.normal),
                                        normal=plane.normal))
                 .rect(max_dim, max_dim)
                 .extrude(max_dim))

    return halfspace


def _perpendicular_vector(normal: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """Find a vector perpendicular to the given normal."""
    nx, ny, nz = normal

    # Choose the axis least aligned with normal
    if abs(nx) <= abs(ny) and abs(nx) <= abs(nz):
        perp = (0, -nz, ny)
    elif abs(ny) <= abs(nz):
        perp = (-nz, 0, nx)
    else:
        perp = (-ny, nx, 0)

    # Normalize
    mag = np.sqrt(perp[0]**2 + perp[1]**2 + perp[2]**2)
    if mag < 1e-10:
        return (1.0, 0.0, 0.0)
    return (perp[0]/mag, perp[1]/mag, perp[2]/mag)


def cut_with_plane(geometry: cq.Workplane, plane: CutPlane,
                   validate: bool = True) -> CutResult:
    """
    Cut geometry with a plane, returning both pieces.

    Uses CadQuery's cut() operation (boolean subtraction) to split
    the geometry into two pieces along the cutting plane.

    Args:
        geometry: CadQuery Workplane to cut
        plane: CutPlane defining the cutting plane
        validate: Whether to perform pre/post validation

    Returns:
        CutResult with both pieces and validation info

    Raises:
        ValueError: If validation fails and cut cannot proceed
    """
    diagnostics = {
        "plane_equation": plane.get_equation(),
        "plane_normal": plane.normal,
        "plane_point": plane.point,
        "validation_enabled": validate
    }

    # Pre-validation: Check original geometry
    if validate:
        orig_validation = validate_geometry(geometry)
        diagnostics["original_validation"] = {
            "is_valid": orig_validation.is_valid,
            "volume": orig_validation.volume,
            "errors": orig_validation.errors
        }

        if not orig_validation.is_valid:
            error = CutError(
                error_type="InvalidInputGeometry",
                error_message=f"Input geometry is invalid: {orig_validation.errors}",
                geometry_volume=orig_validation.volume,
                geometry_bbox=orig_validation.bbox,
                plane_equation=plane.get_equation(),
                plane_normal=plane.normal,
                plane_point=plane.point
            )
            logger.error(f"Cut failed: {error.to_json()}")
            raise ValueError(error.error_message)

        original_volume = orig_validation.volume
    else:
        original_volume = geometry.val().Volume()

    # Pre-validation: Check plane intersection
    if validate:
        intersects, intersection_details = plane_intersects_geometry(geometry, plane)
        diagnostics["intersection_check"] = intersection_details

        if intersection_details.get("misses_geometry", False):
            error = CutError(
                error_type="PlaneDoesNotIntersect",
                error_message="Cutting plane does not intersect geometry",
                geometry_volume=original_volume,
                plane_equation=plane.get_equation(),
                plane_normal=plane.normal,
                plane_point=plane.point,
                additional_info=intersection_details
            )
            logger.error(f"Cut failed: {error.to_json()}")
            raise ValueError(error.error_message)

        if intersection_details.get("at_boundary", False):
            error = CutError(
                error_type="PlaneAtBoundary",
                error_message="Cutting plane is at exact boundary of geometry",
                geometry_volume=original_volume,
                plane_equation=plane.get_equation(),
                plane_normal=plane.normal,
                plane_point=plane.point,
                additional_info=intersection_details
            )
            logger.error(f"Cut failed: {error.to_json()}")
            raise ValueError(error.error_message)

    # Get bounding box for half-space creation
    bb = geometry.val().BoundingBox()
    bounds = (bb.xmin, bb.ymin, bb.zmin, bb.xmax, bb.ymax, bb.zmax)

    try:
        # Create cutting half-space (positive side of plane)
        halfspace = create_cutting_halfspace(plane, bounds)

        # Piece on negative side = original - halfspace
        piece_negative = geometry.cut(halfspace)

        # Piece on positive side = original ∩ halfspace (intersect)
        piece_positive = geometry.intersect(halfspace)

        diagnostics["cut_method"] = "boolean_subtraction_and_intersection"

    except Exception as e:
        error = CutError(
            error_type="BooleanOperationFailed",
            error_message=f"Boolean operation failed: {str(e)}",
            geometry_volume=original_volume,
            geometry_bbox=bounds,
            plane_equation=plane.get_equation(),
            plane_normal=plane.normal,
            plane_point=plane.point
        )
        logger.error(f"Cut failed: {error.to_json()}")
        raise ValueError(error.error_message)

    # Post-validation: Validate resulting pieces
    negative_volume = 0.0
    positive_volume = 0.0

    if validate:
        neg_validation = validate_geometry(piece_negative)
        pos_validation = validate_geometry(piece_positive)

        diagnostics["negative_piece_validation"] = {
            "is_valid": neg_validation.is_valid,
            "volume": neg_validation.volume,
            "errors": neg_validation.errors
        }
        diagnostics["positive_piece_validation"] = {
            "is_valid": pos_validation.is_valid,
            "volume": pos_validation.volume,
            "errors": pos_validation.errors
        }

        if not neg_validation.is_valid or not pos_validation.is_valid:
            all_errors = neg_validation.errors + pos_validation.errors
            error = CutError(
                error_type="InvalidResultGeometry",
                error_message=f"Cut produced invalid geometry: {all_errors}",
                geometry_volume=original_volume,
                plane_equation=plane.get_equation(),
                plane_normal=plane.normal,
                plane_point=plane.point,
                additional_info={"validation_errors": all_errors}
            )
            logger.error(f"Cut failed: {error.to_json()}")
            raise ValueError(error.error_message)

        negative_volume = neg_validation.volume
        positive_volume = pos_validation.volume
    else:
        negative_volume = piece_negative.val().Volume()
        positive_volume = piece_positive.val().Volume()

    # Volume conservation check
    total_volume = negative_volume + positive_volume
    match, abs_error, rel_error = volumes_match(original_volume, total_volume)

    diagnostics["volume_conservation"] = {
        "original": original_volume,
        "negative_piece": negative_volume,
        "positive_piece": positive_volume,
        "total_after": total_volume,
        "absolute_error": abs_error,
        "relative_error": rel_error,
        "is_conserved": match
    }

    if validate and not match:
        error = CutError(
            error_type="VolumeNotConserved",
            error_message=f"Volume not conserved: original={original_volume:.2f}, "
                         f"after={total_volume:.2f}, error={abs_error:.2f} mm³ ({rel_error*100:.4f}%)",
            geometry_volume=original_volume,
            plane_equation=plane.get_equation(),
            plane_normal=plane.normal,
            plane_point=plane.point,
            additional_info=diagnostics["volume_conservation"]
        )
        logger.error(f"Cut failed: {error.to_json()}")
        raise ValueError(error.error_message)

    return CutResult(
        success=True,
        piece_negative=piece_negative,
        piece_positive=piece_positive,
        original_volume=original_volume,
        negative_volume=negative_volume,
        positive_volume=positive_volume,
        volume_error=abs_error,
        relative_error=rel_error,
        message=f"Cut successful: {negative_volume:.2f} + {positive_volume:.2f} = {total_volume:.2f} mm³",
        diagnostics=diagnostics
    )


def cut_sequential(geometry: cq.Workplane, planes: List[CutPlane],
                   validate: bool = True) -> List[cq.Workplane]:
    """
    Apply multiple cuts sequentially, returning all resulting pieces.

    Args:
        geometry: CadQuery Workplane to cut
        planes: List of CutPlane objects defining cuts
        validate: Whether to perform validation

    Returns:
        List of all resulting pieces
    """
    if not planes:
        return [geometry]

    pieces = [geometry]

    for i, plane in enumerate(planes):
        new_pieces = []

        for piece in pieces:
            try:
                # Check if plane intersects this piece
                intersects, _ = plane_intersects_geometry(piece, plane)

                if intersects:
                    result = cut_with_plane(piece, plane, validate=validate)
                    new_pieces.append(result.piece_negative)
                    new_pieces.append(result.piece_positive)
                else:
                    # Plane doesn't intersect, keep piece as-is
                    new_pieces.append(piece)

            except ValueError as e:
                # Cut failed, keep original piece
                logger.warning(f"Cut {i+1} failed for piece: {str(e)}")
                new_pieces.append(piece)

        pieces = new_pieces

    return pieces


def classify_pieces(pieces: List[cq.Workplane],
                    part_boxes: List[Tuple[float, float, float, float, float, float]],
                    tolerance: float = Tolerances.LENGTH) -> Dict[str, List[cq.Workplane]]:
    """
    Classify resulting pieces as parts or scrap.

    Args:
        pieces: List of geometry pieces
        part_boxes: List of part bounding boxes (xmin, ymin, zmin, xmax, ymax, zmax)
        tolerance: Position tolerance

    Returns:
        Dict with 'parts' and 'scrap' lists
    """
    parts = []
    scrap = []

    for piece in pieces:
        bb = piece.val().BoundingBox()
        piece_center = (
            (bb.xmin + bb.xmax) / 2,
            (bb.ymin + bb.ymax) / 2,
            (bb.zmin + bb.zmax) / 2
        )

        is_part = False
        for box in part_boxes:
            xmin, ymin, zmin, xmax, ymax, zmax = box
            if (xmin - tolerance <= piece_center[0] <= xmax + tolerance and
                ymin - tolerance <= piece_center[1] <= ymax + tolerance and
                zmin - tolerance <= piece_center[2] <= zmax + tolerance):
                is_part = True
                break

        if is_part:
            parts.append(piece)
        else:
            scrap.append(piece)

    return {"parts": parts, "scrap": scrap}
