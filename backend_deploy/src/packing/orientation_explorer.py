import numpy as np
import cadquery as cq
from typing import List, Tuple, Optional
from dataclasses import dataclass

from .trapezoid_geometry import TrapezoidGeometry


@dataclass
class Orientation:
    index: int
    face: int
    rotation_angle: float
    dims: Tuple[float, float, float]
    volume: float

    def __repr__(self):
        return f"Orientation({self.index}: face={self.face}, rot={self.rotation_angle}°, dims={self.dims})"


class OrientationExplorer:
    """
    Explores all valid orientations of a trapezoidal prism.

    Uses 3 faces × 45 rotation angles (0° to 88° in 2° increments) = 135 orientations.
    Face 0: Default (W1×D base)
    Face 1: Rotated 90° around X (W1×thickness base)
    Face 2: Rotated 90° around Y (thickness×D base)
    """

    ROTATION_INCREMENT = 2
    NUM_ROTATIONS = 45

    def __init__(self, part_geom: TrapezoidGeometry,rotation_inc = 2, can_use_mirrored_pair = False):
        self.part_geom = part_geom
        self.orientations: List[Orientation] = []
        self.ROTATION_INCREMENT = rotation_inc
        self._generate_orientations()

    def _generate_orientations(self):
        idx = 0
        for face in range(3):
            for rot_idx in range(self.NUM_ROTATIONS):
                angle = rot_idx * self.ROTATION_INCREMENT
                dims = self._compute_bounding_dims(face, angle)
                self.orientations.append(Orientation(
                    index=idx,
                    face=face,
                    rotation_angle=angle,
                    dims=dims,
                    volume=self.part_geom.volume
                ))
                idx += 1

    def _compute_bounding_dims(self, face: int, angle: float) -> Tuple[float, float, float]:
        geom = self.part_geom
        pts_2d = geom.get_base_vertices_2d()

        vertices = []
        for x, y in pts_2d:
            vertices.append([x, y, 0])
            vertices.append([x, y, geom.thickness])
        vertices = np.array(vertices)

        if face == 1:
            rot_x = np.array([
                [1, 0, 0],
                [0, 0, -1],
                [0, 1, 0]
            ])
            vertices = vertices @ rot_x.T
        elif face == 2:
            rot_y = np.array([
                [0, 0, 1],
                [0, 1, 0],
                [-1, 0, 0]
            ])
            vertices = vertices @ rot_y.T

        if angle != 0:
            rad = np.radians(angle)
            rot_z = np.array([
                [np.cos(rad), -np.sin(rad), 0],
                [np.sin(rad), np.cos(rad), 0],
                [0, 0, 1]
            ])
            vertices = vertices @ rot_z.T

        mins = vertices.min(axis=0)
        maxs = vertices.max(axis=0)
        dims = tuple(maxs - mins)

        return dims

    def create_oriented_geometry(self, orientation: Orientation) -> cq.Workplane:
        geom = self.part_geom
        pts = geom.get_base_vertices_2d()

        prism = cq.Workplane("XY").polyline(pts).close().extrude(geom.thickness)

        if orientation.face == 1:
            prism = prism.rotate((0, 0, 0), (1, 0, 0), 90)
        elif orientation.face == 2:
            prism = prism.rotate((0, 0, 0), (0, 1, 0), 90)

        if orientation.rotation_angle != 0:
            prism = prism.rotate((0, 0, 0), (0, 0, 1), orientation.rotation_angle)

        bbox = prism.val().BoundingBox()
        prism = prism.translate((-bbox.xmin, -bbox.ymin, -bbox.zmin))

        return prism

    def filter_fitting_orientations(self, stock_dims: Tuple[float, float, float]) -> List[Orientation]:
        sx, sy, sz = stock_dims
        fitting = []
        for ori in self.orientations:
            dx, dy, dz = ori.dims
            if dx <= sx and dy <= sy and dz <= sz:
                # Calculate volume occupied for this orientation
                face_id = ori.face
                fdx = dx - self.part_geom.C * (face_id - 1) * (face_id - 2) / 2
                fdy = dy - self.part_geom.C * face_id * (face_id - 2)
                fdz = dz - self.part_geom.C * face_id * (face_id - 1) / 2

                fnx = int(sx / fdx) if fdx > 0 else 0
                fny = int(sy / fdy) if fdy > 0 else 0
                fnz = int(sz / fdz) if fdz > 0 else 0

                volume_occupied = fnx * fny * fnz * self.part_geom.volume
                fitting.append((ori, volume_occupied))

        # Sort by volume_occupied in descending order
        fitting.sort(key=lambda x: -x[1])

        # Return only the orientations (without volume_occupied)
        return [ori for ori, _ in fitting]

    def get_orientation(self, index: int) -> Optional[Orientation]:
        if 0 <= index < len(self.orientations):
            return self.orientations[index]
        return None

    def count_parts_for_orientation(self, orientation: Orientation,
                                     stock_dims: Tuple[float, float, float],
                                     margin: float = 0.0) -> int:
        sx, sy, sz = stock_dims
        dx, dy, dz = orientation.dims

        if dx > sx or dy > sy or dz > sz:
            return 0

        nx = int(sx // (dx + margin))
        ny = int(sy // (dy + margin))
        nz = int(sz // (dz + margin))

        return nx * ny * nz

    def rank_orientations(self, stock_dims: Tuple[float, float, float],
                          margin: float = 0.0) -> List[Tuple[Orientation, int]]:
        ranked = []
        for ori in self.orientations:
            count = self.count_parts_for_orientation(ori, stock_dims, margin)
            if count > 0:
                ranked.append((ori, count))

        ranked.sort(key=lambda x: -x[1])
        return ranked
