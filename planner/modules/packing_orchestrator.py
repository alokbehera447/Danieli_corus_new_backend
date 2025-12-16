# planner/modules/packing_orchestrator.py
import math
import numpy as np
import os
import sys

# Add the parent directory to sys.path to access trapezoidal_packing
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
trapezoidal_dir = os.path.join(project_root, 'trapezoidal_packing')
sys.path.insert(0, trapezoidal_dir)

# Import your existing functions from trapezoidal_packing
try:
    from edges import *
    from scrap import *
    from .fill import fill_the_box, draw, place_box, get_scrap_vol
    from .fill import rotate as fill_rotate
    # optional: silence or log import success
    # print(f"[packing_orchestrator] Successfully imported trapezoidal_packing modules")
except ImportError as e:
    # If import fails, provide placeholders so module still loads (useful for testing)
    # print(f"[packing_orchestrator] ERROR: Could not import trapezoidal packing modules: {e}")

    def fill_the_box(*args, **kwargs):
        raise ImportError("fill_the_box not available")

    def get_scrap_vol(*args, **kwargs):
        raise ImportError("get_scrap_vol not available")

    def draw(*args, **kwargs):
        print("Draw function not available")

    def place_box(*args, **kwargs):
        return []

    fill_rotate = None


class Prisms:
    def __init__(self, code, size, quantity, roundingoff=2):
        """
        Follow notebook semantics:
        size -> if len==4: [bottom_length, top_length, width, height]
                if len==3: intended as [bottom_length, width, height] and top_length = bottom_length
        """
        self.code = code
        self.quantity = quantity
        self.prism_left = quantity
        self.roundingoff = roundingoff

        if len(size) == 4:
            self.size = size
        elif len(size) == 3:
            # Notebook intended to expand to 4 values.
            # Fixing the buggy `.extend()` behaviour by explicitly creating the list:
            # bottom_length, top_length (same as bottom), width, height
            self.size = [size[0], size[0], size[1], size[2]]
        else:
            # match notebook message
            raise ValueError('dimension of the prism is more then 4. But the code is written only for trapezodial prisms')

        self.bottom_length = self.size[0]
        self.top_length = self.size[1]
        self.width = self.size[2]
        self.height = self.size[3]
        self.angle = self.angle_from_height_length()

        # volume for trapezoidal prism
        self.volume = 0.5 * (self.bottom_length + self.top_length) * self.width * self.height

    def angle_from_height_length(self):
        height = self.height
        length = (self.bottom_length - self.top_length) / 2
        # protect against division by zero
        if height == 0:
            angle_deg = 0.0
        else:
            angle_rad = math.atan(length / height)
            angle_deg = math.degrees(angle_rad)
        return np.round(angle_deg, self.roundingoff)

    def update_prism_left(self, used_quantity):
        self.prism_left = self.prism_left - used_quantity

    def get_volume(self):
        return self.volume

    def __repr__(self):
        return f"Prism({self.code}, bottom={self.bottom_length}, top={self.top_length}, w={self.width}, h={self.height}, left={self.prism_left})"


class Rotation:
    def __init__(self, axis_order=['x', 'y'], pivot=(0, 0, 0), roundingoff=2):
        self.roundingoff = roundingoff
        self.axis_order = axis_order
        self.pivot = pivot

    def get_new_lwh(self, size):
        def get_lwh(axis, size):
            l, w, h = size
            if axis == 'z':
                l, w, h = w, l, h
            elif axis == 'y':
                l, w, h = h, w, l
            elif axis == 'x':
                l, w, h = l, h, w
            return [l, w, h]

        for axis in self.axis_order:
            size = get_lwh(axis, size)
        return size

    def rotate(self, points, angle_deg, axis='z', pivot=(0, 0, 0)):
        pts = np.array(points, dtype=float)
        if pts.ndim == 1:
            pts = pts.reshape(1, 3)

        angle = np.radians(angle_deg)
        px, py, pz = pivot
        shifted = pts - np.array([px, py, pz])

        if axis == 'x':
            R = np.array([
                [1, 0, 0],
                [0, np.cos(angle), -np.sin(angle)],
                [0, np.sin(angle), np.cos(angle)]
            ])
        elif axis == 'y':
            R = np.array([
                [np.cos(angle), 0, np.sin(angle)],
                [0, 1, 0],
                [-np.sin(angle), 0, np.cos(angle)]
            ])
        elif axis == 'z':
            R = np.array([
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1]
            ])
        else:
            raise ValueError("axis must be one of 'x', 'y', 'z'")

        rotated = shifted @ R.T
        result = rotated + np.array([px, py, pz])
        return np.round(result, self.roundingoff)

    def rotate_in_order(self, points):
        new_pts = points
        for axis in self.axis_order:
            new_pts = self.rotate(new_pts, 90, axis, self.pivot)
        return new_pts

    def rotate_in_reverse_order(self, points):
        new_pts = points
        for axis in reversed(self.axis_order):
            new_pts = self.rotate(new_pts, -90, axis, self.pivot)
        return new_pts

    def get_starting_co_and_size(self, pts, after_rotation=True):
        """
        Return starting coordinate and size (l,w,h) of the rotated cuboid described by pts.
        pts: array-like list of 8 corner coordinates of the cuboid.
        """
        if after_rotation:
            pts = self.rotate_in_order(pts)
        pts = np.array(pts, dtype=float)

        xs = pts[:, 0]; ys = pts[:, 1]; zs = pts[:, 2]
        xmin, xmax = xs.min(), xs.max()
        ymin, ymax = ys.min(), ys.max()
        zmin, zmax = zs.min(), zs.max()

        starting_point = [xmin, ymin, zmin]
        size = [xmax - xmin, ymax - ymin, zmax - zmin]
        return starting_point, size


class Block:
    def __init__(self, unique_code, size, start_coord=[0, 0, 0]):
        self.unique_code = unique_code
        self.size = size
        self.start_coord = start_coord
        self.volume = size[0] * size[1] * size[2]
        self.place_box()

        self.scraps = []   # store Scrap objects
        self.prism_details = []
        self.all_prisms_coordinates = []

    def add_scrap(self, scrap_obj):
        scrap_obj.parent_block = self  # assign parent
        self.scraps.append(scrap_obj)

    def add_prisms_coordinates(self, prism, coordinates):
        prism_detail = {'prism': prism, 'coordinates': coordinates}
        self.prism_details.append(prism_detail)
        self.all_prisms_coordinates.extend(coordinates)

    def get_efficiency(self):
        prism_volume = 0
        for prism_detail in self.prism_details:
            prism_volume += prism_detail['prism'].volume * len(prism_detail['coordinates'])
        eff = (prism_volume / self.volume) * 100 if self.volume > 0 else 0
        return eff

    def can_fit_with_rotation(self, prism, rotation_axis=[[], ['z'], ['z', 'x'], ['z', 'y'], ['x'], ['y']]):
        rotation_axis_new = []
        if prism.volume > self.volume:
            return False, rotation_axis_new
        for axis_order in rotation_axis:
            if len(axis_order) > 0:
                rot = Rotation(axis_order=axis_order, pivot=self.start_coord)
                size = rot.get_new_lwh(self.size)
            else:
                size = self.size
            # Keep the same strict < comparisons as notebook
            if (prism.bottom_length < size[0] and
                prism.width < size[1] and
                prism.height < size[2]):
                rotation_axis_new.append(axis_order)
        if len(rotation_axis_new) > 0:
            return True, rotation_axis_new
        else:
            return False, rotation_axis_new

    def place_box(self):
        length, width, height = self.size[0], self.size[1], self.size[2]
        [x, y, z] = self.start_coord
        self.box_coordinate = [
            [x, y, z], [x + length, y, z], [x + length, y + width, z], [x, y + width, z],
            [x, y, z + height], [x + length, y, z + height], [x + length, y + width, z + height], [x, y + width, z + height]
        ]

    def draw_it(self, only_scrap=False):
        """
        Draw the block similar to Jupyter notebook draw_it:
        - if only_scrap: draw only scrap volumes
        - else: draw prisms coordinates (filling)
        Uses `draw` imported from fill module (if available).
        """
        big_block_coordinate = self.box_coordinate
        co_ordinates_list = self.all_prisms_coordinates
        scrap_volumes = []
        for scrap in self.scraps:
            scrap_volumes.append(scrap.box_coordinate)
        if only_scrap:
            draw(big_block_coordinate, co_ordinates_list=[], x_edges=[], y_edges=[], z_edges=[],
                 planes={"xy_planes": [], "zx_planes": [], "yz_planes": []}, scrap_volumes=scrap_volumes)
        else:
            draw(big_block_coordinate, co_ordinates_list, x_edges=[], y_edges=[], z_edges=[],
                 planes={"xy_planes": [], "zx_planes": [], "yz_planes": []}, scrap_volumes=[])


class Scrap(Block):
    def __init__(self, unique_code, size, start_coord):
        super().__init__(unique_code, size, start_coord)
        self.parent_block = None    # this will be assigned when added

    def delete_scrap(self):
        if self.parent_block and self in self.parent_block.scraps:
            self.parent_block.scraps.remove(self)


class People_helper:
    def __init__(self, buffer=2, parent_block_sizes=[[1870, 800, 350]]):
        self.all_scrap = []
        self.scrap_count = 0
        self.big_block_count = 0
        self.all_big_blocks = []
        self.rotation_axis = [[], ['z'], ['z', 'x'], ['z', 'y'], ['x'], ['y']]

        self.buffer = buffer
        # self.Bigblock_size = Bigblock_size
        self.parent_block_sizes = parent_block_sizes

        self.all_scrap_temp = []

    def add_one_big_block(self, size, code='B'):
        self.big_block_count += 1
        origine = [0, 0, 0]
        starting_point = origine
        block = Block(code + str(self.big_block_count), size, start_coord=starting_point)
        self.all_big_blocks.append(block)

        return block

    def get_a_temp_block(self, size, code='Temp'):
        origine = [0, 0, 0]
        starting_point = origine
        block = Block(code, size, start_coord=starting_point)

        return block

    def try_to_pack_inside_all_scrap(self, prism, all_scrap=None):
        """
        Follow notebook behavior: iterate all scrap and attempt fill, but DO NOT return early.
        (Notebook version loops and calls fill_the_prism_optimally but did not return anything.)
        We will keep same behavior (and if required elsewhere, results are stored in helper state).
        """
        if all_scrap is None:
            all_scrap = self.all_scrap
        for scrap in all_scrap[:]:
            # notebook simply called fill_the_prism_optimally(prism, scrap)
            # but did not use return; we'll call it to allow it to update helper state
            try:
                self.fill_the_prism_optimally(prism, scrap)
            except Exception:
                # keep consistent with notebook (which swallowed exceptions sometimes)
                # but do not crash the orchestrator
                continue

    def check_which_block_to_add(self, prism):
        size_list_global = []
        prism_count_list_global = []
        axis_order_list_global = []
        for parent_size in self.parent_block_sizes:
            size_list = []
            prism_count_list = []
            block = self.get_a_temp_block(parent_size, code='Temp')
            cond, rotation_axis_new = block.can_fit_with_rotation(prism, self.rotation_axis)
            if not cond:
                continue
            for axis_order in rotation_axis_new:
                if len(axis_order) != 0:
                    rot = Rotation(axis_order=axis_order, pivot=block.start_coord)
                    size = rot.get_new_lwh(block.size)
                else:
                    size = block.size
                size_list.append(size)

                # pack it inside the block (call to fill_the_box as in notebook)
                co_ordinates_list, big_block_coordinate, end_coordinates, prism_count = fill_the_box(prism,
                                                                                                     Block_size=size,
                                                                                                     starting_co=block.start_coord,
                                                                                                     buffer=self.buffer)
                prism_count_list.append(prism_count)

            if len(prism_count_list) == 0:
                continue

            prism_count_max = max(prism_count_list)
            max_index = prism_count_list.index(prism_count_max)
            axis_order_max = rotation_axis_new[max_index]
            size_max = size_list[max_index]

            prism_count_list_global.append(prism_count_max)
            size_list_global.append(size_max)
            axis_order_list_global.append(axis_order_max)

        if len(prism_count_list_global) == 0:
            return None

        prism_count_global_max = max(prism_count_list_global)
        max_index = prism_count_list_global.index(prism_count_global_max)
        axis_order_max = axis_order_list_global[max_index]
        size_global_max = size_list_global[max_index]

        # return the parent block size selected (same as notebook)
        return self.parent_block_sizes[max_index]

    def fill_the_prism_optimally(self, prism, scrap):
        """
        Major function — follows notebook logic closely and returns:
        co_ordinates_list, big_block_coordinate, scrap_volumes, prism_count, scrap_blocks_list_temp
        """
        self.all_scrap_temp = []
        size_list = []
        prism_count_list = []
        cond, rotation_axis_new = scrap.can_fit_with_rotation(prism, self.rotation_axis)
        if not cond:
            # Notebook printed a message; here we simply return Nones
            return None, None, None, None, None

        for axis_order in rotation_axis_new:
            if len(axis_order) != 0:
                rot = Rotation(axis_order=axis_order, pivot=scrap.start_coord)
                size = rot.get_new_lwh(scrap.size)
            else:
                size = scrap.size
            size_list.append(size)

            # pack it inside the block
            co_ordinates_list, big_block_coordinate, end_coordinates, prism_count = fill_the_box(prism,
                                                                                                 Block_size=size,
                                                                                                 starting_co=scrap.start_coord,
                                                                                                 buffer=self.buffer)
            prism_count_list.append(prism_count)

        if len(prism_count_list) == 0:
            return None, None, None, None, None

        prism_count_max = max(prism_count_list)
        max_index = prism_count_list.index(prism_count_max)
        axis_order_max = rotation_axis_new[max_index]
        size_max = size_list[max_index]

        if prism_count_max == 0:
            return None, None, None, None, None

        # Compute efficiency if needed (not strictly required but matches notebook flow)
        eff = (prism.volume * prism_count_max / scrap.volume) * 100 if scrap.volume > 0 else 0

        # get the rotation tools
        if len(axis_order_max) != 0:
            rot = Rotation(axis_order=axis_order_max, pivot=scrap.start_coord)
            new_starting_point, size = rot.get_starting_co_and_size(scrap.box_coordinate)

            # after rotation fill the block
            co_ordinates_list, big_block_coordinate, end_coordinates, prism_count = fill_the_box(prism,
                                                                                                 Block_size=size_max,
                                                                                                 starting_co=new_starting_point,
                                                                                                 buffer=self.buffer)
            # rotated case: notebook used 3-argument signature for get_scrap_vol
            scrap_volumes, scrap_Boxes_new = get_scrap_vol(end_coordinates, size_max, new_starting_point)

            # Now rotate in the reverse direction
            co_ordinates_list = rot.rotate_in_reverse_order(co_ordinates_list).tolist()
            big_block_coordinate = rot.rotate_in_reverse_order(big_block_coordinate)
            scrap_volumes = rot.rotate_in_reverse_order(scrap_volumes)
        else:
            co_ordinates_list, big_block_coordinate, end_coordinates, prism_count = fill_the_box(prism,
                                                                                                 Block_size=size_max,
                                                                                                 starting_co=scrap.start_coord,
                                                                                                 buffer=self.buffer)
            # unrotated case: notebook used 4-argument signature for get_scrap_vol
            scrap_volumes, scrap_Boxes_new = get_scrap_vol(end_coordinates, size_max, scrap.start_coord, co_ordinates_list)

        assert prism_count > 0

        if type(scrap) is Block:
            block = scrap

            prism.update_prism_left(prism_count)
            block.add_prisms_coordinates(prism, co_ordinates_list)
            scrap_blocks_list_temp = self.add_update_scrap_list(block, scrap_volumes)

            self.all_scrap_temp = scrap_blocks_list_temp

        elif type(scrap) is Scrap:
            block = scrap.parent_block

            prism.update_prism_left(prism_count)
            block.add_prisms_coordinates(prism, co_ordinates_list)
            scrap_blocks_list_temp = self.add_update_scrap_list(block, scrap_volumes)

            self.delete_scrap(scrap)
        else:
            raise Exception('Scrap element inside fill_the_prism_optimally function must be a scrap or block')

        return co_ordinates_list, big_block_coordinate, scrap_volumes, prism_count, scrap_blocks_list_temp

    def is_small_size(self, size):
        volume = size[0] * size[1] * size[2]
        is_small_volume = volume < 10
        is_small_length = size[0] < 2 or size[1] < 2 or size[2] < 2

        return is_small_volume or is_small_length

    def add_update_scrap_list(self, block, scrap_volumes):
        scrap_blocks_list_temp = []
        for scrap_vol in scrap_volumes:
            self.scrap_count += 1
            rot = Rotation()
            scrap_starting_point, scrap_size = rot.get_starting_co_and_size(scrap_vol, after_rotation=False)

            if self.is_small_size(scrap_size):
                continue
            # add scrap element
            s = Scrap('s' + str(self.scrap_count), scrap_size, scrap_starting_point)
            block.add_scrap(s)

            # add scrap to emp scrap list.
            scrap_blocks_list_temp.append(s)

        # sort low volume first inside the scrap elements (same as notebook)
        scrap_blocks_list_temp = sorted(scrap_blocks_list_temp, key=lambda s: s.volume)

        # add temp scrap temp list to global scrap list
        self.all_scrap.extend(scrap_blocks_list_temp)

        return scrap_blocks_list_temp

    def delete_scrap(self, scrap):
        # remove it from global list
        if scrap in self.all_scrap:
            self.all_scrap.remove(scrap)
        scrap.delete_scrap()


def run_final_code(all_prisms, buffer=2, parent_block_sizes=[[1870, 800, 350], [2000, 800, 400]]):
    """
    Reproduce notebook flow: helper tries to pack prisms using existing scraps,
    creates new blocks as needed and fills them until all prisms consumed.
    """
    helper = People_helper(buffer, parent_block_sizes)
    for prism in all_prisms[:]:
        helper.try_to_pack_inside_all_scrap(prism)
        while True:
            if prism.prism_left == 0:
                break
            # create a block
            size = helper.check_which_block_to_add(prism)
            if size is None:
                # No block can contain this prism (matches notebook handling)
                break
            b = helper.add_one_big_block(size)
            co_ordinates_list, big_block_coordinate, scrap_volumes, prism_count, scrap_blocks_list_temp = helper.fill_the_prism_optimally(
                prism, b)
            helper.try_to_pack_inside_all_scrap(prism, scrap_blocks_list_temp)
    return helper


def get_block_details(helper):
    block_details = {
        "Total_number_of_blocks": len(helper.all_big_blocks),
        "Total_eff": 0,
        "blocks": [],
        "scraps": []
    }

    total_eff_sum = 0

    # ---- PROCESS ALL BLOCKS ----
    for block in helper.all_big_blocks:
        block_eff = round(block.get_efficiency(), 2)
        total_eff_sum += block_eff

        # Count prisms inside this block grouped by prism code
        prism_count_dict = {}   # { 'G14': 16, 'G15': 2 }

        for entry in block.prism_details:
            prism = entry['prism']
            count = len(entry['coordinates'])

            if prism.code not in prism_count_dict:
                prism_count_dict[prism.code] = 0
            prism_count_dict[prism.code] += count

        # Convert to required list format
        prism_list = [
            {"code": code, "number": num}
            for code, num in prism_count_dict.items()
        ]

        block_details["blocks"].append({
            "code": block.unique_code,
            "eff": block_eff,
            "prisms": prism_list
        })

    # ---- COMPUTE TOTAL EFFICIENCY ----
    if len(helper.all_big_blocks) > 0:
        block_details["Total_eff"] = round(total_eff_sum / len(helper.all_big_blocks), 2)
    else:
        block_details["Total_eff"] = 0

    # ---- ADD ALL SCRAPS ----
    for scrap in helper.all_scrap:
        block_details["scraps"].append({
            "code": scrap.unique_code,
            "size": scrap.size,
            "volume": scrap.volume
        })

    return block_details
