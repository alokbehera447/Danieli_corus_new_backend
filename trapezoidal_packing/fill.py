def place_box(starting_co, length, width, height):
    x, y, z = starting_co
    box_coordinate = [[x,y,z],[x+length,y,z], [x+length, y+width, z], [x, y+width, z], 
                            [x, y, z+height], [x+ length, y, z+height], [x+length, y+width, z+height], [x, y+ width, z+height]]
    return box_coordinate
def place_prism_flip(prism, buffer, coordinate=[0,0,0],flip = False):
    bottom_length = prism.bottom_length
    top_length = prism.top_length
    assert bottom_length >= top_length
    width = prism.width
    height = prism.height
    extra_length = (bottom_length - top_length)/2

    if not flip:
        x, y, z = coordinate
        co_ordinate1 = [x, y,z]
        x1, y1, z1 = co_ordinate1
        co_ordinate2 = [x1 + bottom_length, y1,z1]
        co_ordinate3 = [x1 + bottom_length, y1+ width, z1]
        co_ordinate4 = [x1, y1+ width, z1]
        co_ordinate5 = [x1+extra_length, y1, z1+height]
        co_ordinate6 = [x1+extra_length + top_length, y1, z1+height]
        co_ordinate7 = [x1+extra_length + top_length, y1+ width, z1+height]
        co_ordinate8 = [x1+extra_length,y1+ width, z1+height]

        x1max = x1 + bottom_length
        y1max = y1+ width
        z1max = z1 + height

    else:
        x, y, z = coordinate
        co_ordinate1 = [x, y,z]
        x1, y1, z1 = co_ordinate1
        co_ordinate2 = [x1 + top_length, y1,z1]
        co_ordinate3 = [x1 + top_length, y1+ width, z1]
        co_ordinate4 = [x1, y1+ width, z1]
        co_ordinate5 = [x1-extra_length, y1, z1+height]
        co_ordinate6 = [x1-extra_length + bottom_length, y1, z1+height]
        co_ordinate7 = [x1-extra_length + bottom_length, y1+ width, z1+height]
        co_ordinate8 = [x1-extra_length,y1+ width, z1+height]

        x1max = x1-extra_length + bottom_length
        y1max = y1+ width
        z1max = z1 + height

    return [co_ordinate1, co_ordinate2, co_ordinate3, co_ordinate4, co_ordinate5, co_ordinate6, co_ordinate7, co_ordinate8], x1max, y1max, z1max

def fill_the_box(trapezoid_prisms, Block_size =  [2000,800, 400],starting_co= [0,0,0], buffer = 2):

    x_min,y_min,z_min = starting_co
    x,y,z = x_min, y_min, z_min
    length = Block_size[0]
    width = Block_size[1]
    height = Block_size[2]
    x_max, y_max, z_max = x_min + length, y_min+width, z_min+height
    big_block_coordinate = [[x,y,z],[x+length,y,z], [x+length, y+width, z], [x, y+width, z], 
                            [x, y, z+height], [x+ length, y, z+height], [x+length, y+width, z+height], [x, y+ width, z+height]]
    bottomsup = False
    pre_angle = 0
    co_ordinates_list=[]
    buffer = buffer
    prism_count = 0
    is_new_row = True
    is_new_col = True
    new_row_coordinate = [x_min,y_min,z_min]
    new_col_coordinate = [x_min,y_min,z_min]
    #end_coordinates = {col_No:{row_no:[x,y,]}
    end_coordinates = {}
    row_no = 0
    col_no = 0
    
    for _ in range(trapezoid_prisms.prism_left):
        #print('start')
        prism = trapezoid_prisms
        angle= trapezoid_prisms.angle
        bottom_length = trapezoid_prisms.bottom_length
        top_length = trapezoid_prisms.top_length
        assert bottom_length >= top_length
        width = trapezoid_prisms.width
        height = trapezoid_prisms.height
        extra_length = (bottom_length - top_length)/2
        #print(prism_count,"---------------------------------------------")
        
        if not is_new_row and not is_new_col:
            if bottomsup :
                if angle == pre_angle:
                    x, y, z = co_ordinates_list[-1][1] # prvious prism's second coordinate
                    co_ordinates, x1max, y1max, z1max = place_prism_flip(prism,buffer, coordinate=[x+buffer,y,z],flip = False)
                    
                bottomsup = False
            else:
                if angle == pre_angle:
                    x, y, z = co_ordinates_list[-1][1] # prvious prism's second coordinate
                    co_ordinates, x1max, y1max, z1max = place_prism_flip(prism, buffer,coordinate=[x+buffer,y,z],flip = True)
                    
                bottomsup = True
            if x1max > x_max:
                is_new_row = True
            else:
                end_coordinates[col_no-1][row_no] = [x1max, y1max, z1max]

        if is_new_row and not is_new_col:
            row_no += 1
            x, y, z = new_row_coordinate
            co_ordinates, x1max, y1max, z1max = place_prism_flip(prism,buffer, coordinate=[x+buffer,y+buffer,z+buffer],flip = False)
            #print(x1max, y1max, z1max, x, y, z)

            '''if x1max > x_max:
                print('not possible to pack in x direction, May be prism length is more then Block length')
                return co_ordinates_list, big_block_coordinate, end_coordinates'''
    
            if y1max> y_max:
                is_new_row = True 
                is_new_col = True
            else:
                new_row_coordinate = [x,y+width+buffer,z]
                is_new_col = False
                is_new_row = False
                end_coordinates[col_no-1][row_no] = [x1max, y1max, z1max]
                
    
            bottomsup = False
            
                
        if is_new_row and is_new_col:
            x, y, z = new_col_coordinate
            co_ordinates, x1max, y1max, z1max = place_prism_flip(prism, buffer,coordinate=[x+buffer,y+buffer,z+buffer],flip = False)
    
            new_row_coordinate = [x,y+width+buffer,z]
            new_col_coordinate = [x, y,z+height+buffer]
            is_new_col = False
            is_new_row = False

            if x1max > x_max:
                 ##print('not possible to pack in x direction, May be prism length is more then Block length')
                 return None, big_block_coordinate, None, prism_count
            if y1max > y_max or x1max >x_max:
                ##print('not possible to pack the prism in y direction. May be prism width is more then Block width')
                return None, big_block_coordinate, None, prism_count   
            if z1max > z_max:
                #print('end', z1max, z_max)
                break
            row_no = 0

            end_coordinates[col_no] = {}
            end_coordinates[col_no][row_no] = [x1max, y1max, z1max]

            col_no +=1

        #print(end_coordinates)
        pre_angle = angle
        prism_count += 1 
        #co_ordinates = [co_ordinate1, co_ordinate2, co_ordinate3, co_ordinate4, co_ordinate5, co_ordinate6, co_ordinate7, co_ordinate8]
        co_ordinates_list.append(co_ordinates)
    return co_ordinates_list, big_block_coordinate, end_coordinates, prism_count



import plotly.graph_objects as go
import numpy as np

import plotly.io as pio
pio.renderers.default = "browser"


def draw(big_block_coordinate, co_ordinates_list, x_edges=[], y_edges=[], z_edges=[], 
         planes={"xy_planes":[],"zx_planes":[],"yz_planes":[],}, scrap_volumes =[]):

    # Big block (8 corner coordinates)
    big_block = np.array(big_block_coordinate)
    
    prisms = [np.array(p) for p in co_ordinates_list]
    
    # ---------------------------
    # Helper function to draw triangular mesh faces
    # ---------------------------
    
    def prism_mesh(vertices, color='blue', opacity=1.0):
        # Vertices index for each triangular face (12 triangles for a prism)
        faces = np.array([
            [0,1,2], [0,2,3],   # bottom
            [4,5,6], [4,6,7],   # top
            [0,1,5], [0,5,4],   # side 1
            [1,2,6], [1,6,5],   # side 2
            [2,3,7], [2,7,6],   # side 3
            [3,0,4], [3,4,7]    # side 4
        ])
    
        x = vertices[:,0]
        y = vertices[:,1]
        z = vertices[:,2]
    
        i, j, k = faces[:,0], faces[:,1], faces[:,2]
    
        return go.Mesh3d(
            x=x, y=y, z=z,
            i=i, j=j, k=k,
            color=color,
            opacity=opacity,
            flatshading=True
        )
    def draw_line(p1, p2, color="black", width=6):
        return go.Scatter3d(
            x=[p1[0], p2[0]],
            y=[p1[1], p2[1]],
            z=[p1[2], p2[2]],
            mode="lines",
            line=dict(color=color, width=width)
        )

    # ---------------------------
    # Helper: draw YX plane (constant Z)
    # ---------------------------
    def draw_yx_plane(pl, color):
        z = pl['z']
        xs, xe = pl['x_start'], pl['x_end']
        ys, ye = pl['y_start'], pl['y_end']

        vertices = np.array([
            [xs, ys, z],
            [xe, ys, z],
            [xe, ye, z],
            [xs, ye, z]
        ])

        faces = np.array([[0,1,2], [0,2,3]])

        return go.Mesh3d(
            x=vertices[:,0],
            y=vertices[:,1],
            z=vertices[:,2],
            i=faces[:,0],
            j=faces[:,1],
            k=faces[:,2],
            color=color,
            opacity=0.3
        )

     # ---------------------------
    # Helper: draw XZ plane (constant Y)
    # ---------------------------
    def draw_xz_plane(pl, color):
        y = pl['y']
        xs, xe = pl['x_start'], pl['x_end']
        zs, ze = pl['z_start'], pl['z_end']
    
        vertices = np.array([
            [xs, y, zs],
            [xe, y, zs],
            [xe, y, ze],
            [xs, y, ze]
        ])
    
        faces = np.array([[0,1,2], [0,2,3]])
    
        return go.Mesh3d(
            x=vertices[:,0],
            y=vertices[:,1],
            z=vertices[:,2],
            i=faces[:,0],
            j=faces[:,1],
            k=faces[:,2],
            color= color,
            opacity=0.35
        )

    # ---------------------------
    # Helper: draw YZ plane (constant X)
    # ---------------------------
    def draw_yz_plane(pl, color):
        x = pl['x']
        zs, ze = pl['z_start'], pl['z_end']
        ys, ye = pl['y_start'], pl['y_end']

        vertices = np.array([
            [x, ys, zs],
            [x, ye, zs],
            [x, ye, ze],
            [x, ys, ze]
        ])

        faces = np.array([[0,1,2], [0,2,3]])

        return go.Mesh3d(
            x=vertices[:,0],
            y=vertices[:,1],
            z=vertices[:,2],
            i=faces[:,0],
            j=faces[:,1],
            k=faces[:,2],
            color= color,
            opacity=0.3
        )

    # ---------------------------
    # Build Plot
    # ---------------------------
    
    fig = go.Figure()
    
    # Draw big block (transparent)
    fig.add_trace(prism_mesh(big_block, color='lightgray', opacity=0.15))

    # Draw prisms with different colors
    colors = ["red", "green", "blue", "orange", "purple", "cyan"]

    
    for i, scrap_coordinate in enumerate(scrap_volumes):
        color = colors[i % len(colors)]
        fig.add_trace(prism_mesh(np.array(scrap_coordinate), color=color, opacity=0.15))

    '''i =0
    for scrap_coordinate in scrap_volumes:
        color = colors[i]
        i+=1
        fig.add_trace(prism_mesh(np.array(scrap_coordinate), color=color, opacity=0.15))'''
    
    
    
    for idx, p in enumerate(prisms):
        color = colors[idx % len(colors)]
        fig.add_trace(prism_mesh(p, color=color, opacity=0.9))

        # Draw x_edges lines
    for edge in x_edges:
        p1, p2 = edge
        fig.add_trace(draw_line(p1, p2, color="black", width=8))

    for edge in y_edges:
        p1, p2 = edge
        #print(p2)
        fig.add_trace(draw_line(p1, p2, color="red", width=8))

    for edge in z_edges:
        p1, p2 = edge
        fig.add_trace(draw_line(p1, p2, color="green", width=8))

    # ------------- Add YX (Z-constant) planes -------------
   
    for pl in planes['xy_planes']:
        fig.add_trace(draw_yx_plane(pl, color='black'))

    # ------------- Add YZ (X-constant) planes -------------
    for pl in planes['yz_planes']:
        fig.add_trace(draw_yz_plane(pl, color='red'))

  
    for pl in planes['zx_planes']:
        fig.add_trace(draw_xz_plane(pl, color='green'))
    
   
    
    
    # ---------------------------
    # Layout / Axis settings
    # ---------------------------
    
    fig.update_layout(
        title="3D Interactive Plot: Trapezoidal Prisms Inside Big Block",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode='data'
        ),
        width=900,
        height=700
    )

    
    fig.show()

    
import numpy as np

def rotate(points, angle_deg, axis='z', pivot=(0, 0, 0), roundingoff = 2):
    """
    Rotate 3D point(s) around a chosen axis with an optional pivot.
    Using positive angle (+θ):
    Rotate around X-axis → rotation is counter-clockwise when looking from +X toward the origin
    Rotate around Y-axis → rotation is counter-clockwise when looking from +Y toward the origin
    Rotate around Z-axis → rotation is counter-clockwise when looking from +Z toward the origin

    points : (N,3) array or (3,) single point
    angle_deg : rotation angle in degrees
    axis : 'x', 'y', or 'z'
    pivot : point to rotate around (default = origin)
    """
    pts = np.array(points, dtype=float)
    if pts.ndim == 1:
        pts = pts.reshape(1, 3)

    angle = np.radians(angle_deg)
    px, py, pz = pivot

    # ---- Shift points so pivot becomes origin ----
    shifted = pts - np.array([px, py, pz])

    # ---- Rotation matrix for X, Y, or Z axis ----
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
            [np.sin(angle),  np.cos(angle), 0],
            [0, 0, 1]
        ])
    else:
        raise ValueError("axis must be one of 'x', 'y', 'z'")

    # ---- Apply rotation ----
    rotated = shifted @ R.T

    # ---- Shift back to original pivot ----
    rounded = np.round(rotated + np.array([px, py, pz]), roundingoff)
    return rounded



def get_scrap_vol(end_coordinates, Block_size, st_co= [0,0,0], co_ordinates_list = []):
    from edges import get_type, process_groups, pre_z_edges, connect_lines_same_x, pre_x_edges, process_groups_yxz, group_by_common_y, pre_y_edges, group_by_common_x, y_edges_process, x_edges_process
    from scrap import get_scrap_volume_of_type4, get_scrap_volume_of_type3, get_scrap_volume_of_type2, get_scrap_volume_of_type1
    z_edges = pre_z_edges(end_coordinates)
    z_edges = connect_lines_same_x(z_edges)
    x_edges = pre_x_edges(end_coordinates)
    x_edges = process_groups_yxz(group_by_common_y(x_edges))
    x_edges = x_edges_process(x_edges)
    y_edges = pre_y_edges(end_coordinates)
    groups = group_by_common_x(y_edges)
    y_edges = process_groups(groups)
    y_edges = y_edges_process(y_edges)
    edges = {'x_edges': x_edges, 'y_edges': y_edges, 'z_edges': z_edges}

    num_x = len(edges['x_edges'])
    num_y = len(edges['y_edges'])
    num_z = len(edges['z_edges'])
    num = num_x + num_y + num_z
    num_cond = num == 3 or num == 6 or num == 9
    num_x_cond = num_x == 1 or num_x == 2 or num_x == 3
    num_y_cond = num_y == 1 or num_y == 2 or num_y == 3
    num_z_cond = num_z == 1 or num_z == 2 or num_z == 3
    
    if  not num_x_cond or not num_y_cond or not num_z_cond or not num_cond:
        big_block_coordinate = place_box(st_co, Block_size[0], Block_size[1], Block_size[2])
        draw(big_block_coordinate, co_ordinates_list , x_edges= edges['x_edges'], y_edges= edges['y_edges'], z_edges=edges['z_edges'], 
             planes={"xy_planes":[],"zx_planes":[],"yz_planes":[],}, scrap_volumes =[])
    
    
    #print('This is type: ', get_type(edges))
    t = get_type(edges)
    if t ==1:
        scrap_Boxes= get_scrap_volume_of_type1(Block_size, edges, st_co)
    elif t==2:
        scrap_Boxes= get_scrap_volume_of_type2(Block_size, edges, st_co)
    elif t==3:
        scrap_Boxes= get_scrap_volume_of_type3(Block_size, edges, st_co)
    elif t==4:
        scrap_Boxes= get_scrap_volume_of_type4(Block_size, edges, st_co)
    
    scrap_volumes = []
    scrap_Boxes_new = []
    for box in scrap_Boxes:
        if box['Box_size'][0] * box['Box_size'][1] * box['Box_size'][2] < 0:
                continue
        scrap_Boxes_new.append(box)
        scrap_coordinate = place_box(box['starting_co'], length= box['Box_size'][0], width= box['Box_size'][1], height= box['Box_size'][2])
        scrap_volumes.append(scrap_coordinate)
    return scrap_volumes,  scrap_Boxes_new