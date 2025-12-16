from collections import defaultdict
def pre_z_edges(end_coordinates):
    z_edges = []
    z_start = 0
    for col in end_coordinates.keys():
        for row in end_coordinates[col].keys():
            x,y,z = end_coordinates[col][row]
            if row > 0:
                x_pre, y_pre,z_pre = end_coordinates[col][row-1]
                if x != x_pre:
                    z_edges.append([[x_pre,y_pre,z_start],[x_pre,y_pre,z]])
        z_edges.append([[x,y,z_start],[x,y,z]])
        z_start = z

    return z_edges


def connect_lines_same_x(lines):
    groups = defaultdict(list)

    # Group by common X
    for p1, p2 in lines:
        x = p1[0]
        groups[x].append([p1, p2])

    merged = []

    for x, segs in groups.items():
        # Sort segments by Z
        segs_sorted = sorted(segs, key=lambda s: min(s[0][2], s[1][2]))

        # Start chain
        start = segs_sorted[0][0]
        end   = segs_sorted[0][1]

        for p1, p2 in segs_sorted[1:]:
            if p1 == end:      # correctly chained
                end = p2       # extend chain
            elif p2 == end:    # reversed but touching
                end = p1
            else:
                # Not touching → new chain
                merged.append([start, end])
                start, end = p1, p2

        merged.append([start, end])

    return merged

#---------------------------------------------------------------------------------------------------



def pre_x_edges(end_coordinates):
    x_edges = []
    for col in end_coordinates.keys(): 
        x_start = 0
        for row in end_coordinates[col].keys():
            x,y,z = end_coordinates[col][row]
            if row > 0:
                x_pre, y_pre,z_pre = end_coordinates[col][row-1]
                if x != x_pre:
                    x_edges.append([[x_pre,y_pre,z],[x,y_pre,z]])
                    y_start = y_pre
                    
        x_edges.append([[x_start,y,z],[x,y,z]])      

    return x_edges


def group_by_common_y(edges):
    groups = {}
    for p1, p2 in edges:
        y_val = p1[1]  # both points have same x
        y_val = round(y_val, 1)  # avoid floating point noise
        if y_val not in groups:
            groups[y_val] = []
        groups[y_val].append([p1, p2])
    return groups


def subtract_x_spans(low_line, high_line):
    (x1, y1, z1), (x2, y2, z2) = low_line
    low_min, low_max = sorted([x1, x2])

    (xh1, yh1, zh1), (xh2, yh2, zh2) = high_line
    high_min, high_max = sorted([xh1, xh2])

    # CASE 1 — lower completely left of higher
    if low_max <= high_min:
        return [low_line]

    # CASE 2 — lower completely right of higher
    if low_min >= high_max:
        return [low_line]

    segments = []

    # Left remaining part: before high_min
    if low_min < high_min:
        segments.append([[low_min, y1, z1], [high_min, y1, z1]])

    # Right remaining part: after high_max
    if low_max > high_max:
        segments.append([[high_max, y1, z1], [low_max, y1, z1]])

    return segments
def process_groups_yxz(groups):
    output = []

    for y, lines in groups.items():

        # highest Z line first
        highest = max(lines, key=lambda L: max(L[0][2], L[1][2]))
        output.append(highest)

        # subtract X-span for lower Z lines
        for line in lines:
            if line == highest:
                continue
            pieces = subtract_x_spans(line, highest)
            output.extend(pieces)

    return output


#-------------------------------------------------------------------------------------------------------



def pre_y_edges(end_coordinates):
    y_edges = []
    for col in end_coordinates.keys(): 
        y_start = 0
        for row in end_coordinates[col].keys():
    
            x,y,z = end_coordinates[col][row]
            if row > 0:
                x_pre, y_pre,z_pre = end_coordinates[col][row-1]
                if x != x_pre:
                    y_edges.append([[x_pre,y_start,z],[x_pre,y_pre,z]])
                    y_start = y_pre
                
        y_edges.append([[x,y_start,z],[x,y,z]])      
    return y_edges
        
def group_by_common_x(edges):
    groups = {}
    for p1, p2 in edges:
        x_val = p1[0]  # both points have same x
        x_val = round(x_val, 4)  # avoid floating point noise
        if x_val not in groups:
            groups[x_val] = []
        groups[x_val].append([p1, p2])
    return groups

# -------------------------------------
# FUNCTION 1: Remove overlapping Y-span
# -------------------------------------
def subtract_spans(low_line, high_line):
    (x1, y1, z1), (x2, y2, z2) = low_line
    low_min, low_max = sorted([y1, y2])

    (xh1, yh1, zh1), (xh2, yh2, zh2) = high_line
    high_min, high_max = sorted([yh1, yh2])

    # CASE 1 — lower completely below higher
    if low_max <= high_min:
        return [low_line]

    # CASE 2 — lower completely above higher
    if low_min >= high_max:
        return [low_line]

    # CASE 3 — subtract overlapping region
    segments = []

    # Left remaining part: before high_min
    if low_min < high_min:
        segments.append([[x1, low_min, z1], [x1, high_min, z1]])

    # Right remaining part: after high_max
    if low_max > high_max:
        segments.append([[x1, high_max, z1], [x1, low_max, z1]])

    return segments


# -------------------------------------
# FUNCTION 2: Process all groups
# -------------------------------------
def process_groups(groups):
    output = []

    for x, lines in groups.items():

        # Pick highest-Z line
        highest = max(lines, key=lambda L: max(L[0][2], L[1][2]))
        output.append(highest)

        # Process lower-Z lines
        for line in lines:
            if line == highest:
                continue

            pieces = subtract_spans(line, highest)
            output.extend(pieces)

    return output

'''def y_edges_process(y_edges):
    best_by_y = {}
    filtered = []
    
    for line in y_edges:
        p1, p2 = line
    
        # Key: y-values of the line
        y_key = (p1[1], p2[1])
    
        # Compare using max x-value in the line
        max_x = max(p1[0], p2[0])
    
        if y_key not in best_by_y:
            best_by_y[y_key] = (max_x, line)
        else:
            # Keep the line with larger x value
            if max_x > best_by_y[y_key][0]:
                best_by_y[y_key] = (max_x, line)
    
    # Collect final lines
    filtered = [item[1] for item in best_by_y.values()]
    return '''

def y_edges_process(y_edges):
    best_by_xy = {}

    for line in y_edges:
        p1, p2 = line

        # Create sorted keys for x and y so order of points does not matter
        x_key = tuple(sorted([p1[0], p2[0]]))
        y_key = tuple(sorted([p1[1], p2[1]]))

        # Combined unique key
        key = (x_key, y_key)

        # Compare using max Z value of the line
        max_z = max(p1[2], p2[2])

        # Store the line with highest Z for this (x,y) key
        if key not in best_by_xy or max_z > best_by_xy[key][0]:
            best_by_xy[key] = (max_z, line)

    # Return only the selected lines
    return [item[1] for item in best_by_xy.values()]


def x_edges_process(x_edges):
    best_by_xy = {}
    
    for line in x_edges:
        p1, p2 = line

        # Extract coordinates
        x_key = tuple(sorted([p1[0], p2[0]]))   # X values -> sorted
        y_key = tuple(sorted([p1[1], p2[1]]))   # Y values -> sorted

        # Combined key
        key = (x_key, y_key)

        # Compare using max Z value
        max_z = max(p1[2], p2[2])

        # Store only the line with highest Z for each XY key
        if key not in best_by_xy or max_z > best_by_xy[key][0]:
            best_by_xy[key] = (max_z, line)

    # Return only lines
    return [item[1] for item in best_by_xy.values()]





#----------------------------------------------------------------------------------------------
# Type 2 scrap separation
def get_type(edges):
    '''
    There are total 4 types of formation
    '''
    if len(edges['x_edges']) == 1 and len(edges['y_edges']) == 1 and len(edges['z_edges']) == 1:
        #print('this is type1')
        return 1
    elif len(edges['x_edges']) == 2 and len(edges['y_edges']) == 2 and len(edges['z_edges']) == 2:
        p1, p2 = edges['x_edges'][0]
        p3, p4 =  edges['x_edges'][1]
        if p1[2] == p3[2]:
            #print('This is type2')
            return 2
        else:
            #print('This is type3')
            return 3
    elif len(edges['x_edges']) == 3 and len(edges['y_edges']) == 3 and len(edges['z_edges']) == 3:
        #print('This is type4')
        return 4
    else:
        from fill import draw
        num = len(edges['x_edges'])+ len(edges['y_edges'])+ len(edges['z_edges'])
        #print('x_edges: ',len(edges['x_edges']), ', y_edges: ',len(edges['y_edges']), ', z_edges: ',len(edges['z_edges']))
        raise Exception('The edges must be inside 4types. Here number of edges are', num)