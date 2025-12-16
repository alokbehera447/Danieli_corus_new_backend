def get_scrap_volume_of_type1(Block_size, edges, st_co= [0,0,0]):
    import random
    # randomly chose any two edges
    if len(edges['x_edges']) == 1 and len(edges['y_edges']) == 1 and len(edges['z_edges']) == 1:
        
    
        item = ['x', 'y', 'z']
    
        result = random.sample(item, 2)
        #print(result)
    x_start = st_co[0]
    y_start = st_co[1]
    z_start = st_co[2]
    y_end = Block_size[1] + y_start
    z_end = Block_size[2] + z_start
    x_end = Block_size[0] + x_start
    
    
    planes = {}
    xy_planes= []
    zx_planes = []
    yz_planes = []
    #scrap_volumes = []
    scrap_Boxes = []
    if 'x' in result and 'y' in result:
        #print('xy_plane')
        [p1,p2] = edges['x_edges'][0]
        #xy_planes.append({'z':p1[2], 'x_start': x_start, 'x_end': x_end, 'y_start': 0, "y_end":y_end})
    

       
        x,y,z = x_start, y_start, p1[2]
        x_final, y_final, z_final = x_end, y_end, z_end
        #if (x_final - x)/x > epsillon and (y_final - y)/y > epsillon and (z_final - z)/z > epsillon:
        scrap_Boxes.append({'starting_co': [x,y,z],'Box_size':[x_final - x, y_final -y, z_final -z] })
            
    
        # Construct one more plane either zx or yz
        [p3,p4] = edges['z_edges'][0]
        item = ['zx', 'yz']
        result = random.sample(item, 1)
        if 'zx' in result:
            
            x,y,z = x_start, p3[1], z_start
            x_final, y_final, z_final = x_end, y_end, p1[2]
            scrap_Boxes.append({'starting_co': [x,y,z],'Box_size':[x_final - x, y_final -y, z_final -z] })
          
    
            x,y,z = p3[0], y_start, z_start
            x_final, y_final, z_final = x_end, p3[1], p1[2]
            #scrap_coordinate = place_box([x,y,z], length= x_final - x, width= y_final -y, height= z_final -z)
            scrap_Boxes.append({'starting_co': [x,y,z],'Box_size':[x_final - x, y_final -y, z_final -z] })
            #scrap_volumes.append(scrap_coordinate)
    
    
        if 'yz' in result:
            #print('yz_plane')
            #yz_planes.append({'x':p3[0], 'z_start': z_start, 'z_end': p1[2],'y_start': y_start,  "y_end": y_end })
    
            x,y,z = p3[0], y_start, z_start
            x_final, y_final, z_final = x_end, y_end, p1[2]
            #scrap_coordinate = place_box([x,y,z], length= x_final - x, width= y_final -y, height= z_final -z)
            scrap_Boxes.append({'starting_co': [x,y,z],'Box_size':[x_final - x, y_final -y, z_final -z] })
            #scrap_volumes.append(scrap_coordinate)
    
    
            x,y,z = x_start, p3[1], z_start
            x_final, y_final, z_final = p3[0], y_end, p1[2]
            #scrap_coordinate = place_box([x,y,z], length= x_final - x, width= y_final -y, height= z_final -z)
            scrap_Boxes.append({'starting_co': [x,y,z],'Box_size':[x_final - x, y_final -y, z_final -z] })
            #scrap_volumes.append(scrap_coordinate)
    
    
        
    if 'x' in result and 'z' in result:
        #print('zx_plane')
        [p1,p2] = edges['x_edges'][0]
        #[p3,p4] = edges['z_edges'][0]
        #zx_planes.append({'y':p1[1], 'z_start': z_start, 'z_end': z_end,'x_start': x_start, "x_end":x_end })
    
    
        x,y,z = x_start, p1[1], z_start
        x_final, y_final, z_final = x_end, y_end, z_end
        #print(x_final-x)
        scrap_Boxes.append({'starting_co': [x,y,z],'Box_size':[x_final - x, y_final -y, z_final -z] })
        #scrap_volumes.append(- scrap_coordinate)
    
        [p3,p4] = edges['y_edges'][0]
        item = ['xy', 'yz']
        result = random.sample(item, 1)
        if 'xy' in result:
            #print('xy_plane')
            #xy_planes.append({'z':p3[2], 'x_start': x_start, 'x_end': x_end, 'y_start': y_start, "y_end":p1[1]})
    
    
            
            x,y,z = x_start, y_start, p3[2]
            x_final, y_final, z_final = x_end, p1[1], z_end
            scrap_Boxes.append({'starting_co': [x,y,z],'Box_size':[x_final - x, y_final -y, z_final -z] })
            #scrap_volumes.append(scrap_coordinate)
    
            x,y,z = p3[0], y_start, z_start
            x_final, y_final, z_final = x_end, p1[1], p3[2]
            scrap_Boxes.append({'starting_co': [x,y,z],'Box_size':[x_final - x, y_final -y, z_final -z] })
            #scrap_volumes.append(scrap_coordinate)
    
    
            
        if 'yz' in result:
            #print('yz_plane')
            #yz_planes.append({'x':p3[0], 'z_start': z_start, 'z_end': z_end,'y_start': y_start,  "y_end": p1[1] })
    
    
            x,y,z = x_start, y_start, p3[2]
            x_final, y_final, z_final = p3[0], p1[1], z_end
            #scrap_coordinate = place_box([x,y,z], length= x_final -x, width= y_final -y, height= z_final -z)
            #scrap_volumes.append(scrap_coordinate)
            scrap_Boxes.append({'starting_co': [x,y,z],'Box_size':[x_final - x, y_final -y, z_final -z] })
    
            x,y,z = p3[0], y_start, z_start
            x_final, y_final, z_final = x_end, p1[1], z_end
            #scrap_coordinate = place_box([x,y,z], length= x_final -x, width= y_final -y, height= z_final -z)
            #scrap_volumes.append(scrap_coordinate)
            scrap_Boxes.append({'starting_co': [x,y,z],'Box_size':[x_final - x, y_final -y, z_final -z] })
    
            
        
    if 'y' in result and 'z' in result:
        #print('yz_plane')
    
        [p1,p2] = edges['y_edges'][0]
        #yz_planes.append({'x':p1[0], 'z_start': z_start, 'z_end': z_end,'y_start': y_start,  "y_end": y_end })
    
        x,y,z = p1[0], y_start, z_start
        x_final, y_final, z_final = x_end, y_end, z_end
        #scrap_coordinate = place_box([x,y,z], length= x_final -x, width= y_final -y, height= z_final -z)
        #scrap_volumes.append(scrap_coordinate)
        scrap_Boxes.append({'starting_co': [x,y,z],'Box_size':[x_final - x, y_final -y, z_final -z] })
            
        
        [p3,p4] = edges['x_edges'][0]
        item = ['xy', 'zx']
        result = random.sample(item, 1)
        if 'xy' in result:
            #print('xy_plane')
            #xy_planes.append({'z':p3[2], 'x_start': x_start, 'x_end': p1[0], 'y_start': y_start, "y_end":y_end})
    
    
            x,y,z = x_start, y_start, p3[2]
            x_final, y_final, z_final = p1[0], y_end, z_end
            scrap_Boxes.append({'starting_co': [x,y,z],'Box_size':[x_final - x, y_final -y, z_final -z] })
    
            x,y,z = x_start, p3[1], z_start
            x_final, y_final, z_final = p1[0], y_end, p3[2]
            scrap_Boxes.append({'starting_co': [x,y,z],'Box_size':[x_final - x, y_final -y, z_final -z] })
    
    
            
        if 'zx' in result:
            #print('zx_plane')
            #zx_planes.append({'y':p3[1], 'z_start': z_start, 'z_end': z_end,'x_start': x_start, "x_end":p1[0] })
    
    
            x,y,z = x_start, y_start, p3[2]
            x_final, y_final, z_final = p1[0], p3[1], z_end
            #scrap_coordinate = place_box([x,y,z], length= x_final -x, width= y_final -y, height= z_final -z)
            #scrap_volumes.append(scrap_coordinate)
            scrap_Boxes.append({'starting_co': [x,y,z],'Box_size':[x_final - x, y_final -y, z_final -z] })
    
            x,y,z = x_start, p3[1], z_start
            x_final, y_final, z_final = p1[0], y_end, z_end
            #scrap_coordinate = place_box([x,y,z], length= x_final -x, width= y_final -y, height= z_final -z)
            #scrap_volumes.append(scrap_coordinate)
            scrap_Boxes.append({'starting_co': [x,y,z],'Box_size':[x_final - x, y_final -y, z_final -z] })
            
        
    
    #planes['xy_planes'] = xy_planes
    #planes['zx_planes'] = zx_planes
    #planes['yz_planes'] = yz_planes

    return scrap_Boxes




def get_scrap_volume_of_type2(Block_size, edges, st_co= [0,0,0]):
    import random
    x_start = st_co[0]
    y_start = st_co[1]
    z_start = st_co[2]

    y_end = Block_size[1] + y_start
    z_end = Block_size[2] + z_start
    x_end = Block_size[0] + x_start

    scrap_Boxes = []

    z_edge1 = edges['z_edges'][0]
    z_edge2 = edges['z_edges'][1]
    y_edge1 = edges['y_edges'][0]
    y_edge2 = edges['y_edges'][1]
    x_edge1 = edges['x_edges'][0]
    x_edge2 = edges['x_edges'][1]
    

    p1, p2 = edges['x_edges'][0]
    z_boundry = p1[2]

    edges_less_y = {}
    edges_more_y = {}
    p1,p2 = y_edge1
    p3,p4 = y_edge2
    if p1[0]> p3[0]:
        edges_less_y['y_edges'] = [y_edge1]
        edges_more_y['y_edges'] = [y_edge2]
    else:
        edges_less_y['y_edges'] = [y_edge2]
        edges_more_y['y_edges'] = [y_edge1]
    p1,p2 = z_edge1
    p3,p4 = z_edge2
    if p1[0]> p3[0]:
        edges_less_y['z_edges'] = [z_edge1]
        edges_more_y['z_edges'] = [z_edge2]
    else:
        edges_less_y['z_edges'] = [z_edge2]
        edges_more_y['z_edges'] = [z_edge1]
    p1,p2 = x_edge1
    p3,p4 = x_edge2
    if p1[1] < p3[1]:
        edges_less_y['x_edges'] = [x_edge1]
        edges_more_y['x_edges'] = [x_edge2]
    else:
        edges_less_y['x_edges'] = [x_edge2]
        edges_more_y['x_edges'] = [x_edge1]

    p1, p2 = z_edge1
    p3, p4 = z_edge2
    if p1[0] > p3[0]:
        less_x = p3[0]
        more_x = p1[0]
    else:
        less_x = p1[0]
        more_x = p3[0]
    if p1[1] > p3[1]:
        less_y = p3[1]
        more_y = p1[1]
    else:
        less_y = p1[1]
        more_y = p3[1]

    item = [ 'xy','yz', 'zx']
    result = random.sample(item, 1)

    #result = ['zx']

    if 'zx' in result:
        #print('zx')
        item = ['less_y', 'more_y']
        result = random.sample(item, 1)
        #result = ['more_y']

        if 'less_y' in result:
            #print('less_y')
            l1, w1, h1 = Block_size[0], less_y - y_start, Block_size[2]
            l2, w2, h2 = Block_size[0], y_end - less_y, Block_size[2]
            scrap_Boxes.extend(get_scrap_volume_of_type1(Block_size = [l1, w1, h1], edges = edges_less_y, st_co= [x_start,y_start,z_start]))
            scrap_Boxes.extend(get_scrap_volume_of_type1(Block_size = [l2, w2, h2], edges = edges_more_y, st_co= [x_start,less_y,z_start]))
            
        elif 'more_y' in result:
            #print('more_y')
            x_in, y_in, z_in          = x_start, more_y, z_start
            x_final, y_final, z_final = x_end, y_end, z_end
            scrap_Boxes.append({'starting_co': [x_in,y_in,z_in],'Box_size':[x_final - x_in, y_final -y_in, z_final -z_in] })

            item = [ 'xy','yz']
            result = random.sample(item, 1)
            #result=['yz']
            if 'xy' in result:
                #print('xy')
                x_in, y_in, z_in          = x_start, y_start, z_boundry
                x_final, y_final, z_final = x_end, more_y, z_end
                scrap_Boxes.append({'starting_co': [x_in,y_in,z_in],'Box_size':[x_final - x_in, y_final -y_in, z_final -z_in] })

                item = [ 'zx','yz']
                result = random.sample(item, 1)
                if 'zx' in result:
                    x_in, y_in, z_in          = more_x, y_start, z_start
                    x_final, y_final, z_final = x_end, less_y, z_boundry
                    scrap_Boxes.append({'starting_co': [x_in,y_in,z_in],'Box_size':[x_final - x_in, y_final -y_in, z_final -z_in] })

                    x_in, y_in, z_in          = less_x, less_y, z_start
                    x_final, y_final, z_final = x_end, more_y, z_boundry
                    scrap_Boxes.append({'starting_co': [x_in,y_in,z_in],'Box_size':[x_final - x_in, y_final -y_in, z_final -z_in] })
                elif 'yz' in result:
                    x_in, y_in, z_in          = more_x, y_start, z_start
                    x_final, y_final, z_final = x_end, more_y, z_boundry
                    scrap_Boxes.append({'starting_co': [x_in,y_in,z_in],'Box_size':[x_final - x_in, y_final -y_in, z_final -z_in] })

                    x_in, y_in, z_in          = less_x, less_y, z_start
                    x_final, y_final, z_final = more_x, more_y, z_boundry
                    scrap_Boxes.append({'starting_co': [x_in,y_in,z_in],'Box_size':[x_final - x_in, y_final -y_in, z_final -z_in] })
                
                
            elif 'yz' in result:
                #print('yz')
                x_in, y_in, z_in          = more_x, y_start, z_start
                x_final, y_final, z_final = x_end, more_y, z_end
                scrap_Boxes.append({'starting_co': [x_in,y_in,z_in],'Box_size':[x_final - x_in, y_final -y_in, z_final -z_in] })

                item = [ 'zx','xy']
                result = random.sample(item, 1)
                #result = ['zx']
                if 'xy' in result:
                    #print('xy')
                    x_in, y_in, z_in          = x_start, y_start, z_boundry
                    x_final, y_final, z_final = more_x, more_y, z_end
                    scrap_Boxes.append({'starting_co': [x_in,y_in,z_in],'Box_size':[x_final - x_in, y_final -y_in, z_final -z_in] })

                    x_in, y_in, z_in          = less_x, less_y, z_start
                    x_final, y_final, z_final = more_x, more_y, z_boundry
                    scrap_Boxes.append({'starting_co': [x_in,y_in,z_in],'Box_size':[x_final - x_in, y_final -y_in, z_final -z_in] })

                    #return scrap_Boxes

                elif 'zx' in result:

                    x_in, y_in, z_in          = less_x, less_y, z_start
                    x_final, y_final, z_final = more_x, more_y, z_end
                    scrap_Boxes.append({'starting_co': [x_in,y_in,z_in],'Box_size':[x_final - x_in, y_final -y_in, z_final -z_in] })

                    item = [ 'choice1','choice2']
                    result = random.sample(item, 1)
                    #result = ['choice1']
                    if 'choice1' in result:
                        x_in, y_in, z_in          = x_start, y_start, z_boundry
                        x_final, y_final, z_final = less_x, more_y, z_end
                        scrap_Boxes.append({'starting_co': [x_in,y_in,z_in],'Box_size':[x_final - x_in, y_final -y_in, z_final -z_in] })

                        x_in, y_in, z_in          = less_x, y_start, z_boundry
                        x_final, y_final, z_final = more_x, less_y, z_end
                        scrap_Boxes.append({'starting_co': [x_in,y_in,z_in],'Box_size':[x_final - x_in, y_final -y_in, z_final -z_in] })
                        
                    elif 'choice2' in result:
                        #print(result)
                        x_in, y_in, z_in          = x_start, less_y, z_boundry
                        x_final, y_final, z_final = less_x, more_y, z_end
                        scrap_Boxes.append({'starting_co': [x_in,y_in,z_in],'Box_size':[x_final - x_in, y_final -y_in, z_final -z_in] })

                        x_in, y_in, z_in          = x_start, y_start, z_boundry
                        x_final, y_final, z_final = more_x, less_y, z_end
                        scrap_Boxes.append({'starting_co': [x_in,y_in,z_in],'Box_size':[x_final - x_in, y_final -y_in, z_final -z_in] })

                
            

    elif 'yz' in result:
        item = ['less_x', 'more_x']
        result = random.sample(item, 1)
        #result = ['more_x']
        if 'less_x' in result:
            
            l1, w1, h1 = x_end - less_x, Block_size[1], Block_size[2]
            l2, w2, h2 = less_x - x_start, Block_size[1], Block_size[2]
            scrap_Boxes.extend(get_scrap_volume_of_type1(Block_size = [l1, w1, h1], edges = edges_less_y, st_co= [less_x,y_start,z_start]))
            scrap_Boxes.extend(get_scrap_volume_of_type1(Block_size = [l2, w2, h2], edges = edges_more_y, st_co= [x_start,y_start,z_start]))
            return scrap_Boxes

        elif 'more_x' in result:
            x_boundry = more_x
            x_in, y_in, z_in          = x_boundry, y_start, z_start
            x_final, y_final, z_final = x_end, y_end, z_end
            scrap_Boxes.append({'starting_co': [x_in,y_in,z_in],'Box_size':[x_final - x_in, y_final -y_in, z_final -z_in] })

            item = [ 'xy','yz', 'zx']
            result = random.sample(item, 1)
            #result=['zx']
            if 'xy' in result:
              
                x_in, y_in, z_in          = x_start, y_start, z_boundry
                x_final, y_final, z_final = more_x, y_end, z_end
                scrap_Boxes.append({'starting_co': [x_in,y_in,z_in],'Box_size':[x_final - x_in, y_final -y_in, z_final -z_in] })
                
                item = [ 'yz', 'zx']
                result = random.sample(item, 1)
                if 'yz' in result:
                    x_in, y_in, z_in          = x_start, more_y, z_start
                    x_final, y_final, z_final = x_boundry, y_end, z_boundry
                    scrap_Boxes.append({'starting_co': [x_in,y_in,z_in],'Box_size':[x_final - x_in, y_final -y_in, z_final -z_in] })

                    x_in, y_in, z_in          = less_x, less_y, z_start
                    x_final, y_final, z_final = more_x, y_end, z_boundry
                    scrap_Boxes.append({'starting_co': [x_in,y_in,z_in],'Box_size':[x_final - x_in, y_final -y_in, z_final -z_in] })
                    
                elif 'zx' in result:
                    x_in, y_in, z_in          = x_start, more_y, z_start
                    x_final, y_final, z_final = more_x, y_end, z_boundry
                    scrap_Boxes.append({'starting_co': [x_in,y_in,z_in],'Box_size':[x_final - x_in, y_final -y_in, z_final -z_in] })

                    x_in, y_in, z_in          = less_x, less_y, z_start
                    x_final, y_final, z_final = more_x, more_y, z_boundry
                    scrap_Boxes.append({'starting_co': [x_in,y_in,z_in],'Box_size':[x_final - x_in, y_final -y_in, z_final -z_in] })
            if 'yz' in result:
                l1, w1, h1 = more_x - less_x, Block_size[1], Block_size[2]
                l2, w2, h2 = less_x - x_start, Block_size[1], Block_size[2]
                scrap_Boxes.extend(get_scrap_volume_of_type1(Block_size = [l1, w1, h1], edges = edges_less_y, st_co= [less_x,y_start,z_start]))
                scrap_Boxes.extend(get_scrap_volume_of_type1(Block_size = [l2, w2, h2], edges = edges_more_y, st_co= [x_start,y_start,z_start]))
            if 'zx' in result:
                p1, p2 = edges['x_edges'][0]
                z_boundry = p1[2]
                
                item = ['less_y', 'more_y']
                result = random.sample(item, 1)
                #result = ['more_y']
                if 'less_y' in result:
                    
                    x_in, y_in, z_in          = x_start, y_start, z_boundry
                    x_final, y_final, z_final = more_x, less_y, z_end
                    scrap_Boxes.append({'starting_co': [x_in,y_in,z_in],'Box_size':[x_final - x_in, y_final -y_in, z_final -z_in] })
                    
                    l1, w1, h1 = more_x - x_start,y_end - less_y , Block_size[2]
                    scrap_Boxes.extend(get_scrap_volume_of_type1(Block_size = [l1, w1, h1], edges = edges_more_y, st_co= [x_start,less_y,z_start]))

                    
                elif 'more_y' in result:
                    x_in, y_in, z_in          = x_start, more_y, z_start
                    x_final, y_final, z_final = more_x, y_end, z_end
                    scrap_Boxes.append({'starting_co': [x_in,y_in,z_in],'Box_size':[x_final - x_in, y_final -y_in, z_final -z_in] })

                    """"
                    Be care full below two boxes can be extended two three boxes with two different choices also. Latter we can update the code
                    """
                    x_in, y_in, z_in          = x_start, y_start, z_boundry
                    x_final, y_final, z_final = more_x, y_end, z_end
                    scrap_Boxes.append({'starting_co': [x_in,y_in,z_in],'Box_size':[x_final - x_in, y_final -y_in, z_final -z_in] })

                    x_in, y_in, z_in          = less_x, less_y, z_start
                    x_final, y_final, z_final = more_x, more_y, z_boundry
                    scrap_Boxes.append({'starting_co': [x_in,y_in,z_in],'Box_size':[x_final - x_in, y_final -y_in, z_final -z_in] })
                    
                    
    
    elif 'xy' in result:
        z_boundry = 0
        y_boundry = 0
        x_boundry = 0

        p1, p2 = edges['x_edges'][0]
        z_boundry = p1[2]
        # Construct a box
        x_in, y_in, z_in = x_start, y_start, z_boundry
        x_final, y_final, z_final = x_end, y_end, z_end
        scrap_Boxes.append({'starting_co': [x_in,y_in,z_in],'Box_size':[x_final - x_in, y_final -y_in, z_final -z_in] })

        item = [ 'yz', 'zx']
        result = random.sample(item, 1)
        if 'yz' in result:
                
            item = ['less_x', 'more_x']
            result = random.sample(item, 1)
            if 'less_x' in result:
                x_in, y_in, z_in = x_start, more_y, z_start
                x_final, y_final, z_final = less_x, y_end, z_boundry
                scrap_Boxes.append({'starting_co': [x_in,y_in,z_in],'Box_size':[x_final - x_in, y_final -y_in, z_final -z_in] })

                item = ['zx', 'yz']
                result = random.sample(item, 1)
                if 'zx' in result:
                    x_in, y_in, z_in = less_x, less_y, z_start
                    x_final, y_final, z_final = x_end, y_end, z_boundry
                    scrap_Boxes.append({'starting_co': [x_in,y_in,z_in],'Box_size':[x_final - x_in, y_final -y_in, z_final -z_in] })
    
                    x_in, y_in, z_in = more_x, y_start, z_start
                    x_final, y_final, z_final = x_end, less_y, z_boundry
                    scrap_Boxes.append({'starting_co': [x_in,y_in,z_in],'Box_size':[x_final - x_in, y_final -y_in, z_final -z_in] })
                elif 'yz' in result:
                    x_in, y_in, z_in = less_x, less_y, z_start
                    x_final, y_final, z_final = more_x, y_end, z_boundry
                    scrap_Boxes.append({'starting_co': [x_in,y_in,z_in],'Box_size':[x_final - x_in, y_final -y_in, z_final -z_in] })
    
                    x_in, y_in, z_in = more_x, y_start, z_start
                    x_final, y_final, z_final = x_end, y_end, z_boundry
                    scrap_Boxes.append({'starting_co': [x_in,y_in,z_in],'Box_size':[x_final - x_in, y_final -y_in, z_final -z_in] })
                
            elif 'more_x' in result:
                # Construct a box
                x_in, y_in, z_in = more_x, y_start, z_start
                x_final, y_final, z_final = x_end, y_end, z_boundry
                scrap_Boxes.append({'starting_co': [x_in,y_in,z_in],'Box_size':[x_final - x_in, y_final -y_in, z_final -z_in] })

                item = ['zx', 'yz']
                result = random.sample(item, 1)
                if 'zx' in result:

                    x_in, y_in, z_in = x_start, more_y, z_start
                    x_final, y_final, z_final = more_x, y_end, z_boundry
                    scrap_Boxes.append({'starting_co': [x_in,y_in,z_in],'Box_size':[x_final - x_in, y_final -y_in, z_final -z_in] })
    
                    x_in, y_in, z_in = less_x, less_y, z_start
                    x_final, y_final, z_final = more_x, more_y, z_boundry
                    scrap_Boxes.append({'starting_co': [x_in,y_in,z_in],'Box_size':[x_final - x_in, y_final -y_in, z_final -z_in] })
                    
                    
                elif 'yz' in result:
                    x_in, y_in, z_in = x_start, more_y, z_start
                    x_final, y_final, z_final = less_x, y_end, z_boundry
                    scrap_Boxes.append({'starting_co': [x_in,y_in,z_in],'Box_size':[x_final - x_in, y_final -y_in, z_final -z_in] })
    
                    x_in, y_in, z_in = less_x, less_y, z_start
                    x_final, y_final, z_final = more_x, y_end, z_boundry
                    scrap_Boxes.append({'starting_co': [x_in,y_in,z_in],'Box_size':[x_final - x_in, y_final -y_in, z_final -z_in] })
                
        elif 'zx' in result:
            item = ['less_y', 'more_y']
            result = random.sample(item, 1)
            if 'less_y' in result:
                x_in, y_in, z_in = more_x, y_start, z_start
                x_final, y_final, z_final = x_end, less_y, z_boundry
                scrap_Boxes.append({'starting_co': [x_in,y_in,z_in],'Box_size':[x_final - x_in, y_final -y_in, z_final -z_in] })

                item = ['zx', 'yz']
                result = random.sample(item, 1)
                if 'zx' in result:
                    x_in, y_in, z_in = x_start, more_y, z_start
                    x_final, y_final, z_final = x_end, y_end, z_boundry
                    scrap_Boxes.append({'starting_co': [x_in,y_in,z_in],'Box_size':[x_final - x_in, y_final -y_in, z_final -z_in] })
    
                    x_in, y_in, z_in = less_x, less_y, z_start
                    x_final, y_final, z_final = x_end, more_y, z_boundry
                    scrap_Boxes.append({'starting_co': [x_in,y_in,z_in],'Box_size':[x_final - x_in, y_final -y_in, z_final -z_in] })
                elif 'yz' in result:
                    x_in, y_in, z_in = x_start, more_y, z_start
                    x_final, y_final, z_final = less_x, y_end, z_boundry
                    scrap_Boxes.append({'starting_co': [x_in,y_in,z_in],'Box_size':[x_final - x_in, y_final -y_in, z_final -z_in] })
    
                    x_in, y_in, z_in = less_x, less_y, z_start
                    x_final, y_final, z_final = x_end, y_end, z_boundry
                    scrap_Boxes.append({'starting_co': [x_in,y_in,z_in],'Box_size':[x_final - x_in, y_final -y_in, z_final -z_in] })
            elif 'more_y' in result:
                x_in, y_in, z_in = x_start, more_y, z_start
                x_final, y_final, z_final = x_end, y_end, z_boundry
                scrap_Boxes.append({'starting_co': [x_in,y_in,z_in],'Box_size':[x_final - x_in, y_final -y_in, z_final -z_in] })

                item = ['zx', 'yz']
                result = random.sample(item, 1)
                if 'zx' in result:
                    x_in, y_in, z_in = less_x, less_y, z_start
                    x_final, y_final, z_final = x_end, more_y, z_boundry
                    scrap_Boxes.append({'starting_co': [x_in,y_in,z_in],'Box_size':[x_final - x_in, y_final -y_in, z_final -z_in] })
    
                    x_in, y_in, z_in = more_x, y_start, z_start
                    x_final, y_final, z_final = x_end, less_y, z_boundry
                    scrap_Boxes.append({'starting_co': [x_in,y_in,z_in],'Box_size':[x_final - x_in, y_final -y_in, z_final -z_in] })

                elif 'yz' in result:
                    x_in, y_in, z_in = less_x, less_y, z_start
                    x_final, y_final, z_final = more_x, more_y, z_boundry
                    scrap_Boxes.append({'starting_co': [x_in,y_in,z_in],'Box_size':[x_final - x_in, y_final -y_in, z_final -z_in] })
    
                    x_in, y_in, z_in = more_x, y_start, z_start
                    x_final, y_final, z_final = x_end, more_y, z_boundry
                    scrap_Boxes.append({'starting_co': [x_in,y_in,z_in],'Box_size':[x_final - x_in, y_final -y_in, z_final -z_in] })




    return scrap_Boxes





def get_scrap_volume_of_type3(Block_size, edges, st_co= [0,0,0]):
    import random
    x_start = st_co[0]
    y_start = st_co[1]
    z_start = st_co[2]

    y_end = Block_size[1] + y_start
    z_end = Block_size[2] + z_start
    x_end = Block_size[0] + x_start

    scrap_Boxes = []

    z_edge1 = edges['z_edges'][0]
    z_edge2 = edges['z_edges'][1]
    y_edge1 = edges['y_edges'][0]
    y_edge2 = edges['y_edges'][1]
    x_edge1 = edges['x_edges'][0]
    x_edge2 = edges['x_edges'][1]

    p1, p2 = edges['y_edges'][0]
    x_boundry = p1[0]

    edges_less_y = {}
    edges_more_y = {}
    p1,p2 = y_edge1
    p3,p4 = y_edge2
    if p1[2]> p3[2]:
        edges_less_y['y_edges'] = [y_edge1]
        edges_more_y['y_edges'] = [y_edge2]
    else:
        edges_less_y['y_edges'] = [y_edge2]
        edges_more_y['y_edges'] = [y_edge1]
    p1,p2 = z_edge1
    p3,p4 = z_edge2
    if p1[1]> p3[1]:
        edges_less_y['z_edges'] = [z_edge2]
        edges_more_y['z_edges'] = [z_edge1]
    else:
        edges_less_y['z_edges'] = [z_edge1]
        edges_more_y['z_edges'] = [z_edge2]
    p1,p2 = x_edge1
    p3,p4 = x_edge2
    if p1[1] < p3[1]:
        edges_less_y['x_edges'] = [x_edge1]
        edges_more_y['x_edges'] = [x_edge2]
    else:
        edges_less_y['x_edges'] = [x_edge2]
        edges_more_y['x_edges'] = [x_edge1]


    p1, p2 = x_edge1
    p3, p4 = x_edge2
    if p1[2] > p3[2]:
        less_z = p3[2]
        more_z = p1[2]
    else:
        less_z = p1[2]
        more_z = p3[2]
    if p1[1] > p3[1]:
        less_y = p3[1]
        more_y = p1[1]
    else:
        less_y = p1[1]
        more_y = p3[1]



    item = [ 'xy','yz', 'zx']
    result = random.sample(item, 1)

    #esult = ['zx']

    if 'zx' in result:
        '''More choices are possible in the above zx plane. But, other possiblilities are already taken care of in other planes.
        But probability wise there may be an issue.
        '''
        x_in, y_in, z_in          = x_start, more_y, z_start
        x_final, y_final, z_final = x_end, y_end, z_end
        scrap_Boxes.append({'starting_co': [x_in,y_in,z_in],'Box_size':[x_final - x_in, y_final -y_in, z_final -z_in] })
        
        item = [ 'yz', 'xy']
        result = random.sample(item, 1)
        #result = ['xy']
        if 'yz' in result:
            x_in, y_in, z_in          = x_boundry, y_start, z_start
            x_final, y_final, z_final = x_end, more_y, z_end
            scrap_Boxes.append({'starting_co': [x_in,y_in,z_in],'Box_size':[x_final - x_in, y_final -y_in, z_final -z_in] })

            l1, w1, h1 = x_boundry - x_start, more_y - y_start, z_end - less_z
            scrap_Boxes.extend(get_scrap_volume_of_type1(Block_size = [l1, w1, h1], edges = edges_less_y, st_co= [x_start,y_start,less_z]))

            return scrap_Boxes
            
        elif 'xy' in result:
            x_in, y_in, z_in          = x_start, y_start, more_z
            x_final, y_final, z_final = x_end, more_y, z_end
            scrap_Boxes.append({'starting_co': [x_in,y_in,z_in],'Box_size':[x_final - x_in, y_final -y_in, z_final -z_in] })

            x_in, y_in, z_in          = x_boundry, y_start, z_start
            x_final, y_final, z_final = x_end, more_y, more_z
            scrap_Boxes.append({'starting_co': [x_in,y_in,z_in],'Box_size':[x_final - x_in, y_final -y_in, z_final -z_in] })

            x_in, y_in, z_in          = x_start, less_y, less_z
            x_final, y_final, z_final = x_boundry, more_y, more_z
            scrap_Boxes.append({'starting_co': [x_in,y_in,z_in],'Box_size':[x_final - x_in, y_final -y_in, z_final -z_in] })
            
            return scrap_Boxes

       

    elif 'xy' in result:
        item = ['less_z', 'more_z']
        result = random.sample(item, 1)
        #result = ['more_z']
        if 'less_z' in result:
            l1, w1, h1 = Block_size[0], Block_size[1], z_end - less_z
            scrap_Boxes.extend(get_scrap_volume_of_type1(Block_size = [l1, w1, h1], edges = edges_less_y, st_co= [x_start,y_start,less_z]))


            l1, w1, h1 = Block_size[0],Block_size[1], less_z - z_start
            scrap_Boxes.extend(get_scrap_volume_of_type1(Block_size = [l1, w1, h1], edges = edges_more_y, st_co= [x_start,y_start,z_start]))
                
        elif 'more_z' in result:
            x_in, y_in, z_in          = x_start, y_start, more_z
            x_final, y_final, z_final = x_end, y_end, z_end
            scrap_Boxes.append({'starting_co': [x_in,y_in,z_in],'Box_size':[x_final - x_in, y_final -y_in, z_final -z_in] })

            item = [ 'yz', 'zx']
            result = random.sample(item, 1)
            #result = ['zx']
            if 'yz' in result:
                x_in, y_in, z_in          = x_boundry, y_start, z_start
                x_final, y_final, z_final = x_end, y_end, more_z
                scrap_Boxes.append({'starting_co': [x_in,y_in,z_in],'Box_size':[x_final - x_in, y_final -y_in, z_final -z_in] })

                item = [ 'xy', 'zx']
                result = random.sample(item, 1)
                #result = ['zx']
                if 'xy' in result:
                    x_in, y_in, z_in          = x_start, less_y, less_z
                    x_final, y_final, z_final = x_boundry, y_end, more_z
                    scrap_Boxes.append({'starting_co': [x_in,y_in,z_in],'Box_size':[x_final - x_in, y_final -y_in, z_final -z_in] })

                    x_in, y_in, z_in          = x_start, more_y, z_start
                    x_final, y_final, z_final = x_boundry, y_end, less_z
                    scrap_Boxes.append({'starting_co': [x_in,y_in,z_in],'Box_size':[x_final - x_in, y_final -y_in, z_final -z_in] })
                    
                elif 'zx' in result:
                    x_in, y_in, z_in          = x_start, less_y, less_z
                    x_final, y_final, z_final = x_boundry, more_y, more_z
                    scrap_Boxes.append({'starting_co': [x_in,y_in,z_in],'Box_size':[x_final - x_in, y_final -y_in, z_final -z_in] })

                    x_in, y_in, z_in          = x_start, more_y, z_start
                    x_final, y_final, z_final = x_boundry, y_end, more_z
                    scrap_Boxes.append({'starting_co': [x_in,y_in,z_in],'Box_size':[x_final - x_in, y_final -y_in, z_final -z_in] })
                
            elif 'zx' in result:
                x_in, y_in, z_in          = x_start, more_y, z_start
                x_final, y_final, z_final = x_end, y_end, more_z
                scrap_Boxes.append({'starting_co': [x_in,y_in,z_in],'Box_size':[x_final - x_in, y_final -y_in, z_final -z_in] })

                '''Here there are two choices but i am taking only one. Later we can change it.'''
                x_in, y_in, z_in          = x_start, less_y, less_z
                x_final, y_final, z_final = x_boundry, more_y, more_z
                scrap_Boxes.append({'starting_co': [x_in,y_in,z_in],'Box_size':[x_final - x_in, y_final -y_in, z_final -z_in] })

                x_in, y_in, z_in          = x_boundry, y_start, z_start
                x_final, y_final, z_final = x_end, more_y, more_z
                scrap_Boxes.append({'starting_co': [x_in,y_in,z_in],'Box_size':[x_final - x_in, y_final -y_in, z_final -z_in] })
                    

    elif 'yz' in result:
        x_in, y_in, z_in          = x_boundry, y_start, z_start
        x_final, y_final, z_final = x_end, y_end, z_end
        scrap_Boxes.append({'starting_co': [x_in,y_in,z_in],'Box_size':[x_final - x_in, y_final -y_in, z_final -z_in] })


        item = [ 'xy','zx']
        result = random.sample(item, 1)

        #result = ['zx']
        if 'xy' in result:
            item = ['less_z', 'more_z']
            result = random.sample(item, 1)
            #result = ['less_z']
            if 'less_z' in result:
                x_in, y_in, z_in          = x_start, more_y, z_start
                x_final, y_final, z_final = x_boundry, y_end, less_z
                scrap_Boxes.append({'starting_co': [x_in,y_in,z_in],'Box_size':[x_final - x_in, y_final -y_in, z_final -z_in] })

                x_in, y_in, z_in          = x_start, less_y, less_z
                x_final, y_final, z_final = x_boundry, y_end, z_end
                scrap_Boxes.append({'starting_co': [x_in,y_in,z_in],'Box_size':[x_final - x_in, y_final -y_in, z_final -z_in] })

                x_in, y_in, z_in          = x_start, y_start, more_z
                x_final, y_final, z_final = x_boundry, less_y, z_end
                scrap_Boxes.append({'starting_co': [x_in,y_in,z_in],'Box_size':[x_final - x_in, y_final -y_in, z_final -z_in] })

                
            elif 'more_z' in result:
                x_in, y_in, z_in          = x_start, y_start, more_z
                x_final, y_final, z_final = x_boundry, y_end, z_end
                scrap_Boxes.append({'starting_co': [x_in,y_in,z_in],'Box_size':[x_final - x_in, y_final -y_in, z_final -z_in] })

                item = ['xy', 'zx']
                result = random.sample(item, 1)
                #result = ['zx']

                if 'xy' in result:
                    x_in, y_in, z_in          = x_start, less_y, less_z
                    x_final, y_final, z_final = x_boundry, y_end, more_z
                    scrap_Boxes.append({'starting_co': [x_in,y_in,z_in],'Box_size':[x_final - x_in, y_final -y_in, z_final -z_in] })

                    x_in, y_in, z_in          = x_start, more_y, z_start
                    x_final, y_final, z_final = x_boundry, y_end, less_z
                    scrap_Boxes.append({'starting_co': [x_in,y_in,z_in],'Box_size':[x_final - x_in, y_final -y_in, z_final -z_in] })

                elif 'zx' in result:
                    x_in, y_in, z_in          = x_start, less_y, less_z
                    x_final, y_final, z_final = x_boundry, more_y, more_z
                    scrap_Boxes.append({'starting_co': [x_in,y_in,z_in],'Box_size':[x_final - x_in, y_final -y_in, z_final -z_in] })

                    x_in, y_in, z_in          = x_start, more_y, z_start
                    x_final, y_final, z_final = x_boundry, y_end, more_z
                    scrap_Boxes.append({'starting_co': [x_in,y_in,z_in],'Box_size':[x_final - x_in, y_final -y_in, z_final -z_in] })

                

        elif 'zx' in result:
            item = ['less_y', 'more_y']
            result = random.sample(item, 1)
            #result = ['less_y']
            if 'less_y' in result:
                x_in, y_in, z_in          = x_start, y_start, more_z
                x_final, y_final, z_final = x_boundry, less_y, z_end
                scrap_Boxes.append({'starting_co': [x_in,y_in,z_in],'Box_size':[x_final - x_in, y_final -y_in, z_final -z_in] })


                item = ['choice1', 'choice2']
                result = random.sample(item, 1)
                #result = ['choice2']
                if 'choice1' in result:
                    x_in, y_in, z_in          = x_start, less_y, less_z
                    x_final, y_final, z_final = x_boundry, more_y, z_end
                    scrap_Boxes.append({'starting_co': [x_in,y_in,z_in],'Box_size':[x_final - x_in, y_final -y_in, z_final -z_in] })

                    x_in, y_in, z_in          = x_start, more_y, z_start
                    x_final, y_final, z_final = x_boundry, y_end, z_end
                    scrap_Boxes.append({'starting_co': [x_in,y_in,z_in],'Box_size':[x_final - x_in, y_final -y_in, z_final -z_in] })

                if 'choice2' in result:
                    x_in, y_in, z_in          = x_start, less_y, less_z
                    x_final, y_final, z_final = x_boundry, y_end, z_end
                    scrap_Boxes.append({'starting_co': [x_in,y_in,z_in],'Box_size':[x_final - x_in, y_final -y_in, z_final -z_in] })

                    x_in, y_in, z_in          = x_start, more_y, z_start
                    x_final, y_final, z_final = x_boundry, y_end, less_z
                    scrap_Boxes.append({'starting_co': [x_in,y_in,z_in],'Box_size':[x_final - x_in, y_final -y_in, z_final -z_in] })
                
            elif 'more_y' in result:
                x_in, y_in, z_in          = x_start, more_y, z_start
                x_final, y_final, z_final = x_boundry, y_end, z_end
                scrap_Boxes.append({'starting_co': [x_in,y_in,z_in],'Box_size':[x_final - x_in, y_final -y_in, z_final -z_in] })


                l1, w1, h1 = x_boundry - x_start, more_y - y_start, z_end - less_z
                scrap_Boxes.extend(get_scrap_volume_of_type1(Block_size = [l1, w1, h1], edges = edges_less_y, st_co= [x_start,y_start,less_z]))
                return scrap_Boxes
                


    return scrap_Boxes
    
    


def get_scrap_volume_of_type4(Block_size, edges, st_co= [0,0,0]):
    import random
    x_start = st_co[0]
    y_start = st_co[1]
    z_start = st_co[2]

    y_end = Block_size[1] + y_start
    z_end = Block_size[2] + z_start
    x_end = Block_size[0] + x_start

    scrap_Boxes = []

    z_edge1 = edges['z_edges'][0]
    z_edge2 = edges['z_edges'][1]
    z_edge3 = edges['z_edges'][2]
    y_edge1 = edges['y_edges'][0]
    y_edge2 = edges['y_edges'][1]
    y_edge3 = edges['y_edges'][2]
    x_edge1 = edges['x_edges'][0]
    x_edge2 = edges['x_edges'][1]
    x_edge3 = edges['x_edges'][2]

    edges_more_x_less_y = {}
    edges_more_x_more_y = {}
    edges_less_x = {}

    p1,p2 = y_edge1
    p3,p4 = y_edge2
    p5, p6 = y_edge3
    if p1[0] < p3[0]:
        more_z = p1[2]
        less_x = p1[0]
        more_x = p3[0]
        edges_less_x['y_edges'] = [y_edge1]
        if p5[2] > p3[2]:
            less_z =p3[2]
            edges_more_x_less_y['y_edges'] = [y_edge3]
            edges_more_x_more_y['y_edges'] = [y_edge2]
        else:
            less_z =p5[2]
            edges_more_x_less_y['y_edges'] = [y_edge2]
            edges_more_x_more_y['y_edges'] = [y_edge3]
            
    elif p1[0] > p3[0]:
        more_z = p3[2]
        less_x = p3[0]
        more_x = p1[0]
        edges_less_x['y_edges'] = [y_edge2]
        if p5[2] > p1[2]:
            less_z =p1[2]
            edges_more_x_less_y['y_edges'] = [y_edge3]
            edges_more_x_more_y['y_edges'] = [y_edge1]
        else:
            less_z =p5[2]
            edges_more_x_less_y['y_edges'] = [y_edge1]
            edges_more_x_more_y['y_edges'] = [y_edge3]
    else:
        more_z = p5[2]
        less_x = p5[0]
        more_x = p1[0]
        edges_less_x['y_edges'] = [y_edge3]
        if p1[2] > p3[2]:
            less_z =p3[2]
            edges_more_x_less_y['y_edges'] = [y_edge1]
            edges_more_x_more_y['y_edges'] = [y_edge2]
            
    
    p1,p2 = x_edge1
    p3,p4 = x_edge2
    p5, p6 = x_edge3

    if p1[1] <= p3[1] and p3[1] <= p5[1]:
        less_y = p1[1]
        moderate_y = p3[1]
        more_y = p5[1]
        edges_more_x_less_y['x_edges'] = [x_edge1]
        edges_less_x['x_edges'] = [x_edge2]
        edges_more_x_more_y['x_edges'] = [x_edge3]

    elif p1[1] <= p5[1] and p5[1] <= p3[1]:
        edges_more_x_less_y['x_edges'] = [x_edge1]
        edges_less_x['x_edges'] = [x_edge3]
        edges_more_x_more_y['x_edges'] = [x_edge2]
        less_y = p1[1]
        moderate_y = p5[1]
        more_y = p3[1]

    elif p3[1] <= p1[1] and p1[1] <= p5[1]:
        edges_more_x_less_y['x_edges'] = [x_edge2]
        edges_less_x['x_edges'] = [x_edge1]
        edges_more_x_more_y['x_edges'] = [x_edge3]
        less_y = p3[1]
        moderate_y = p1[1]
        more_y = p5[1]

    elif p3[1] <= p5[1] and p5[1] <= p1[1]:
        edges_more_x_less_y['x_edges'] = [x_edge2]
        edges_less_x['x_edges'] = [x_edge3]
        edges_more_x_more_y['x_edges'] = [x_edge1]
        less_y = p3[1]
        moderate_y = p5[1]
        more_y = p1[1]

    elif p5[1] <= p3[1] and p3[1] <= p1[1]:
        edges_more_x_less_y['x_edges'] = [x_edge3]
        edges_less_x['x_edges'] = [x_edge2]
        edges_more_x_more_y['x_edges'] = [x_edge1]
        less_y = p5[1]
        moderate_y = p3[1]
        more_y = p1[1]

    elif p5[1] <= p1[1] and p1[1] <= p3[1]:
        edges_more_x_less_y['x_edges'] = [x_edge3]
        edges_less_x['x_edges'] = [x_edge1]
        edges_more_x_more_y['x_edges'] = [x_edge2]
        less_y = p5[1]
        moderate_y = p1[1]
        more_y = p2[1]
    
    else:
        print(p1[1], p3[1], p5[1])
        assert False

    p1,p2 = z_edge1
    p3,p4 = z_edge2
    p5, p6 = z_edge3

    if p1[1] <= p3[1] and p3[1] <= p5[1]:
        edges_more_x_less_y['z_edges'] = [z_edge1]
        edges_less_x['z_edges'] = [z_edge2]
        edges_more_x_more_y['z_edges'] = [z_edge3]

    elif p1[1] <= p5[1] and p5[1] <= p3[1]:
        edges_more_x_less_y['z_edges'] = [z_edge1]
        edges_less_x['z_edges'] = [z_edge3]
        edges_more_x_more_y['z_edges'] = [z_edge2]
       
    elif p3[1] <= p1[1] and p1[1] <= p5[1]:
        edges_more_x_less_y['z_edges'] = [z_edge2]
        edges_less_x['z_edges'] = [z_edge1]
        edges_more_x_more_y['z_edges'] = [z_edge3]
     
    elif p3[1] <= p5[1] and p5[1] <= p1[1]:
        edges_more_x_less_y['z_edges'] = [z_edge2]
        edges_less_x['z_edges'] = [z_edge3]
        edges_more_x_more_y['z_edges'] = [z_edge1]

    elif p5[1] <= p3[1] and p3[1] <= p1[1]:
        edges_more_x_less_y['z_edges'] = [z_edge3]
        edges_less_x['z_edges'] = [z_edge2]
        edges_more_x_more_y['z_edges'] = [z_edge1]

    elif p5[1] <= p1[1] and p1[1] <= p3[1]:
        edges_more_x_less_y['z_edges'] = [z_edge3]
        edges_less_x['z_edges'] = [z_edge1]
        edges_more_x_more_y['z_edges'] = [z_edge2]
 
    else:
        print(p1[1], p3[1], p5[1])
        assert False

    
    
    item = [ 'xy','yz', 'zx']
    result = random.sample(item, 1)

    result = ['zx']
    if 'zx' in result:
        #print(result)
        item = [ 'moderate_y' ,'more_y']
        result = random.sample(item, 1)
        #result = ['less_y']
        if 'less_y' in result:
            pass
            '''
            Write the necessary code.
            '''
        elif 'moderate_y' in result:
            l1, w1, h1 = Block_size[0],y_end - moderate_y, Block_size[2]
            scrap_Boxes.extend(get_scrap_volume_of_type1(Block_size = [l1, w1, h1], edges = edges_more_x_more_y, st_co= [x_start ,moderate_y,z_start]))

            edges_new = {}
            edges_new['z_edges'] = [edges_less_x['z_edges'][0] , edges_more_x_less_y['z_edges'][0]]
            edges_new['y_edges'] = [edges_less_x['y_edges'][0] , edges_more_x_less_y['y_edges'][0]]
            edges_new['x_edges'] = [edges_less_x['x_edges'][0] , edges_more_x_less_y['x_edges'][0]]

            item = [ 'choice1', 'choice2']
            result = random.sample(item, 1)
            #result = ['choice1']

            if 'choice1' in result:
                l1, w1, h1 = more_x - x_start,moderate_y - y_start, z_end - less_z 
                scrap_Boxes.extend(get_scrap_volume_of_type2(Block_size = [l1, w1, h1], edges = edges_new, st_co= [x_start,y_start,less_z]))

                x_in, y_in, z_in          = more_x, y_start, z_start
                x_final, y_final, z_final = x_end, moderate_y, z_end
                scrap_Boxes.append({'starting_co': [x_in,y_in,z_in],'Box_size':[x_final - x_in, y_final -y_in, z_final -z_in] })

            elif 'choice2' in result:
                l1, w1, h1 = Block_size[0],moderate_y - y_start, z_end - less_z 
                scrap_Boxes.extend(get_scrap_volume_of_type2(Block_size = [l1, w1, h1], edges = edges_new, st_co= [x_start,y_start,less_z]))

                x_in, y_in, z_in          = more_x, y_start, z_start
                x_final, y_final, z_final = x_end, moderate_y, less_z
                scrap_Boxes.append({'starting_co': [x_in,y_in,z_in],'Box_size':[x_final - x_in, y_final -y_in, z_final -z_in] })

                return scrap_Boxes


        elif 'more_y' in result:
            #print(result)
            x_in, y_in, z_in          = x_start, more_y, z_start
            x_final, y_final, z_final = x_end, y_end, z_end
            scrap_Boxes.append({'starting_co': [x_in,y_in,z_in],'Box_size':[x_final - x_in, y_final -y_in, z_final -z_in] })

            item = [ 'xy', 'yz']
            result = random.sample(item, 1)
            #result = ['yz']
            if 'xy' in result:
                #print(result)
                x_in, y_in, z_in          = x_start, y_start, more_z
                x_final, y_final, z_final = x_end, more_y, z_end
                scrap_Boxes.append({'starting_co': [x_in,y_in,z_in],'Box_size':[x_final - x_in, y_final -y_in, z_final -z_in] })

                x_in, y_in, z_in          = more_x, y_start, z_start
                x_final, y_final, z_final = x_end, more_y, more_z
                scrap_Boxes.append({'starting_co': [x_in,y_in,z_in],'Box_size':[x_final - x_in, y_final -y_in, z_final -z_in] })

                item = [ 'choice1', 'choice2']
                result = random.sample(item, 1)
                #result = ['choice2']
                if 'choice1' in result:
                    x_in, y_in, z_in          = less_x, less_z, less_z
                    x_final, y_final, z_final = more_x, more_y, more_z
                    scrap_Boxes.append({'starting_co': [x_in,y_in,z_in],'Box_size':[x_final - x_in, y_final -y_in, z_final -z_in] })

                    x_in, y_in, z_in          = x_start, moderate_y, less_z
                    x_final, y_final, z_final = less_x, more_y, more_z
                    scrap_Boxes.append({'starting_co': [x_in,y_in,z_in],'Box_size':[x_final - x_in, y_final -y_in, z_final -z_in] })

                elif 'choice2' in result:
                    x_in, y_in, z_in          = less_x, less_z, less_z
                    x_final, y_final, z_final = more_x, moderate_y, more_z
                    scrap_Boxes.append({'starting_co': [x_in,y_in,z_in],'Box_size':[x_final - x_in, y_final -y_in, z_final -z_in] })

                    x_in, y_in, z_in          = x_start, moderate_y, less_z
                    x_final, y_final, z_final = more_x, more_y, more_z
                    scrap_Boxes.append({'starting_co': [x_in,y_in,z_in],'Box_size':[x_final - x_in, y_final -y_in, z_final -z_in] })


                

            elif 'yz' in result:
                x_in, y_in, z_in          = more_x, y_start, z_start
                x_final, y_final, z_final = x_end, more_y, z_end
                scrap_Boxes.append({'starting_co': [x_in,y_in,z_in],'Box_size':[x_final - x_in, y_final -y_in, z_final -z_in] })

                x_in, y_in, z_in          = x_start, y_start, more_z
                x_final, y_final, z_final = more_x, more_y, z_end
                scrap_Boxes.append({'starting_co': [x_in,y_in,z_in],'Box_size':[x_final - x_in, y_final -y_in, z_final -z_in] })

                item = [ 'choice1', 'choice2']
                result = random.sample(item, 1)
                #result = ['choice2']
                if 'choice1' in result:
                    x_in, y_in, z_in          = less_x, less_z, less_z
                    x_final, y_final, z_final = more_x, more_y, more_z
                    scrap_Boxes.append({'starting_co': [x_in,y_in,z_in],'Box_size':[x_final - x_in, y_final -y_in, z_final -z_in] })

                    x_in, y_in, z_in          = x_start, moderate_y, less_z
                    x_final, y_final, z_final = less_x, more_y, more_z
                    scrap_Boxes.append({'starting_co': [x_in,y_in,z_in],'Box_size':[x_final - x_in, y_final -y_in, z_final -z_in] })

                elif 'choice2' in result:
                    x_in, y_in, z_in          = less_x, less_z, less_z
                    x_final, y_final, z_final = more_x, moderate_y, more_z
                    scrap_Boxes.append({'starting_co': [x_in,y_in,z_in],'Box_size':[x_final - x_in, y_final -y_in, z_final -z_in] })

                    x_in, y_in, z_in          = x_start, moderate_y, less_z
                    x_final, y_final, z_final = more_x, more_y, more_z
                    scrap_Boxes.append({'starting_co': [x_in,y_in,z_in],'Box_size':[x_final - x_in, y_final -y_in, z_final -z_in] })


                
        

    elif 'yz' in result:
        item = ['less_y', 'more_x']
        result = random.sample(item, 1)
        #result = ['more_x']
        if 'more_x' in result:
            x_in, y_in, z_in          = more_x, y_start, z_start
            x_final, y_final, z_final = x_end, y_end, z_end
            scrap_Boxes.append({'starting_co': [x_in,y_in,z_in],'Box_size':[x_final - x_in, y_final -y_in, z_final -z_in] })

            item = [ 'xy', 'zx']
            result = random.sample(item, 1)
            #result = ['zx']
            if 'xy' in result:
                x_in, y_in, z_in          = x_start, y_start, more_z
                x_final, y_final, z_final = more_x, y_end, z_end
                scrap_Boxes.append({'starting_co': [x_in,y_in,z_in],'Box_size':[x_final - x_in, y_final -y_in, z_final -z_in] })

                x_in, y_in, z_in          = x_start, more_y, z_start
                x_final, y_final, z_final = more_x, y_end, more_z
                scrap_Boxes.append({'starting_co': [x_in,y_in,z_in],'Box_size':[x_final - x_in, y_final -y_in, z_final -z_in] })

                item = [ 'choice1', 'choice2']
                result = random.sample(item, 1)
                #result = ['choice2']
                if 'choice1' in result:
                    x_in, y_in, z_in          = less_x, less_z, less_z
                    x_final, y_final, z_final = more_x, more_y, more_z
                    scrap_Boxes.append({'starting_co': [x_in,y_in,z_in],'Box_size':[x_final - x_in, y_final -y_in, z_final -z_in] })

                    x_in, y_in, z_in          = x_start, moderate_y, less_z
                    x_final, y_final, z_final = less_x, more_y, more_z
                    scrap_Boxes.append({'starting_co': [x_in,y_in,z_in],'Box_size':[x_final - x_in, y_final -y_in, z_final -z_in] })

                elif 'choice2' in result:
                    x_in, y_in, z_in          = less_x, less_z, less_z
                    x_final, y_final, z_final = more_x, moderate_y, more_z
                    scrap_Boxes.append({'starting_co': [x_in,y_in,z_in],'Box_size':[x_final - x_in, y_final -y_in, z_final -z_in] })

                    x_in, y_in, z_in          = x_start, moderate_y, less_z
                    x_final, y_final, z_final = more_x, more_y, more_z
                    scrap_Boxes.append({'starting_co': [x_in,y_in,z_in],'Box_size':[x_final - x_in, y_final -y_in, z_final -z_in] })


                
            if 'zx' in result:
                x_in, y_in, z_in          = x_start, more_y, z_start
                x_final, y_final, z_final = more_x, y_end, z_end
                scrap_Boxes.append({'starting_co': [x_in,y_in,z_in],'Box_size':[x_final - x_in, y_final -y_in, z_final -z_in] })

                x_in, y_in, z_in          = x_start, y_start, more_z
                x_final, y_final, z_final = more_x, more_y, z_end
                scrap_Boxes.append({'starting_co': [x_in,y_in,z_in],'Box_size':[x_final - x_in, y_final -y_in, z_final -z_in] })

                item = [ 'choice1', 'choice2']
                result = random.sample(item, 1)
                #result = ['choice2']
                if 'choice1' in result:
                    x_in, y_in, z_in          = less_x, less_z, less_z
                    x_final, y_final, z_final = more_x, more_y, more_z
                    scrap_Boxes.append({'starting_co': [x_in,y_in,z_in],'Box_size':[x_final - x_in, y_final -y_in, z_final -z_in] })

                    x_in, y_in, z_in          = x_start, moderate_y, less_z
                    x_final, y_final, z_final = less_x, more_y, more_z
                    scrap_Boxes.append({'starting_co': [x_in,y_in,z_in],'Box_size':[x_final - x_in, y_final -y_in, z_final -z_in] })

                elif 'choice2' in result:
                    x_in, y_in, z_in          = less_x, less_z, less_z
                    x_final, y_final, z_final = more_x, moderate_y, more_z
                    scrap_Boxes.append({'starting_co': [x_in,y_in,z_in],'Box_size':[x_final - x_in, y_final -y_in, z_final -z_in] })

                    x_in, y_in, z_in          = x_start, moderate_y, less_z
                    x_final, y_final, z_final = more_x, more_y, more_z
                    scrap_Boxes.append({'starting_co': [x_in,y_in,z_in],'Box_size':[x_final - x_in, y_final -y_in, z_final -z_in] })




            
        elif 'less_x' in result:
            edges_new = {}
            edges_new['z_edges'] = [edges_more_x_more_y['z_edges'][0] , edges_more_x_less_y['z_edges'][0]]
            edges_new['y_edges'] = [edges_more_x_more_y['y_edges'][0] , edges_more_x_less_y['y_edges'][0]]
            edges_new['x_edges'] = [edges_more_x_more_y['x_edges'][0] , edges_more_x_less_y['x_edges'][0]]
            l1, w1, h1 = x_end - less_x,Block_size[1], Block_size[2]
            scrap_Boxes.extend(get_scrap_volume_of_type3(Block_size = [l1, w1, h1], edges = edges_new, st_co= [less_x ,y_start,z_start]))

            l1, w1, h1 = less_x - x_start ,Block_size[1], z_end - less_z
            scrap_Boxes.extend(get_scrap_volume_of_type1(Block_size = [l1, w1, h1], edges = edges_less_x, st_co= [x_start ,y_start,less_z]))

            x_in, y_in, z_in          = x_start, more_y, z_start
            x_final, y_final, z_final = less_x, y_end, less_z
            scrap_Boxes.append({'starting_co': [x_in,y_in,z_in],'Box_size':[x_final - x_in, y_final -y_in, z_final -z_in] })

            return scrap_Boxes

            

    elif 'xy' in result:
        item = ['less_z', 'more_z']
        result = random.sample(item, 1)
        #result = ['more_z']

        if 'less_z' in result:

            
            l1, w1, h1 = Block_size[0],Block_size[1], less_z - z_start
            scrap_Boxes.extend(get_scrap_volume_of_type1(Block_size = [l1, w1, h1], edges = edges_more_x_more_y, st_co= [x_start,y_start,z_start]))

            edges_new = {}
            edges_new['z_edges'] = [edges_less_x['z_edges'][0] , edges_more_x_less_y['z_edges'][0]]
            edges_new['y_edges'] = [edges_less_x['y_edges'][0] , edges_more_x_less_y['y_edges'][0]]
            edges_new['x_edges'] = [edges_less_x['x_edges'][0] , edges_more_x_less_y['x_edges'][0]]
            l1, w1, h1 = Block_size[0],Block_size[1], z_end - less_z 
            scrap_Boxes.extend(get_scrap_volume_of_type2(Block_size = [l1, w1, h1], edges = edges_new, st_co= [x_start,y_start,less_z]))

            return scrap_Boxes


        elif 'more_z' in result:
            x_in, y_in, z_in          = x_start, y_start, more_z
            x_final, y_final, z_final = x_end, y_end, z_end
            scrap_Boxes.append({'starting_co': [x_in,y_in,z_in],'Box_size':[x_final - x_in, y_final -y_in, z_final -z_in] })
            
            item = [ 'yz', 'zx']
            result = random.sample(item, 1)
            #result = ['zx']
            if 'yz' in result:
                x_in, y_in, z_in          = more_x, y_start, z_start
                x_final, y_final, z_final = x_end, y_end, more_z
                scrap_Boxes.append({'starting_co': [x_in,y_in,z_in],'Box_size':[x_final - x_in, y_final -y_in, z_final -z_in] })

                x_in, y_in, z_in          = x_start, more_y, z_start
                x_final, y_final, z_final = more_x, y_end, more_z
                scrap_Boxes.append({'starting_co': [x_in,y_in,z_in],'Box_size':[x_final - x_in, y_final -y_in, z_final -z_in] })

                item = [ 'choice1', 'choice2']
                result = random.sample(item, 1)
                #result = ['choice2']
                if 'choice1' in result:
                    x_in, y_in, z_in          = less_x, less_z, less_z
                    x_final, y_final, z_final = more_x, more_y, more_z
                    scrap_Boxes.append({'starting_co': [x_in,y_in,z_in],'Box_size':[x_final - x_in, y_final -y_in, z_final -z_in] })

                    x_in, y_in, z_in          = x_start, moderate_y, less_z
                    x_final, y_final, z_final = less_x, more_y, more_z
                    scrap_Boxes.append({'starting_co': [x_in,y_in,z_in],'Box_size':[x_final - x_in, y_final -y_in, z_final -z_in] })

                elif 'choice2' in result:
                    x_in, y_in, z_in          = less_x, less_z, less_z
                    x_final, y_final, z_final = more_x, moderate_y, more_z
                    scrap_Boxes.append({'starting_co': [x_in,y_in,z_in],'Box_size':[x_final - x_in, y_final -y_in, z_final -z_in] })

                    x_in, y_in, z_in          = x_start, moderate_y, less_z
                    x_final, y_final, z_final = more_x, more_y, more_z
                    scrap_Boxes.append({'starting_co': [x_in,y_in,z_in],'Box_size':[x_final - x_in, y_final -y_in, z_final -z_in] })

                
            elif 'zx' in result:
                x_in, y_in, z_in          = x_start, more_y, z_start
                x_final, y_final, z_final = x_end, y_end, more_z
                scrap_Boxes.append({'starting_co': [x_in,y_in,z_in],'Box_size':[x_final - x_in, y_final -y_in, z_final -z_in] })

                x_in, y_in, z_in          = more_x, y_start, z_start
                x_final, y_final, z_final = x_end, more_y, more_z
                scrap_Boxes.append({'starting_co': [x_in,y_in,z_in],'Box_size':[x_final - x_in, y_final -y_in, z_final -z_in] })

                item = [ 'choice1', 'choice2']
                result = random.sample(item, 1)
                #result = ['choice2']
                if 'choice1' in result:
                    x_in, y_in, z_in          = less_x, less_z, less_z
                    x_final, y_final, z_final = more_x, more_y, more_z
                    scrap_Boxes.append({'starting_co': [x_in,y_in,z_in],'Box_size':[x_final - x_in, y_final -y_in, z_final -z_in] })

                    x_in, y_in, z_in          = x_start, moderate_y, less_z
                    x_final, y_final, z_final = less_x, more_y, more_z
                    scrap_Boxes.append({'starting_co': [x_in,y_in,z_in],'Box_size':[x_final - x_in, y_final -y_in, z_final -z_in] })

                elif 'choice2' in result:
                    x_in, y_in, z_in          = less_x, less_z, less_z
                    x_final, y_final, z_final = more_x, moderate_y, more_z
                    scrap_Boxes.append({'starting_co': [x_in,y_in,z_in],'Box_size':[x_final - x_in, y_final -y_in, z_final -z_in] })

                    x_in, y_in, z_in          = x_start, moderate_y, less_z
                    x_final, y_final, z_final = more_x, more_y, more_z
                    scrap_Boxes.append({'starting_co': [x_in,y_in,z_in],'Box_size':[x_final - x_in, y_final -y_in, z_final -z_in] })

                
            
            
        

    return scrap_Boxes

    