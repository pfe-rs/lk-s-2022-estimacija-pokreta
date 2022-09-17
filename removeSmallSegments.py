import numpy as np

def removeSmallSegments(flow, tresh, min_segment_size):
    width, height, _ = flow.shape
    check = np.zeros((width, height))

    u_neighbor = np.zeros(4, int)
    v_neighbor = np.zeros(4, int)

    for v  in range(height):
        for u in range(width):
            seg_list_u = []
            seg_list_v = []
            #ako nije vec deo celine
            if not(check[u][v]):
                seg_list_u.append(u)
                seg_list_v.append(v)
                count = 1
                curr = 0

                while(curr < count):
                    u_seg_curr = seg_list_u[curr]
                    v_seg_curr = seg_list_v[curr]
                    
                    u_neighbor[0] = u_seg_curr - 1
                    u_neighbor[1] = u_seg_curr + 1
                    u_neighbor[2] = u_seg_curr
                    u_neighbor[3] = u_seg_curr

                    v_neighbor[0] = v_seg_curr
                    v_neighbor[1] = v_seg_curr
                    v_neighbor[2] = v_seg_curr - 1
                    v_neighbor[3] = v_seg_curr + 1

                    for u_n, v_n in zip(u_neighbor, v_neighbor):
                        if (u_n>=0 and v_n >= 0 and u_n < width and v_n < height):
                                if (not(check[u_n][v_n]) and (flow[u_n][v_n][2] > 0.5)):
                                    if (abs(flow[u_seg_curr][v_seg_curr][0] - flow[u_n][v_n][0]) +
                                    abs(flow[u_seg_curr][v_seg_curr][1] - flow[u_n][v_n][1]) <= tresh):
                                        
                                        seg_list_u.append(u_n)
                                        seg_list_v.append(v_n)
                                        count += 1
                                        check[u_n][v_n] = 1
                                    
                    curr += 1
                    check[u_seg_curr][v_seg_curr] = 1
                if (1 < count < min_segment_size):
                    for u, v in zip(seg_list_u, seg_list_v):
                        flow[u][v][2] = 0

#flow1 pravi, flow2 unazad
def consistencyCheck(flow1, flow2, u1, v1, tresh):
    width, height, _ = flow1.shape


    #ako nije validan
    if not(flow1[u1][v1][2]> 0.5):
        return

    u2 = int(flow1[u1][v1][0] + u1)
    v2 = int(flow1[u1][v1][1] + v1)

    if (u2<0 or v2<0 or u2>=width or v2>= height):
        flow1[u1][v1][0] = 0
        flow1[u1][v1][1] = 0
        flow1[u1][v1][2] = 0
        return
    
    if not(flow2[u2][v2][2]> 0.5):
        flow1[u1][v1][0] = 0
        flow1[u1][v1][1] = 0
        flow1[u1][v1][2] = 0
        return

    du = flow1[u1][v1][0] + flow2[u2][v2][0]
    dv = flow1[u1][v1][1] + flow2[u2][v2][1]
    err = np.sqrt(dv*dv + du*du)
    if (err > tresh):
        flow1[u1][v1][0] = 0
        flow1[u1][v1][1] = 0
        flow1[u1][v1][2] = 0
        return



def fowardBackwardConsistency(flow1, flow2, tresh):
    for u in range(flow1.shape[0]):
            for v in range(flow1.shape[1]):
                consistencyCheck(flow1, flow2, u, v, tresh)

    for u in range(flow2.shape[0]):
            for v in range(flow2.shape[1]):
                consistencyCheck(flow2, flow1, u, v, tresh)


