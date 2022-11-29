import numpy as np
import sys
import os

class FlowImage:

    def ucitajFlow(self, path):
        name, extension = os.path.splitext(path)
        if extension == '.npy':
            flow = np.load(path)
            self.height, self.width, _= flow.shape
            self.flow = np.zeros((self.height, self.width,3),  dtype=np.float32)
            for v in range(self.height):
                for u in range(self.width):
                    self.setFlowU(u,v,flow[v][u][1])
                    self.setFlowV(u,v,flow[v][u][0])
                    self.setValid(u,v,True)
    def setFlowU(self, u,v,val):
        self.flow[v][u][0] = val

    def setFlowV (self,u,v,val):
        self.flow[v][u][1] = val
    
    def setValid(self, u,v,val):
        if val: self.flow[v][u][2]=1
        else: self.flow[v][u][2] = 0


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

    # for u in range(flow2.shape[0]):
    #         for v in range(flow2.shape[1]):
    #             consistencyCheck(flow2, flow1, u, v, tresh)

def postProcessing(filename1,filename2, con_tresh, npysave):

    test1 = FlowImage()
    test1.ucitajFlow(filename1)


    test2 = FlowImage()
    test2.ucitajFlow(filename2)

    #removeSmallSegments(test1.flow, 10,100 )
    fowardBackwardConsistency(test1.flow, test2.flow, con_tresh)
    np.save(npysave,test1.flow)
    return test1

# file1 = sys.argv[1]
# file2 = sys.argv[2]
# con_tresh = int(sys.argv[3])


# postProcessing(file1, file2, con_tresh, 'sredjeni_flow.npy')
