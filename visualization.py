import numpy as np
import cv2
from matplotlib.cm import get_cmap

cmap = get_cmap('jet')


class FlowImage:

    def readFlowField(self, file_name):
        picw = 240
        pich = 100
        x = 700
        y = 122
        img = cv2.imread(file_name, -1)
        # img = cv2.resize(img1,(picw,pich))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img[y:y+pich, x:x+picw, :]
        # img[:, :, 0] = 0
        # img[:, :, 1] = 0
        # img[:, :, 2] *= 2**16-1
        # cv2.imshow('ime prozora', img)
        # cv2.waitKey(0)
        self.width, self.height,  _ = img.shape
        self.flow = np.zeros((self.width, self.height, 3),  dtype=np.float32)
        for v in range(0, self.height):
            for u in range(0, self.width):
                boje = img[u][v]
                if (boje[2]>0):
                    self.setFlowU(u,v,(boje[0] - 32768.0)/64.0)
                    self.setFlowV(u,v,(boje[1] - 32768.0)/64.0)
                    self.setValid(u,v,True)
                else:
                    self.setFlowU(u,v,0)
                    self.setFlowV(u,v,0)
                    self.setValid(u,v,False)
        

    def setValid(self, u,v,val):
        if val: self.flow[u][v][2]=1
        else: self.flow[u][v][2] = 0

    def isValid(self,u,v):
        return (self.flow[u][v][2]>0.5)

    def setFlowU(self, u,v,val):
        self.flow[u][v][0] = val

    def setFlowV (self,u,v,val):
        self.flow[u][v][1] = val
    
    def getFlowU(self,u,v):
        return self.flow[u][v][0]
    
    def getFlowV (self,u,v):
        return self.flow[u][v][1]

    def writeFlowField(self):
        #ispravnije je 16 ali 8 je vise kyl
        image = np.zeros((self.width,self.height,3),  dtype=np.uint16)
        for v in range (self.height):
            for u in range(self.width):
                if self.isValid(u,v):
                    image[u][v] = [self.getFlowU(u,v)*64.0+32768, self.getFlowV(u,v)*64.0+32768, 1]
        cv2.imshow('flofi',cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


    def getFlowMagnitude(self,u,v):
        fu = np.int64(self.getFlowU(u,v))
        fv = np.int64(self.getFlowV(u,v))
        return np.int16(np.sqrt(fu*fu+fv*fv))
    


def errorImage(test, ground_t):
    image = np.zeros((ground_t.width,ground_t.height,3), np.uint8)
    image_fl = np.zeros((ground_t.width,ground_t.height,3), np.uint8)
    image_gt = np.zeros((ground_t.width,ground_t.height,3), np.uint8)
    errs = []
    for v in range(ground_t.height):
        for u in range(ground_t.width):
            if ground_t.isValid(u,v):
                dfu = test.getFlowU(u,v) - ground_t.getFlowU(u,v)
                dfv = test.getFlowV(u,v) - ground_t.getFlowV(u,v)
                f_err = np.sqrt(dfv*dfv + dfu*dfu)
                errs.append(f_err)
                r, g, b, _ = cmap(min(f_err, 3.0)/3.0)
                image[u][v] = (np.uint8(b * 255), np.uint8(g * 255), np.uint8(r * 255))
                # r, g, b, _ = cmap(min(test.getFlowU(u,v), 10.0))
                # image_fl[u][v] = (b * 255, g * 255, r * 255)
                # r, g, b, _ = cmap(min(ground_t.getFlowU(u,v), 10.0))
                # image_gt[u][v] = (b * 255, g * 255, r * 255)
                #print(image[u][v])
    errs_np = np.array(errs)
    cv2.imshow('greska',image)
    # cv2.imshow('flow',image_fl)
    # cv2.imshow('ground truth',image_gt)
    

def flowErrorsOutlier(test, ground_t):
    ABS_THRESH = 3.0
    REL_THRESH = 0.05
    num_errors = 0
    for v in range(0, ground_t.height):
        for u in range(0, ground_t.width):
            if ground_t.isValid(u,v):
                fu = test.getFlowU(u,v) - ground_t.getFlowU(u,v)
                fv = test.getFlowV(u,v) - ground_t.getFlowV(u,v)
                f_dist = np.sqrt(fu*fu+fv*fv)
                f_mag  = ground_t.getFlowMagnitude(u,v)
                #print('mag',f_mag )
                f_err  = (f_dist>ABS_THRESH) and (f_dist/f_mag>REL_THRESH)
                if f_err:
                    num_errors+=1
    return num_errors


#load groun truth
groundt = FlowImage()
groundt.readFlowField("../data_scene_flow/training/flow_noc/000000_10.png")

# load optical flow
flow = np.load('mali_flow_nakon_6_2.npy')
test = FlowImage()
test.width, test.height, _ = flow.shape
test.flow = np.zeros((test.width,test.height,3),  dtype=np.float32)

for v in range(test.height):
    for u in range(test.width):
        test.setFlowU(u,v,flow[u][v][1])
        test.setFlowV(u,v,flow[u][v][0])
        test.setValid(u,v,True)
        #print((flow[u][v][0]))

groundt.writeFlowField()

errorImage(test, groundt)
# print(flowErrorsOutlier(test, groundt))
cv2.waitKey(0)
cv2.destroyAllWindows()