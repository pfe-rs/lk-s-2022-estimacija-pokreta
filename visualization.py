import numpy as np
import cv2
from matplotlib.cm import get_cmap

cmap = get_cmap('jet')

class FlowImage:
    def __init__(self ):
        self.width = 0
        self.height = 0


    def readFlowField(self, file_name):
        img = cv2.imread(file_name, -1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.width, self.height,  _ = img.shape
        self.data = np.zeros(self.width*self.height*3,  dtype=np.float32)
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
        if val: self.data[3*(v*self.width+u)+2] =1
        else: self.data[3*(v*self.width+u)+2] = 0

    def isValid(self,u,v):
        return (self.data[3*(v*self.width+u)+2]>0.5)

    def setFlowU(self, u,v,val):
        self.data[3*(v*self.width+u)+0] = val

    def setFlowV (self,u,v,val):
        self.data[3*(v*self.width+u)+1] = val
    
    def getFlowU(self,u,v):
        return self.data[3*(v*self.width+u)+0]
    
    def getFlowV (self,u,v):
        return self.data[3*(v*self.width+u)+1]

    def writeFlowField(self):
        image = np.zeros((self.width,self.height,3),  dtype=np.uint8)
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
    #zasto ne od 0, h
    errs = []
    for v in range(1, ground_t.height-1):
        for u in range(1, ground_t.width-1):
            if ground_t.isValid(u,v):
                dfu = test.getFlowU(u,v) - ground_t.getFlowU(u,v)
                dfv = test.getFlowV(u,v) - ground_t.getFlowV(u,v)
                f_err = np.sqrt(dfu*dfu + dfv*dfv)
                errs.append(f_err)
                r, g, b, _ = cmap(min(f_err, 100.0)/100.0)
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
                
test = FlowImage()

old_frame = cv2.imread("C:/Users/Milica/Desktop/data_scene_flow/training/image_2/000001_10.png") 
###definisi datu za test
test.width, test.height, _ = old_frame.shape
test.data = np.zeros(test.width*test.height*3, dtype=np.float32)
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
#trenutna slika
frame = cv2.imread("C:/Users/Milica/Desktop/data_scene_flow/training/image_2/000001_11.png")
frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# calculate dense optical flow
flow = cv2.calcOpticalFlowFarneback(old_gray, frame_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
#vizuelizacija
# hsv = np.zeros_like(frame)
# hsv[..., 1] = 255
# mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
# hsv[..., 0] = ang*180/np.pi/2
# hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
# bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
# cv2.imshow('frame2', bgr)
# cv2.waitKey(0)

groundt = FlowImage()
groundt.readFlowField("C:/Users/Milica/Desktop/data_scene_flow/training/flow_noc/000001_10.png")

br = 0
for v in range(test.height):
    for u in range(test.width):
        #skalirane
        # test.setFlowU(u,v,(flow[u][v][0]-2**15)/64)
        # test.setFlowV(u,v,(flow[u][v][1]-2**15)/64)
        test.setFlowU(u,v,flow[u][v][0])
        test.setFlowV(u,v,flow[u][v][1])
        test.setValid(u,v,True)
        #print((flow[u][v][0]))

imgasdadadsasd = cv2.imread("C:/Users/Milica/Desktop/data_scene_flow/training/flow_noc/000000_10.png", -1)
cv2.imshow('aaa', imgasdadadsasd)

test.writeFlowField()

errorImage(test, groundt)
# print(flowErrorsOutlier(test, groundt))
cv2.waitKey(0)
cv2.destroyAllWindows()