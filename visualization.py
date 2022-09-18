import numpy as np
import cv2
from matplotlib.cm import get_cmap
import sys
import os

cmap = get_cmap('jet')

def read_flo_file(filename):
    """
    Read from Middlebury .flo file
    :param flow_file: name of the flow file
    :return: optical flow data in matrix
    """
    f = open(filename, 'rb')
    magic = np.fromfile(f, np.float32, count=1)
    data2d = None

    if 202021.25 != magic:
        print('Magic number incorrect. Invalid .flo file')
    else:
        w = np.fromfile(f, np.int32, count=1)
        h = np.fromfile(f, np.int32, count=1)
        # print "Reading %d x %d flow file in .flo format" % (h, w)
        data2d = np.fromfile(f, np.float32, count=2 * int(w) * int(h))
        # reshape data into 3D array (columns, rows, channels)
        data2d = np.resize(data2d, (int(h), int(w), 2))
    f.close()
    return data2d

def crop(img, x, y, picw, pich):
    return img[y:y+pich, x:x+picw, :]


class FlowImage:

    def readFlowFieldFromImage(self, file_name):
        img = cv2.imread(file_name, -1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #img = crop(img, 700, 122, 240, 100)

        self.height, self.width,  _ = img.shape
        self.flow = np.zeros(( self.height, self.width, 3),  dtype=np.float32)
        for v in range(0, self.height):
            for u in range(0, self.width):
                boje = img[v][u]
                if (boje[2]>0):
                    self.setFlowU(u,v,(boje[0] - 32768.0)/64.0)
                    self.setFlowV(u,v,(boje[1] - 32768.0)/64.0)
                    self.setValid(u,v,True)
                else:
                    self.setFlowU(u,v,0)
                    self.setFlowV(u,v,0)
                    self.setValid(u,v,False)
        

    def setValid(self, u,v,val):
        if val: self.flow[v][u][2]=1
        else: self.flow[v][u][2] = 0

    def isValid(self,u,v):
        return (self.flow[v][u][2]>0.5)

    def setFlowU(self, u,v,val):
        self.flow[v][u][0] = val

    def setFlowV (self,u,v,val):
        self.flow[v][u][1] = val
    
    def getFlowU(self,u,v):
        return self.flow[v][u][0]
    
    def getFlowV (self,u,v):
        return self.flow[v][u][1]

    def writeFlowField(self):
        #ispravnije je 16 ali 8 je vise kyl
        image = np.zeros((self.height, self.width,3),  dtype=np.uint16)
        for v in range (self.height):
            for u in range(self.width):
                if self.isValid(u,v):
                    image[v][u] = [self.getFlowU(u,v)*64.0+32768, self.getFlowV(u,v)*64.0+32768, 1]
        cv2.imshow('flofi',cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


    def getFlowMagnitude(self,u,v):
        fu = np.int64(self.getFlowU(u,v))
        fv = np.int64(self.getFlowV(u,v))
        return np.int16(np.sqrt(fv*fv + fu*fu))

    def ucitajFlow(self, path):
        name, extension = os.path.splitext(path)

        if extension == '.png':
            #to je kitti ground truth
            groundt.readFlowFieldFromImage(path)

        if extension == '.npy':
            np.load(path)
            for v in range(self.height):
                for u in range(self.width):
                    self.setFlowU(u,v,flow[v][u][1])
                    self.setFlowV(u,v,flow[v][u][0])
                    self.setValid(u,v,True)

        if extension == '.flo':
            flow = read_flo_file(path)
            self.height, self.width, _= flow.shape
            self.flow = np.zeros((self.height, self.width,3),  dtype=np.float32)
            for v in range(self.height):
                for u in range(self.width):
                    self.setFlowU(u,v,flow[v][u][0])
                    self.setFlowV(u,v,flow[v][u][1])
                    self.setValid(u,v,True)
    
    

def errorImage(test, ground_t, filename = None):
    ABS_THRESH = 3.0
    #REL_THRESH = 0.05
    num_errors = 0
    br_valid = 0
    image = np.zeros((ground_t.height, ground_t.width,3), np.uint8)
    errs = []
    for v in range(ground_t.height):
        for u in range(ground_t.width):
            if ground_t.isValid(u,v):
                #and test.isValid(u,v)
                dfu = test.getFlowU(u,v) - ground_t.getFlowU(u,v)
                dfv = test.getFlowV(u,v) - ground_t.getFlowV(u,v)
                f_err = np.sqrt(dfu*dfu + dfv*dfv)
                errs.append(f_err)
                r, g, b, _ = cmap(min(f_err, 3.0)/3.0)
                image[v][u] = (np.uint8(b * 255), np.uint8(g * 255), np.uint8(r * 255))

                br_valid += 1
                if f_err>ABS_THRESH:
                    num_errors+=1
    errs_np = np.average(np.array(errs))
    print('srednja greska', errs_np)
    print('procenat outliera', num_errors*100/br_valid)
    if not(filename == None):
        cv2.imwrite(filename, image)
    cv2.imshow('greska',image)
    

#load ground truth
groundt = FlowImage()
# kitti ground truh = "../data_scene_flow/training/flow_noc/000002_10.png"
gt_path = sys.argv[1]
groundt.ucitajFlow(gt_path)

#ucitaj EpicFlow
test = FlowImage()
test_path = sys.argv[2]
test.ucitajFlow(test_path)


# save = sys.argv[3]
# if save:
#     name, extension = os.path.splitext(test_path)
#     path_greske = 'greske/' + name + '.jpg'
# else: path_greske = None


try:
  path_greske = sys.argv[3]
except IndexError:
  path_greske = None


test.writeFlowField()
errorImage(test, groundt, path_greske)
# print(flowErrorsOutlier(test, groundt))
cv2.waitKey(0)
cv2.destroyAllWindows()