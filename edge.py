import cv2
import numpy as np

def sed_ivice(fileslike, binfile):
    img = cv2.imread(fileslike)
    edgedetector = cv2.ximgproc.createStructuredEdgeDetection('model.yml')
    src = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    edges = edgedetector.detectEdges(np.float32(src) / 255.0)
    inverted_edges = (1 - edges) 

    print(inverted_edges)
    cv2.imshow("edges", inverted_edges)
    cv2.waitKey(0)

    f= open(binfile,"wb")
    f.write(inverted_edges)
    f.close()
    
def canny_ivice(fileslike, binfile):
    img = cv2.imread(fileslike) 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(gray,(3,3),0)


    edges = cv2.Canny(image=img, threshold1=100, threshold2=200)
    inverted_edges1 = np.array(edges, dtype='float32')

    inverted_edges1 = np.array((255 - edges)/255, dtype='float32')
    #inverted_edges1 = np.copy(inverted_edges.transpose(), order='C') 
    cv2.imshow('Canny Edge Detection', inverted_edges1)
    cv2.waitKey(0)

    #print(inverted_edges)
    f= open(binfile,"wb")
    f.write(inverted_edges1)
