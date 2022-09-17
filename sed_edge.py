import cv2
import numpy as np
img = cv2.imread('../data_scene_flow/training/image_2/000002_11.png')
edgedetector = cv2.ximgproc.createStructuredEdgeDetection('../model.yml')
src = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
edges = edgedetector.detectEdges(np.float32(src) / 255.0)
inverted_edges = (1 - edges) 

print(inverted_edges)
cv2.imshow("edges", inverted_edges)
cv2.waitKey(0)

f= open("ivice.bin","wb")
f.write(inverted_edges)
f.close()