import cv2
import numpy
img = cv2.imread("krop1.png") 
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.GaussianBlur(gray,(3,3),0)


edges = cv2.Canny(image=img, threshold1=100, threshold2=200)
edges = numpy.array(edges, dtype='float32')

#edges = edges.transpose()

f= open("ivice.bin","wb")
f.write(edges)
print(edges.shape)
cv2.imshow('Canny Edge Detection', edges)

cv2.waitKey(0)
