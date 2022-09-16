import cv2
from PIL import Image
#"C:/Users/Milica/Desktop/data_scene_flow/training/image_2/000000_10.png"
#"C:/Users/Milica/Desktop/data_scene_flow/training/image_2/000000_11.png"
img = Image.open("C:/Users/Milica/Desktop/data_scene_flow/training/image_2/000000_11.png")

picw = 240
pich = 100
x = 700
y = 122

box = (x, y, x+picw, y + pich)
img2 = img.crop(box)

img2.save("krop2.png")
# bin_img = sklearn.preprocessing.binarize(img)
# f= open("krop1.png","wb")
# f.write(bin_img)