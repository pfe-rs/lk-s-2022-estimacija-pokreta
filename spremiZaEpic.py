from napravi_parove import parovi
from edge import *
#from subprocess import run
import sys

# index = sys.argv[1]

#flow_field je npy file
flow_field = sys.argv[1]
parovi(flow_field, "parovi.txt")


path_slike = sys.argv[2]
# path_slike = '../data_scene_flow/training/image_2/' + str(index).zfill(6) +  '_10.png'
if sys.argv[3]== 'sed':
    sed_ivice(path_slike, "ivice.bin")
elif sys.argv[3] == 'canny':
    canny_ivice(path_slike, "ivice.bin")