from napravi_parove import parovi
from edge import *
from subprocess import run
import sys
import os
from postprocessing import postProcessing

#par slika iz baze podataka, relativna lokacija, nalazi se u istom nad direktorijumu kao kod
kitti1 = sys.argv[1]
kitti2 = sys.argv[2]

foward = sys.argv[3]
backward = sys.argv[4]
con_tresh = int(sys.argv[5])
postProcessing(foward, backward, con_tresh, "sparse_field.npy")

parovi("sparse_field.npy", "parovi.txt")

if sys.argv[6]== 'sed':
    sed_ivice(kitti1, "ivice.bin")
elif sys.argv[6] == 'canny':
    canny_ivice(kitti1, "ivice.bin")



run(["../discrete_flow/external/EpicFlow_v1.00/epicflow-static",kitti1 ,kitti2, "ivice.bin", "parovi.txt", "epic.flo"])
