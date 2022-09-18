from operator import truediv
from struct import pack
import cv2
import numpy as np
from scipy import spatial as sp
import pyflann as fl
import datetime as dt
import sys


picindex=sys.argv[1]
backward=sys.argv[2] #u konzoli, 0 znaci forward, 1 znaci backward

if(len(picindex)==1):
    picindex='0'+picindex

picw = 1242
pich = 375
# picw=240
# pich=125
#x = 700
x = 0
#y = 122
y = 0
cellw = 54
cellh = 25
# cellw=50
# cellh=25
tphi = 2.5
tpsi = 8
lamda = 0.05
unpacktime = 0.0
truestime = 0.0

descrs1 = np.zeros((pich, picw, 68), dtype=np.float32)
descrs2 = np.zeros((pich, picw, 68), dtype=np.float32)
#bcd_times = 5
bigtpsi = tpsi+cellw+cellh

ncellx = picw//cellw  # oko 20
ncelly = pich//cellh  # oko 10
#maxflowdist = 100
maxnprop = 150
proposals = np.full((pich, picw, maxnprop, 2), -1, dtype=int)
lcosts = np.full((pich, picw, maxnprop), 1000.0, dtype=np.double)
nprop = np.zeros((pich, picw), dtype=int)
# ngaussprop=np.zeros((pich,picw),dtype=int)
mindists = np.full((pich, picw), 1000.0)
#minvecs=np.zeros((pich,picw,2), dtype=int)
bestlabels = np.zeros((pich, picw), dtype=int)


kdim = maxnprop*maxnprop//8+1
packedksets = np.zeros((pich, picw, 2, kdim),
                       dtype=np.uint8)  # cuva labele, ne psi

ksets = np.zeros((maxnprop, maxnprop), dtype=bool)
tv = np.zeros(2, dtype=int)
myb = np.zeros(maxnprop)


def sidepsi(y1, x1, label1, y2, x2):
    if (y2 >= 0 and y2 < pich and x2 >= 0 and x2 < picw):
        # treba refinisati
        return min(tpsi, np.sum(np.abs(proposals[y1, x1, label1]-proposals[y2, x2, bestlabels[y2, x2]])))
    return 0


def purepsi(yv1, xv1, yv2, xv2):
    return np.abs(yv2-yv1)+np.abs(xv2-xv1)



def pripremi_za_oba_pakovanja():
    global proposals
    global lcosts
    global nprop
    global bestlabels
    proposals=np.load('Daisy output slike '+picindex+' backward='+backward+' proposals_nakon_gausa.npy')
    lcosts=np.load('Daisy output slike '+picindex+' backward='+backward+' lcosts_nakon_gausa.npy')
    nprop=np.load('Daisy output slike '+picindex+' backward='+backward+' nprop.npy')
    bestlabels=np.load('Bestlabels fajl slike '+picindex+' backward='+backward+' posle 00 BCD.npy')

def pakovanjeZaC():
    packedksets0 = np.zeros(((picw+1)//2, pich, kdim),dtype=np.uint8)
    packedksets1 = np.zeros(((pich+1)//2, picw, kdim),dtype=np.uint8)
    packedksets2 = np.zeros(((picw+1)//2, pich, kdim),dtype=np.uint8)
    packedksets3 = np.zeros(((pich+1)//2, picw, kdim),dtype=np.uint8)
    # gornji za mene, levi za mene, moj za donji, moj za desni (smrt)
    ksets4 = np.zeros((4, maxnprop, maxnprop), dtype=bool)
    for ty in range(pich-1):
        print('pocinjem red y=', ty)
        for tx in range(picw-1):
            for tl in range(nprop[ty, tx]):
                tv = proposals[ty, tx, tl]
                neix = tx
                neiy = ty+1
                for qw in range(2):

                    # ................................................sporije
                    # for neil in range(nprop[neiy,neix]):
                    #    neiv=proposals[neiy,neix,neil]
                    #    raz=purepsi(tv[0],tv[1],neiv[0],neiv[1])
                    #    if(raz>bigtpsi):
                    #        neil=5*(neil//5+1)
                    #    elif(raz<tpsi):
                    #        ksets4[qw,tl,neil]=True

                    ksets4[qw, tl, 0:nprop[neiy, neix]] = (tpsi > purepsi(
                        tv[0], tv[1], proposals[neiy, neix, 0:nprop[neiy, neix], 0], proposals[neiy, neix, 0:nprop[neiy, neix], 1]))

                    neiy = ty
                    neix = tx+1
            if (tx % 2 == 0):
                packedksets0[tx//2, ty, :maxnprop*maxnprop] = np.packbits(
                    np.reshape(ksets4[0], maxnprop*maxnprop))
            if (ty % 2 == 0):
                packedksets1[ty//2, (picw-2-tx), :maxnprop*maxnprop] = np.packbits(
                    np.reshape(ksets4[1], maxnprop*maxnprop))
            if (tx % 2 == 1):
                packedksets2[(picw-1-tx)//2, (pich-2-ty), :maxnprop *
                             maxnprop] = np.packbits(np.reshape(ksets4[0], maxnprop*maxnprop))
            if (ty % 2 == 1):
                packedksets3[(pich-1-ty)//2, tx, :maxnprop *
                             maxnprop] = np.packbits(np.reshape(ksets4[1], maxnprop*maxnprop))
            ksets4 = np.zeros((4, maxnprop, maxnprop), dtype=bool)
    ty = pich-1
    for tx in range(picw-1):
        for tl in range(nprop[ty, tx]):
            tv = proposals[ty, tx, tl]
            neix = tx+1
            neiy = ty
            ksets4[1, tl, 0:nprop[neiy, neix]] = (tpsi > purepsi(
                tv[0], tv[1], proposals[neiy, neix, 0:nprop[neiy, neix], 0], proposals[neiy, neix, 0:nprop[neiy, neix], 1]))
        # packedksets[ty,tx,1,:maxnprop*maxnprop]=np.packbits(np.reshape(ksets4[1],maxnprop*maxnprop))
        if (ty % 2 == 0):
            packedksets1[ty//2, (picw-2-tx), :maxnprop*maxnprop] = np.packbits(
                np.reshape(ksets4[1], maxnprop*maxnprop))
        if (ty % 2 == 1):
            packedksets3[(pich-1-ty)//2, tx, :maxnprop *
                         maxnprop] = np.packbits(np.reshape(ksets4[1], maxnprop*maxnprop))
    tx = picw-1
    for ty in range(pich-1):
        for tl in range(nprop[ty, tx]):
            tv = proposals[ty, tx, tl]
            neix = tx
            neiy = ty+1
            ksets4[0, tl, 0:nprop[neiy, neix]] = (tpsi > purepsi(
                tv[0], tv[1], proposals[neiy, neix, 0:nprop[neiy, neix], 0], proposals[neiy, neix, 0:nprop[neiy, neix], 1]))
        # packedksets[ty,tx,0,:maxnprop*maxnprop]=np.packbits(np.reshape(ksets4[0],maxnprop*maxnprop))
        if (tx % 2 == 0):
            packedksets0[tx//2, ty, :maxnprop *
                         maxnprop] = np.packbits(np.reshape(ksets4[0], maxnprop*maxnprop))
        if (tx % 2 == 1):
            packedksets2[(picw-1-tx)//2, (pich-2-ty), :maxnprop *
                         maxnprop] = np.packbits(np.reshape(ksets4[0], maxnprop*maxnprop))
    np.save('Daisy output slike '+picindex+' backward='+backward+' pakovani za c 0', packedksets0)
    np.save('Daisy output slike '+picindex+' backward='+backward+' pakovani za c 1', packedksets1)
    np.save('Daisy output slike '+picindex+' backward='+backward+' pakovani za c 2', packedksets2)
    np.save('Daisy output slike '+picindex+' backward='+backward+' pakovani za c 3', packedksets3)
    ksets4 = np.zeros((4, maxnprop, maxnprop), dtype=bool)
# a