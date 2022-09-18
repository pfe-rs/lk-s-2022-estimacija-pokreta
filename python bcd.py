from operator import truediv
from struct import pack
import cv2
import numpy as np
from scipy import spatial as sp
import pyflann as fl
import datetime as dt
import sys
# DAISY deo

print(dt.datetime.now())

picindex=sys.argv[1]
backward=sys.argv[2]
bcd_times=int(sys.argv[3])
if(len(picindex)==1):
    picindex='0'+picindex
pastw=0

# print(pic1)
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


def ucitajSvePodatkeDoBCD():
    global proposals
    global lcosts
    global nprop
    global packedksets
    global bestlabels
    proposals = np.load('Daisy output slike '+picindex+' backward='+backward+' proposals_nakon_gausa.npy')
    lcosts = np.load('Daisy output slike '+picindex+' backward='+backward+' lcosts_nakon_gausa.npy')
    nprop = np.load('Daisy output slike '+picindex+' backward='+backward+' nprop.npy')
    packedksets = np.load('Daisy output slike '+picindex+' backward='+backward+' packedksets.npy')
    #bestlabels=np.load('Dobri fajlovi2_b/bestlabels.npy')
    pastwstr=str(pastw)
    if(len(pastwstr)==1):
        pastwstr='0'+pastwstr
    bestlabels = np.load('Bestlabels fajl slike '+picindex+' backward='+backward+' posle '+pastwstr+' BCD.npy')


def sidepsi(y1, x1, label1, y2, x2):
    if (y2 >= 0 and y2 < pich and x2 >= 0 and x2 < picw):
        # treba refinisati
        return min(tpsi, np.sum(np.abs(proposals[y1, x1, label1]-proposals[y2, x2, bestlabels[y2, x2]])))
    return 0

def vratiKonacniFlow():
    finalpic = np.zeros((pich, picw, 2))
    for ty in range(pich):
        for tx in range(picw):
            finalpic[ty, tx] = proposals[ty, tx, bestlabels[ty, tx]]
    return finalpic


def purepsi(yv1, xv1, yv2, xv2):
    return np.abs(yv2-yv1)+np.abs(xv2-xv1)

def bcd(ystep, xstep, ty, tx):
    global unpacktime
    global truestime
    global bestlabels
    blabels = bestlabels
    minarr = np.zeros(50, dtype=int)
    trues = np.nonzero([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    if (ystep == 0):
        xside = 1
        yside = 0
    else:
        xside = 0
        yside = 1
    i = 0
    dp = (np.full((2*max(pich, picw), maxnprop), 1000.0))
    pastlabels = (np.full((2*max(pich, picw), maxnprop+1),
                  1000, dtype=int)).tolist()
    for tl in range(nprop[ty, tx]):
        dp[0, tl] = sidepsi(ty, tx, tl, ty+yside, tx+xside) + \
            sidepsi(ty, tx, tl, ty-yside, tx-xside) + lcosts[ty, tx, tl]
    while (True):
        # ksets je prosli u DP
        #
        ty += ystep
        tx += xstep
        i += 1
        if (tx < 0 or ty < 0 or tx >= picw or ty >= pich):
            break
        #if(i==6): print('e 6')
        t1 = dt.datetime.now()
        if (ystep == -1 and xstep == 0):
            ksets = np.reshape(np.unpackbits(packedksets[ty, tx, 0])[
                               :maxnprop*maxnprop], (maxnprop, maxnprop))
        elif (ystep == 1 and xstep == 0):
            ksets = np.reshape(np.unpackbits(
                packedksets[ty-1, tx, 0])[:maxnprop*maxnprop], (maxnprop, maxnprop))
        elif (ystep == 0 and xstep == 1):
            ksets = np.reshape(np.unpackbits(
                packedksets[ty, tx-1, 1])[:maxnprop*maxnprop], (maxnprop, maxnprop))
        else:
            ksets = np.reshape(np.unpackbits(packedksets[ty, tx, 1])[
                               :maxnprop*maxnprop], (maxnprop, maxnprop))
            #dp[i,0:nprop[ty,tx]]=np.max(dp[i-1,0:nprop[ty-ystep,tx-xstep]]+np.where(ksets4[0,0:nprop[ty-ystep,tx-xstep],0:nprop[ty,tx]],  psi(ty-ystep,tx-xstep, uf-ovo-nije-sidepsi, ty,tx), tpsi))
        t2 = dt.datetime.now()
        delta = t2-t1
        unpacktime = unpacktime + delta.microseconds/1000000.0

        minc = 10000.0
        tnprop = nprop[ty, tx]
        pnprop = nprop[ty-ystep, tx-xstep]
        #psicosts=np.where(ksets4[2,:tnprop,:pnprop], sidepsi(ty,tx,mat1[:tnprop,:pnprop],ty,tx+1),tpsi)
        permmincost = 800000.0
        permminlabel = -8
        for tk in range(pnprop):
            if (tpsi+dp[i-1, tk] < permmincost):
                permmincost = tpsi+dp[i-1, tk]
                permminlabel = tk

        if (ystep == -1 or xstep == -1):
            for tl in range(tnprop):
                smallcosts = lamda*lcosts[ty, tx, tl] + sidepsi(ty, tx, tl, ty+yside, tx+xside) + sidepsi(
                    ty, tx, tl, ty-yside, tx-xside)  # postoji li ?
                mincost = permmincost
                af, bf = proposals[ty, tx, tl, 0], proposals[ty, tx, tl, 1]
                pastlabels[i][tl] = permminlabel

                #myb[:pnprop]=np.where(ksets[tl,:pnprop], dp[i-1,:pnprop]+purepsi(af,bf,proposals[ty-ystep,tx-xstep,:pnprop,0],proposals[ty-ystep,tx-xstep,:pnprop,1]), mincost)
                # dp[i,tl]=np.min(myb[:pnprop])+smallcosts

                trues = np.nonzero(ksets[tl, :pnprop])
                if (trues[0].size > 0):
                    minarr = (dp[i-1, trues[0]]+purepsi(af, bf, proposals[ty-ystep, tx -
                              xstep, trues[0], 0], proposals[ty-ystep, tx-xstep, trues[0], 1]))
                    mincost = np.min(minarr)
                    pastlabels[i][tl] = trues[0][np.argmin(minarr)]
                dp[i, tl] = mincost+smallcosts

                # dp[i,tl]=mincost+smallcosts

                # if(ty==50):
                #    print('oblik',np.shape(trues))

                # dp[i,tl]=mincost+smallcosts
                # nebitno:
                # psicosts=np.full((nprop[ty-1,tx],nprop[ty,tx]),tpsi)
                # ovo je vrv tacno:
                # psicosts=np.where(ksets4[0,:anprop,:bnprop],purepsi(proposals[ty-1,tx,mat1[:anprop,:bnprop],0],proposals[ty-1,tx,mat1[:anprop,:bnprop],1],proposals[ty,tx,mat2[:anprop,:bnprop],0],proposals[ty,tx,mat2[:anprop,:bnprop],1]),tpsi)
                # BUDI JAKO PAZLJIV SVE OVDE MOZE BITI NETACNO
                # for tl in range(nprop[ty-1,tx]):
                #    for tk in range(nprop[ty,tx]):
                #        if(ksets4[0,tl,tk]):\

                #            psicosts[tl,tk]=purepsi(proposals[ty-1,tx,tl,0],proposals[ty-1,tx,tl,1],proposals[ty,tx,tk,0],proposals[ty,tx,tk,1])
        else:
            for tl in range(tnprop):
                smallcosts = lamda*lcosts[ty, tx, tl] + sidepsi(ty, tx, tl, ty+yside, tx+xside) + sidepsi(
                    ty, tx, tl, ty-yside, tx-xside)  # postoji li ?
                mincost = permmincost
                af, bf = proposals[ty, tx, tl, 0], proposals[ty, tx, tl, 1]
                pastlabels[i][tl] = permminlabel

                #myb[:pnprop]=np.where(ksets[:pnprop,tl], dp[i-1,:pnprop]+purepsi(af,bf,proposals[ty-ystep,tx-xstep,:pnprop,0],proposals[ty-ystep,tx-xstep,:pnprop,1]), mincost)
                # dp[i,tl]=np.min(myb[:pnprop])+smallcosts

                '''
                for tk in range(pnprop):
                    if(ksets[tk,tl]):
                        mybcost=dp[i-1,tk]+purepsi(af,bf,proposals[ty-ystep,tx-xstep,tk,0],proposals[ty-ystep,tx-xstep,tk,1])
                        if(mybcost<mincost):
                            mincost=mybcost
                            pastlabels[i][tl]=tk
                '''
                trues = np.nonzero(ksets[:pnprop, tl])
                if (trues[0].size > 0):
                    minarr = (dp[i-1, trues[0]]+purepsi(af, bf, proposals[ty-ystep, tx -
                              xstep, trues[0], 0], proposals[ty-ystep, tx-xstep, trues[0], 1]))
                    mincost = np.min(minarr)
                    pastlabels[i][tl] = trues[0][np.argmin(minarr)]
                dp[i, tl] = mincost+smallcosts

                # RESETAVAJ TRUES

                #if(tl==0): print('oblik',np.shape(trues))
                # dp[i,tl]=mincost+smallcosts

    # sad rekonstrukcija
    # uzmem min dp[posl-1]
    ty -= ystep
    tx -= xstep
    i -= 1
    mincost = 800000.0
    minlabel = 0
    for tl in range(nprop[ty, tx]):
        if (dp[i, tl] < mincost):
            mincost = dp[i, tl]
            minlabel = tl
    blabels[ty, tx] = minlabel
    pl = minlabel
    while (True):
        ty -= ystep
        tx -= xstep
        if (i < 0):
            print('i je manje od 0', ty, tx, i, pl)
        if (pl == 1000):
            print('pl je 1000', ty, tx, i, pl)
            np.save('error bestlabels', bestlabels)
            np.save('error pastlabels', pastlabels)
            np.save('error dp', dp)
        if (tx < 0 or ty < 0 or tx >= picw or ty >= pich):
            break
        pl = pastlabels[i][pl]
        i -= 1
        blabels[ty, tx] = pl
    # print(bestlabels)
    # print(pastlabels)

    return blabels
# print(bestlabels[35])


def ceoBCD():
    global bestlabels
    global pastw
    for w in range(1,bcd_times+1):
        for xloc in range(0, picw, 2):
            bestlabels = bcd(1, 0, 0, xloc)
            # print('a')
        # print(nprop[50])
        print(bestlabels[35])
        for yloc in range(0, pich, 2):
            bestlabels = bcd(0, -1, yloc, picw-1)
        print('f')
        for xloc in range((picw//2)*2-1, -1, -2):
            bestlabels = bcd(-1, 0, pich-1, xloc)
        print('g')
        for yloc in range((pich//2)*2-1, -1, -2):
            bestlabels = bcd(0, 1, yloc, 0)
        pastw=w
        wstr=str(w)
        if(len(wstr)==1):
            wstr='0'+wstr
        np.save('Gotova flow slika '+picindex+' backward='+backward+' posle '+wstr+' BCD.npy', vratiKonacniFlow())
        np.save('Bestlabels fajl slike '+picindex+' backward='+backward+' posle '+wstr+' BCD.npy', bestlabels)
        print('uradjen bcd broj', w)


def sracunajBestlabelsKadNemas():
    for y in range(pich):
        for x in range(picw):
            bestlabels[y, x] = np.argmin(lcosts[y, x, :nprop[y, x]])
    return bestlabels
ucitajSvePodatkeDoBCD()
ceoBCD()