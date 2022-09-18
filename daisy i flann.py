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

flann = fl.FLANN()

picindex=sys.argv[1]
backward=sys.argv[2] #u konzoli, 0 znaci forward, 1 znaci backward
dopython=(sys.argv[3]=='1') #dopython = 1 saveuje za BCD u pythonu; = 0 saveuje za BCD u C
if(backward==0):
    pic2str='1'
else:
    pic2str='0'

if(len(picindex)==1):
    picindex='0'+picindex
pic1 = cv2.imread('../data_scene_flow/training/image_2/0000'+picindex+'_1'+backward+'.png')
pic2 = cv2.imread('../data_scene_flow/training/image_2/0000'+picindex+'_1'+pic2str+'.png')

''' 
pic1 = cv2.imread('C:/Users/JovNov/Desktop/Estimacija Pokreta/slicice/A.png')
pic2 = cv2.imread('C:/Users/JovNov/Desktop/Estimacija Pokreta/slicice/B.png')
'''
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

pic3 = pic1[y:y+pich, x:x+picw, :]
pic4 = pic2[y:y+pich, x:x+picw, :]

#
#testdp = (np.full((2*max(pich,picw),150),1000.0)).tolist()
# print(testdp[0][1])
# print(np.where(True,testdp[0][0:150],-1))
# finalpic=np.zeros((pich,picw,2))
#finalpic = np.load('bebaflow.npy')
# print(finalpic[50])

# print(pic3.shape)
# cv2.imshow('r',pic3)
# daisy = cv2.xfeatures2d.DAISY_create([, 5[, 4[, 4[, 4[, norm[, H[, interpolation[, use_orientation]]]]]]]])
daisy = cv2.xfeatures2d.DAISY_create(radius=5, q_radius=4, q_theta=4, q_hist=4)


def izracunajDaisy(picture):
    kp = [cv2.KeyPoint(x, y, 1) for y in range(pich) for x in range(picw)]
    print('poc')
    descrsold = np.zeros((pich*picw, 68))
    kp, descrsold = daisy.compute(picture, kp)
    print(descrsold.shape)
    descrs1 = np.zeros((pich, picw, 68), dtype=np.float32)
    descrs1 = np.reshape(descrsold, ((pich, picw, 68)))
    return descrs1


descrs1 = np.zeros((pich, picw, 68), dtype=np.float32)
descrs2 = np.zeros((pich, picw, 68), dtype=np.float32)
bcd_times = 5
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


# TEST:

#print(purepsi(np.zeros(8), np.arange(8),np.arange(8),np.full(8,8)))


# treba podeliti sliku na celije
def check1(yl, xl, yv, xv):

    # cellxl=
    mincellyl = max(0, yl//cellh-2)
    # maxcellyl=
    ncellyl = min(ncelly, yl//cellh+2)-mincellyl
    mincellxl = max(0, xl//cellw-2)
    #maxcellxl=min(ncellx, xl//cellw+2)

    broj = ((yv//cellh-mincellyl)+(xv//cellw-mincellxl)*ncellyl)
    return 5*broj
    # sad proveri broj*5:(broj+1)*5
    # sem toga proveri i dole RIP

# print(check1(23,50,66,1))


celldescrs2 = np.zeros((ncellx*ncelly, cellw*cellh, 68), dtype=np.float32)


def napraviCD2():
    for tci in range(ncellx):
        for tcj in range(ncelly):
            celldescrs2[tci+tcj*ncellx] = np.reshape(
                descrs2[tcj*cellh:(1+tcj)*cellh, tci*cellw:(1+tci)*cellw], (cellw*cellh, 68))

# print(descrs1[123,234])
# print(celldescrs2[3,45])

# napravim kdtree za trenutnu celiju
# uzmem sve piksele u okolini, nadjem K proposala za njih


def generisi():
    global proposals
    global lcosts
    global bestlabels
    global nprop
    for ci in range(ncellx):
        for cj in range(ncelly):
            params = flann.build_index(pts=celldescrs2[ci+cj*ncellx])
            # print('bQ',(celldescrs[0]).dtype.type)
            # print(params)
            for x in range(max(0, cellw*(ci-2)), min(picw, cellw*(ci+3))):
                for y in range(max(0, cellh*(cj-2)), min(pich, cellh*(cj+3))):
                    # print('aQ',(descrs[y,x]).dtype.type)
                    # k neighbours je 5, i dole je implementirano
                    res, dists = flann.nn_index(
                        qpts=descrs1[y, x], num_neighbors=5)
                    # proposals[,,,0] je y a [,,,1] je x
                    proposals[y, x, nprop[y, x]:nprop[y, x] +
                              5, 1] = ci*cellw+res[0] % cellw-x
                    proposals[y, x, nprop[y, x]:nprop[y, x] +
                              5, 0] = cj*cellh+res[0]//cellw-y
                    for qq in range(5):
                        lcosts[y, x, nprop[y, x]+qq] = min(tphi, np.sum(np.absolute(
                            descrs1[y, x]-celldescrs2[ci+cj*ncellx, res[0, qq]])))  # !!!!!!!!
                        if (lcosts[y, x, nprop[y, x]+qq] < mindists[y, x]):
                            mindists[y, x] = lcosts[y, x, nprop[y, x]+qq]
                            # minvecs[y,x]=proposals[y,x,nprop[y,x]+qq]
                            bestlabels[y, x] = nprop[y, x]+qq
                    # lcosts[y,x,nprop[y,x]:nprop[y,x]+5]=dists
                    #print("afesfadgws", res.shape)
                    # for qi in range(5):
                    # if(dists[0,qi]==0.0): print("alo",y,x,cj*cellh+res[qi]//cellw,ci*cellw+res[qi]%cellw)
                    nprop[y, x] += 5


def vratiKonacniFlow():
    finalpic = np.zeros((pich, picw, 2))
    for ty in range(pich):
        for tx in range(picw):
            finalpic[ty, tx] = proposals[ty, tx, bestlabels[ty, tx]]
    return finalpic


def sacuvajPodatke0():
    np.save('Gotova flow slika '+picindex+' backward='+backward+' posle 00 BCD.npy', vratiKonacniFlow())
    np.save('Bestlabels fajl slike '+picindex+' backward='+backward+' posle 00 BCD.npy', bestlabels)


def nasumicni():
    ngaussprop = np.zeros((pich, picw), dtype=int)
    ngauss = 25
    sigma = 8
    for x in range(picw):
        for y in range(pich):

            mincellyl = max(0, y//cellh-2)
            # maxcellyl=
            ncellyl = min(ncelly, y//cellh+2)-mincellyl
            mincellxl = max(0, x//cellw-2)

            i = 0
            while (i < ngauss):
                tgy = int(np.random.normal(y, sigma))
                if (tgy >= 0 and tgy < pich):
                    tgx = int(np.random.normal(x, sigma))
                    if (tgx >= 0 and tgx < picw):
                        broj = 5*((tgy//cellh-mincellyl) +
                                  (tgx//cellw-mincellxl)*ncellyl)
                        tv = proposals[tgy, tgx, bestlabels[tgy, tgx]]
                        if ((tv not in proposals[y, x, broj:broj+5]) and (tv not in proposals[y, x, (nprop[y, x]-ngaussprop[y, x]):nprop[y, x]])):
                            proposals[y, x, nprop[y, x]] = tv
                            lcosts[y, x, nprop[y, x]] = min(
                                tphi, abs(np.sum(descrs1[y, x]-descrs2[tgy, tgx])))
                            nprop[y, x] += 1
                            ngaussprop[y, x] += 1

                        i += 1


# nisu unique
# ovako uvek ima 50 random suseda pa su u coskovima slike gusci


# for y in range(pich):
#    for x in range(picw):
#        print(y,x,ngaussprop[y,x])
# pazi, sada su u proposalu vektori a ne destinacije


# treba uzeti minvecove i raditi dinamicko s njima red po red. treba izracunati K(p,p+-1,l)


def sacuvajPodatke1():
    #np.save('Daisy output slike '+picindex+' backward='+backward+'/Flow posle 0 bcd', vratiKonacniFlow())
    np.save('Daisy output slike '+picindex+' backward='+backward+' proposals_nakon_gausa.npy', proposals)
    np.save('Daisy output slike '+picindex+' backward='+backward+' lcosts_nakon_gausa.npy', lcosts)
    np.save('Daisy output slike '+picindex+' backward='+backward+' nprop.npy', nprop)


def pakovanje():
    global packedksets
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
            packedksets[ty, tx, 0, :maxnprop *
                        maxnprop] = np.packbits(np.reshape(ksets4[0], maxnprop*maxnprop))
            packedksets[ty, tx, 1, :maxnprop *
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
        packedksets[ty, tx, 1, :maxnprop *
                    maxnprop] = np.packbits(np.reshape(ksets4[1], maxnprop*maxnprop))
    tx = picw-1
    for ty in range(pich-1):
        for tl in range(nprop[ty, tx]):
            tv = proposals[ty, tx, tl]
            neix = tx
            neiy = ty+1
            ksets4[0, tl, 0:nprop[neiy, neix]] = (tpsi > purepsi(
                tv[0], tv[1], proposals[neiy, neix, 0:nprop[neiy, neix], 0], proposals[neiy, neix, 0:nprop[neiy, neix], 1]))
        packedksets[ty, tx, 0, :maxnprop *
                    maxnprop] = np.packbits(np.reshape(ksets4[0], maxnprop*maxnprop))
    np.save('Daisy output slike '+picindex+' backward='+backward+' packedksets.npy', packedksets)
    ksets4 = np.zeros((4, maxnprop, maxnprop), dtype=bool)

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


# a

print(dt.datetime.now())

descrs1 = izracunajDaisy(pic3)
descrs2 = izracunajDaisy(pic4)

napraviCD2()
print('napravio')
print(dt.datetime.now())
generisi()
print('generisao')
print(dt.datetime.now())
sacuvajPodatke0()
print('sacuvao0')
print(dt.datetime.now())
nasumicni()
print('i ovo')
print(dt.datetime.now())

sacuvajPodatke1()
print('sacuvao1')
print(dt.datetime.now())
if(not dopython):
    pakovanjeZaC()
    print('spakovao za C')
    print(dt.datetime.now())
if(dopython):
    pakovanje()
    print('spakovao za python')
    print(dt.datetime.now())
#packedksets = np.load('pakovani3.npy')

