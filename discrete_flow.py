import cv2
import numpy as np
from scipy import spatial as sp
import pyflann as fl
#DAISY deo
flann = fl.FLANN()
pic1 = cv2.imread('C:/Users/JovNov/Desktop/Estimacija Pokreta/slicice/000000_10.png')
pic2 = cv2.imread('C:/Users/JovNov/Desktop/Estimacija Pokreta/slicice/000000_11.png')
#print(pic1)
picw = 400 #valjalo bi da je parno
pich = 150
pic3 = cv2.resize(pic1,(picw,pich))
pic4 = cv2.resize(pic2,(picw,pich))
tphi = 2.5
tpsi=15.0

print(pic3.shape)
#cv2.imshow('r',pic3)
#daisy = cv2.xfeatures2d.DAISY_create([, 5[, 4[, 4[, 4[, norm[, H[, interpolation[, use_orientation]]]]]]]])
daisy = cv2.xfeatures2d.DAISY_create(radius=5,q_radius=4,q_theta=4,q_hist=4)
kp = [cv2.KeyPoint(x, y, 1) for y in range(pich) for x in range(picw)]
print('poc')
descrsold = np.zeros((pich*picw,68))
kp, descrsold = daisy.compute(pic3, kp)
print(descrsold.shape)
descrs1= np.zeros((pich,picw,68),dtype=np.float32)
descrs1 = np.resize(descrsold,((pich,picw,68)))

kp = [cv2.KeyPoint(x, y, 1) for y in range(pich) for x in range(picw)]
kp, descrsold = daisy.compute(pic4, kp)
descrs2 = np.zeros((pich,picw,68),dtype=np.float32)
descrs2 = np.resize(descrsold,((pich,picw,68)))

'''
norms = np.zeros((pich,picw))
for i in range(picw):
    for j in range(pich):
        norms[j,i]=np.sqrt(np.dot(descrs[j,i],descrs[j,i]))

cv2.imshow('d',norms)
cv2.waitKey(0)
'''
print('gotov')

#treba podeliti sliku na celije

cellw = 50
cellh = 50
ncellx = picw//cellw #oko 20
ncelly = pich//cellh #oko 10
#maxflowdist = 100
celldescrs2 = np.zeros((ncellx*ncelly,cellw*cellh,68), dtype=np.float32)
maxnprop = 250
proposals = np.zeros((pich,picw,maxnprop,2), dtype = int)
lcosts = np.full((pich,picw,maxnprop),1000.0)
nprop = np.zeros((pich,picw), dtype=int)
mindists = np.full((pich,picw), 1000.0)
#minvecs=np.zeros((pich,picw,2), dtype=int)
labels=np.zeros((pich,picw), dtype=int)
for tci in range(ncellx):
    for tcj in range(ncelly):
        celldescrs2[tci+tcj*ncellx]=np.resize(descrs2[tcj*cellh:(1+tcj)*cellh,tci*cellw:(1+tci)*cellw],(cellw*cellh,68))
#print(descrs1[123,234])
#print(celldescrs2[3,45])
'''
kdtrees = np.ndarray(ncellx*ncelly, dtype=sp.KDTree)
for i in range(ncellx*ncelly):
    kdtrees[i]=sp.KDTree(celldescrs[i])

#uf

for ci in range(ncellx):
    for cj in range(ncelly):
        
        for ti in range(cellw):
            for tj in range(cellh):
                tx = ti+ci*cellw
                ty=tj+cj*cellh
                td = descrs[ty,tx] #ako ovo ne radi ubaci descrs[ty+tx*cellh]
                for tci in range(np.max(ci-2,0),np.min(ci+3,ncellx)):
                    for tcj in range(np.max(cj-2,0),np.min(cj+3,ncelly)):
                        xx = (kdtrees[tci+tcj*ncellx]).query(td,5)

'''
#napravim kdtree za trenutnu celiju
#uzmem sve piksele u okolini, nadjem K proposala za njih
for ci in range(ncellx):
    for cj in range(ncelly):
        params = flann.build_index(pts=celldescrs2[ci+cj*ncellx])
        #print('bQ',(celldescrs[0]).dtype.type)
        print(params)
        for x in range(max(0,cellw*(ci-2)),min(picw,cellw*(ci+3))):
            for y in range(max(0,cellh*(cj-2)),min(pich,cellh*(cj+3))):
                #print('aQ',(descrs[y,x]).dtype.type)
                res, dists = flann.nn_index(qpts=descrs1[y,x],num_neighbors=5) #k neighbours je 5, i dole je implementirano
                proposals[y,x,nprop[y,x]:nprop[y,x]+5,1]=ci*cellw+res[0]%cellw-x #proposals[,,,0] je y a [,,,1] je x
                proposals[y,x,nprop[y,x]:nprop[y,x]+5,0]=cj*cellh+res[0]//cellw-y
                for qq in range(5):
                    lcosts[y,x,nprop[y,x]+qq]=min(tphi,np.sum(np.absolute(descrs1[y,x]-celldescrs2[ci+cj*ncellx,res[0,qq]]))) #!!!!!!!!
                    if(lcosts[y,x,nprop[y,x]+qq]<mindists[y,x]):
                        mindists[y,x]=lcosts[y,x,nprop[y,x]+qq]
                        #minvecs[y,x]=proposals[y,x,nprop[y,x]+qq]
                        labels[y,x]=nprop[y,x]+qq
                #lcosts[y,x,nprop[y,x]:nprop[y,x]+5]=dists
                #print("afesfadgws", res.shape)
                #for qi in range(5):
                   # if(dists[0,qi]==0.0): print("alo",y,x,cj*cellh+res[qi]//cellw,ci*cellw+res[qi]%cellw)
                nprop[y,x]+=5


print('MINVEC ZA 33 33',labels[33,33],proposals[33,33,labels[33,33]], mindists[33,33])
'''
sortorder = np.zeros(maxnprop,dtype=int)
for x in range(picw):
    for y in range(pich):
        sortorder = np.argsort(lcosts[y,x])
        lcosts=lcosts[:,:,sortorder]
        proposals=proposals[:,:,sortorder,:]
        print('ee!',lcosts[y,x])

print(proposals[33,33]) #Radi !
print(lcosts[33,33])
'''
ngauss = 20
sigma = 5
for x in range(picw):
    for y in range(pich):
        i=0
        while(i<ngauss):
            tgy = int(np.random.normal(y,sigma))
            if(tgy>=0 and tgy<pich):
                tgx = int(np.random.normal(x,sigma))
                if(tgx>=0 and tgx<picw):
                    proposals[y,x,nprop[y,x]]=proposals[tgy,tgx,labels[tgy,tgx]]
                    # proposals[y,x,nprop[y,x]]=minvecs[tgy,tgx]
                    lcosts[y,x,nprop[y,x]]=min(tphi,abs(np.sum(descrs1[y,x]-descrs2[tgy,tgx])))
                    i+=1
                    nprop[y,x]+=1
#nisu unique
#ovako uvek ima 50 random suseda pa su u coskovima slike gusci

print(proposals[33,33]) #Radi !
print(lcosts[33,33])
#pazi, sada su u proposalu vektori a ne destinacije

#treba uzeti minvecove i raditi dinamicko s njima red po red. treba izracunati K(p,p+-1,l)
bcd_times = 10
ystep=0
xstep=0
dp=np.full((2*max(pich,picw),maxnprop),1000.0)

def psi(y1,x1,label1,y2,x2):
    return min(tpsi,np.sum(np.abs(proposals[y1,x1,label1]-proposals[y2,x2,labels[y2,x2]])))

def bcd(ystep,xstep,ty,tx):

    return 1
for w in range(bcd_times):
    for xloc in range(0,picw,2):
        bcd(1,0,0,xloc)
    for yloc in range(0,pich,2):
        bcd(0,-1,yloc,picw-1)
    for xloc in range(picw-1,-1,-2):
        bcd(-1,0,pich-1,xloc)
    for yloc in range(pich-1,-1,-2):
        bcd(0,1,yloc,0)

'''
ovo vise ne radi zbog reshape
for i in range(500000):
    norms[i]=np.linalg.norm(descrs[i])
print(norms.shape)
pic5 = np.resize(norms,((500,1000)))
cv2.imshow('q',pic5)
cv2.waitKey(0)
'''