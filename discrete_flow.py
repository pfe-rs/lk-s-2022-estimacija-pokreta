import cv2
import numpy as np
from scipy import spatial as sp
import pyflann as fl
#DAISY deo
flann = fl.FLANN()

'''
pic1 = cv2.imread('../data_scene_flow/testing/image_2/000000_10.png') !!!!!!!!!!OVO JE TACNO
pic2 = cv2.imread('../data_scene_flow/testing/image_2/000000_11.png')
''' 
pic1 = cv2.imread('C:/Users/JovNov/Desktop/Estimacija Pokreta/slicice/000000_10.png')
pic2 = cv2.imread('C:/Users/JovNov/Desktop/Estimacija Pokreta/slicice/000000_10.png')

#print(pic1)
picw = 400 #valjalo bi da je parno
pich = 150
pic3 = cv2.resize(pic1,(picw,pich))
pic4 = cv2.resize(pic2,(picw,pich))
tphi = 0.5
tpsi=5


#print(pic3.shape)
#cv2.imshow('r',pic3)
#daisy = cv2.xfeatures2d.DAISY_create([, 5[, 4[, 4[, 4[, norm[, H[, interpolation[, use_orientation]]]]]]]])
daisy = cv2.xfeatures2d.DAISY_create(radius=5,q_radius=4,q_theta=4,q_hist=4)
kp = [cv2.KeyPoint(x, y, 1) for y in range(pich) for x in range(picw)]
print('poc')
descrsold = np.zeros((pich*picw,68))
kp, descrsold = daisy.compute(pic3, kp)
print(descrsold.shape)
descrs1= np.zeros((pich,picw,68),dtype=np.float32)
descrs1 = np.reshape(descrsold,((pich,picw,68)))

kp = [cv2.KeyPoint(x, y, 1) for y in range(pich) for x in range(picw)]
kp, descrsold = daisy.compute(pic4, kp)
descrs2 = np.zeros((pich,picw,68),dtype=np.float32)
descrs2 = np.reshape(descrsold,((pich,picw,68)))



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

cellw = 25
cellh = 25
ncellx = picw//cellw #oko 20
ncelly = pich//cellh #oko 10
#maxflowdist = 100
celldescrs2 = np.zeros((ncellx*ncelly,cellw*cellh,68), dtype=np.float32)
maxnprop = 150
proposals = np.full((pich,picw,maxnprop,2),-1, dtype = int)
lcosts = np.full((pich,picw,maxnprop),1000.0)
nprop = np.zeros((pich,picw), dtype=int)
ngaussprop=np.zeros((pich,picw),dtype=int)
mindists = np.full((pich,picw), 1000.0)
#minvecs=np.zeros((pich,picw,2), dtype=int)
labels=np.zeros((pich,picw), dtype=int)

def check1(yl,xl,yv,xv):
    
    #cellxl=
    mincellyl=max(0,yl//cellh-2)
    #maxcellyl=
    ncellyl=min(ncelly, yl//cellh+2)-mincellyl
    mincellxl=max(0,xl//cellw-2)
    #maxcellxl=min(ncellx, xl//cellw+2)
    
    broj=((yv//cellh-mincellyl)+(xv//cellw-mincellxl)*ncellyl)
    return 5*broj
    #sad proveri broj*5:(broj+1)*5
    #sem toga proveri i dole RIP

print(check1(23,50,66,1))

for tci in range(ncellx):
    for tcj in range(ncelly):
        celldescrs2[tci+tcj*ncellx]=np.reshape(descrs2[tcj*cellh:(1+tcj)*cellh,tci*cellw:(1+tci)*cellw],(cellw*cellh,68))
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
ngauss = 25
sigma = 5
for x in range(picw):
    for y in range(pich):
        
        mincellyl=max(0,y//cellh-2)
        #maxcellyl=
        ncellyl=min(ncelly, y//cellh+2)-mincellyl
        mincellxl=max(0,x//cellw-2)

        i=0
        while(i<ngauss):
            tgy = int(np.random.normal(y,sigma))
            if(tgy>=0 and tgy<pich):
                tgx = int(np.random.normal(x,sigma))
                if(tgx>=0 and tgx<picw):
                    broj=5*((tgy//cellh-mincellyl)+(tgx//cellw-mincellxl)*ncellyl)
                    tv=proposals[tgy,tgx,labels[tgy,tgx]]
                    if((tv not in proposals[y,x,broj:broj+5]) and (tv not in proposals[y,x,(nprop[y,x]-ngaussprop[y,x]):nprop[y,x]])):
                        proposals[y,x,nprop[y,x]]=tv
                        nprop[y,x]+=1
                        ngaussprop[y,x]+=1
                        lcosts[y,x,nprop[y,x]]=min(tphi,abs(np.sum(descrs1[y,x]-descrs2[tgy,tgx])))
                    
                    i+=1
                    
#nisu unique
#ovako uvek ima 50 random suseda pa su u coskovima slike gusci

print(proposals[50,332]) #Radi !
print(lcosts[50,332])
for y in range(pich):
    for x in range(picw):
        print(y,x,ngaussprop[y,x])
#pazi, sada su u proposalu vektori a ne destinacije

def psi(y1,x1,label1,y2,x2):
    return min(tpsi,np.sum(np.abs(proposals[y1,x1,label1]-proposals[y2,x2,labels[y2,x2]]))) #treba refinisati

def purepsi(yv1,xv1,yv2,xv2):
    return np.abs(yv2-yv1)+np.abs(xv2-xv1)


#treba uzeti minvecove i raditi dinamicko s njima red po red. treba izracunati K(p,p+-1,l)
bcd_times = 10
ystep=0
xstep=0
bigtpsi=tpsi+cellw+cellh
dp=np.full((2*max(pich,picw),maxnprop),1000.0)
kdim=maxnprop*maxnprop//8+1
packedksets=np.zeros((pich-1,picw-1,2,kdim),dtype=np.uint8) #cuva labele, ne psi
ksets4=np.zeros((4,maxnprop,maxnprop),dtype=bool) #gornji za mene, levi za mene, moj za donji, moj za desni (smrt)
tv=np.zeros(2,dtype=int)
for ty in range(pich-1):
    print('pocinjem red y=',ty)
    for tx in range(picw-1):
        for tl in range(nprop[ty,tx]):
            tv = proposals[ty,tx,tl]
            neix=tx
            neiy=ty+1
            for qw in range(2):
                for neil in range(nprop[neiy,neix]):
                    neiv=proposals[neiy,neix,neil]
                    raz=purepsi(tv[0],tv[1],neiv[0],neiv[1])
                    if(raz>bigtpsi):
                        neil=5*(neil//5+1)
                    elif(raz<tpsi):
                        ksets4[qw,tl,neil]=True
                neiy=ty
                neix=tx+1
        packedksets[ty,tx,0]=np.packbits(ksets4[0])
        packedksets[ty,tx,1]=np.packbits(ksets4[1])
        ksets4=np.zeros((4,maxnprop,maxnprop),dtype=bool)
#a
np.save('pakovani',packedksets)
'''
def bcd(ystep,xstep,ty,tx):
    if(ystep==0):
        xside=1
        yside=0
    else:
        xside=0
        yside=1
    i = 0
    dp = np.full((2*max(pich,picw),maxnprop),1000.0)
    dp[0,0:nprop[ty,tx]]=psi(ty,tx,labels[0:nprop[ty,tx]], ty+yside, tx+xside) + psi(ty,tx,labels[0:nprop[ty,tx]], ty-yside, tx-xside) + lcosts[ty,tx,0:nprop[ty,tx]]
    
    while(True):
        
        ty+=ystep
        tx+=xstep
        if(tx<0 or ty<0 or tx>=picw or ty>=pich): break
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
'''
ovo vise ne radi zbog reshape
for i in range(500000):
    norms[i]=np.linalg.norm(descrs[i])
print(norms.shape)
pic5 = np.reshape(norms,((500,1000)))
cv2.imshow('q',pic5)
cv2.waitKey(0)
'''