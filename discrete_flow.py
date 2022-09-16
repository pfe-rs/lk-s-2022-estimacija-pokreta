from operator import truediv
from struct import pack
import cv2
import numpy as np
from scipy import spatial as sp
import pyflann as fl
import datetime as dt
#DAISY deo

print(dt.datetime.now())

flann = fl.FLANN()


pic1 = cv2.imread('../data_scene_flow/training/image_2/000000_10.png') 
pic2 = cv2.imread('../data_scene_flow/training/image_2/000000_11.png')
''' 
pic1 = cv2.imread('C:/Users/JovNov/Desktop/Estimacija Pokreta/slicice/A.png')
pic2 = cv2.imread('C:/Users/JovNov/Desktop/Estimacija Pokreta/slicice/B.png')
'''
#print(pic1)
picw = 1242
pich = 375
#picw=240
#pich=125
#x = 700
x = 0
#y = 122
y = 0
cellw = 54
cellh = 25
#cellw=50
#cellh=25
tphi = 2.5
tpsi=8
lamda=0.05
unpacktime=0.0
truestime=0.0

pic3 = pic1[y:y+pich, x:x+picw, :]
pic4 = pic2[y:y+pich, x:x+picw, :]

print('a')
#
#testdp = (np.full((2*max(pich,picw),150),1000.0)).tolist()
#print(testdp[0][1])
#print(np.where(True,testdp[0][0:150],-1))
#finalpic=np.zeros((pich,picw,2))
#finalpic = np.load('bebaflow.npy')
#print(finalpic[50])

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


def sidepsi(y1,x1,label1,y2,x2):
    if(y2>=0 and y2<pich and x2>=0 and x2<picw):
        return min(tpsi,np.sum(np.abs(proposals[y1,x1,label1]-proposals[y2,x2,bestlabels[y2,x2]]))) #treba refinisati
    return 0

def purepsi(yv1,xv1,yv2,xv2):
    return np.abs(yv2-yv1)+np.abs(xv2-xv1)


#TEST:

#print(purepsi(np.zeros(8), np.arange(8),np.arange(8),np.full(8,8)))



#treba podeliti sliku na celije

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
bestlabels=np.zeros((pich,picw), dtype=int)

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

#print(check1(23,50,66,1))

for tci in range(ncellx):
    for tcj in range(ncelly):
        celldescrs2[tci+tcj*ncellx]=np.reshape(descrs2[tcj*cellh:(1+tcj)*cellh,tci*cellw:(1+tci)*cellw],(cellw*cellh,68))
#print(descrs1[123,234])
#print(celldescrs2[3,45])

#napravim kdtree za trenutnu celiju
#uzmem sve piksele u okolini, nadjem K proposala za njih
for ci in range(ncellx):
    for cj in range(ncelly):
        params = flann.build_index(pts=celldescrs2[ci+cj*ncellx])
        #print('bQ',(celldescrs[0]).dtype.type)
        #print(params)
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
                        bestlabels[y,x]=nprop[y,x]+qq
                #lcosts[y,x,nprop[y,x]:nprop[y,x]+5]=dists
                #print("afesfadgws", res.shape)
                #for qi in range(5):
                   # if(dists[0,qi]==0.0): print("alo",y,x,cj*cellh+res[qi]//cellw,ci*cellw+res[qi]%cellw)
                nprop[y,x]+=5


np.save('veliki_baby_flow',bestlabels)

print('MINVEC ZA 33 33',bestlabels[33,33],proposals[33,33,bestlabels[33,33]], mindists[33,33])

ngauss = 25
sigma = 8
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
                    tv=proposals[tgy,tgx,bestlabels[tgy,tgx]]
                    if((tv not in proposals[y,x,broj:broj+5]) and (tv not in proposals[y,x,(nprop[y,x]-ngaussprop[y,x]):nprop[y,x]])):
                        proposals[y,x,nprop[y,x]]=tv
                        lcosts[y,x,nprop[y,x]]=min(tphi,abs(np.sum(descrs1[y,x]-descrs2[tgy,tgx])))
                        nprop[y,x]+=1
                        ngaussprop[y,x]+=1
                    
                    i+=1
                    
#nisu unique
#ovako uvek ima 50 random suseda pa su u coskovima slike gusci

print(proposals[35,11]) #Radi !
print(lcosts[35,11])
#for y in range(pich):
#    for x in range(picw):
#        print(y,x,ngaussprop[y,x])
#pazi, sada su u proposalu vektori a ne destinacije


#treba uzeti minvecove i raditi dinamicko s njima red po red. treba izracunati K(p,p+-1,l)
bcd_times = 1
ystep=0
xstep=0
bigtpsi=tpsi+cellw+cellh

kdim=maxnprop*maxnprop//8+1
packedksets=np.zeros((pich,picw,2,kdim),dtype=np.uint8) #cuva labele, ne psi
ksets4=np.zeros((4,maxnprop,maxnprop),dtype=bool) #gornji za mene, levi za mene, moj za donji, moj za desni (smrt)
ksets=np.zeros((maxnprop,maxnprop),dtype=bool)
tv=np.zeros(2,dtype=int)
myb=np.zeros(maxnprop)

finalpic=np.zeros((pich,picw,2))
for ty in range(pich):
    for tx in range(picw):
        finalpic[ty,tx]=proposals[ty,tx,bestlabels[ty,tx]]

np.save('bebaflow',finalpic)



for ty in range(pich-1):
    print('pocinjem red y=',ty)
    for tx in range(picw-1):
        for tl in range(nprop[ty,tx]):
            tv = proposals[ty,tx,tl]
            neix=tx
            neiy=ty+1
            for qw in range(2):
                
                #................................................sporije
                #for neil in range(nprop[neiy,neix]):
                #    neiv=proposals[neiy,neix,neil]
                #    raz=purepsi(tv[0],tv[1],neiv[0],neiv[1])
                #    if(raz>bigtpsi):
                #        neil=5*(neil//5+1)
                #    elif(raz<tpsi):
                #        ksets4[qw,tl,neil]=True
                
                ksets4[qw,tl,0:nprop[neiy,neix]]=(tpsi>purepsi(tv[0],tv[1],proposals[neiy,neix,0:nprop[neiy,neix],0],proposals[neiy,neix,0:nprop[neiy,neix],1]))


                neiy=ty
                neix=tx+1
        packedksets[ty,tx,0,:maxnprop*maxnprop]=np.packbits(np.reshape(ksets4[0],maxnprop*maxnprop))
        packedksets[ty,tx,1,:maxnprop*maxnprop]=np.packbits(np.reshape(ksets4[1],maxnprop*maxnprop))
        ksets4=np.zeros((4,maxnprop,maxnprop),dtype=bool)
ty=pich-1
for tx in range(picw-1):
    for tl in range(nprop[ty,tx]):
        tv = proposals[ty,tx,tl]
        neix=tx+1
        neiy=ty
        ksets4[1,tl,0:nprop[neiy,neix]]=(tpsi>purepsi(tv[0],tv[1],proposals[neiy,neix,0:nprop[neiy,neix],0],proposals[neiy,neix,0:nprop[neiy,neix],1]))
    packedksets[ty,tx,1,:maxnprop*maxnprop]=np.packbits(np.reshape(ksets4[1],maxnprop*maxnprop))
tx=picw-1
for ty in range(pich-1):
    for tl in range(nprop[ty,tx]):
        tv = proposals[ty,tx,tl]
        neix=tx
        neiy=ty+1
        ksets4[0,tl,0:nprop[neiy,neix]]=(tpsi>purepsi(tv[0],tv[1],proposals[neiy,neix,0:nprop[neiy,neix],0],proposals[neiy,neix,0:nprop[neiy,neix],1]))
    packedksets[ty,tx,0,:maxnprop*maxnprop]=np.packbits(np.reshape(ksets4[0],maxnprop*maxnprop))

ksets4=np.zeros((4,maxnprop,maxnprop),dtype=bool)

#a
np.save('pakovani3',packedksets)

print(dt.datetime.now())

#packedksets = np.load('pakovani3.npy')



#testiranje

'''
ksets4[0]=np.resize(np.unpackbits(packedksets[69,82,0])[:maxnprop*maxnprop],(maxnprop,maxnprop))
print(proposals[69,82])
print('.')
print(proposals[70,82])
print('.')
np.save('evoga0',ksets4[0,:nprop[69,82],:nprop[70,82]])
print('.')

ksets4[1]=np.resize(np.unpackbits(packedksets[34,70,0])[:maxnprop*maxnprop],(maxnprop,maxnprop))
print(proposals[34,70])
print('.')
print(proposals[34,71])
print('.')
np.save('evoga1',ksets4[1,:nprop[34,70],:nprop[34,71]])
print('.')
'''

print(0)
psicosts=np.zeros((maxnprop,maxnprop))
#mat1=np.zeros((maxnprop,maxnprop))
#mat2=np.zeros((maxnprop,maxnprop))
#for qi in range(maxnprop):
#    for qj in range(maxnprop):
#        mat1[qi,qj]=qi
#        mat2[qi,qj]=qj
def bcd(ystep,xstep,ty,tx):
    global unpacktime
    global truestime
    minarr=np.zeros(50,dtype=int)
    trues=np.nonzero([1,1,1,1,1,1,1,1,1,1])
    if(ystep==0):
        xside=1
        yside=0
    else:
        xside=0
        yside=1
    i = 0
    dp = (np.full((2*max(pich,picw),maxnprop),1000.0))
    pastlabels = (np.full((2*max(pich,picw),maxnprop+1),1000,dtype=int)).tolist()
    for tl in range(nprop[ty,tx]):
        dp[0,tl]=sidepsi(ty,tx,tl, ty+yside, tx+xside) + sidepsi(ty,tx,tl, ty-yside, tx-xside) + lcosts[ty,tx,tl]
    while(True):
        #ksets je prosli u DP
        #
        ty+=ystep
        tx+=xstep
        i+=1
        if(tx<0 or ty<0 or tx>=picw or ty>=pich): break
        #if(i==6): print('e 6')
        t1=dt.datetime.now()
        if(ystep==-1 and xstep == 0):
            ksets=np.reshape(np.unpackbits(packedksets[ty,tx,0])[:maxnprop*maxnprop],(maxnprop,maxnprop))
        elif(ystep==1 and xstep==0):
            ksets=np.reshape(np.unpackbits(packedksets[ty-1,tx,0])[:maxnprop*maxnprop],(maxnprop,maxnprop))
        elif(ystep==0 and xstep==1):
            ksets=np.reshape(np.unpackbits(packedksets[ty,tx-1,1])[:maxnprop*maxnprop],(maxnprop,maxnprop))
        else:
            ksets=np.reshape(np.unpackbits(packedksets[ty,tx,1])[:maxnprop*maxnprop],(maxnprop,maxnprop))
            #dp[i,0:nprop[ty,tx]]=np.max(dp[i-1,0:nprop[ty-ystep,tx-xstep]]+np.where(ksets4[0,0:nprop[ty-ystep,tx-xstep],0:nprop[ty,tx]],  psi(ty-ystep,tx-xstep, uf-ovo-nije-sidepsi, ty,tx), tpsi))
        t2=dt.datetime.now()
        delta=t2-t1
        unpacktime = unpacktime + delta.microseconds/1000000.0
        
        minc=10000.0
        tnprop=nprop[ty,tx]
        pnprop=nprop[ty-ystep,tx-xstep]
        #psicosts=np.where(ksets4[2,:tnprop,:pnprop], sidepsi(ty,tx,mat1[:tnprop,:pnprop],ty,tx+1),tpsi)
        permmincost=10000.0
        permminlabel=-8
        for tk in range(pnprop):
            if(tpsi+dp[i-1,tk]<permmincost):
                permmincost=tpsi+dp[i-1,tk]
                permminlabel=tk

        if(ystep==-1 or xstep==-1):
            for tl in range(tnprop):
                smallcosts=lamda*lcosts[ty,tx,tl] + sidepsi(ty,tx,tl,ty+yside,tx+xside) + sidepsi(ty,tx,tl,ty-yside,tx-xside) #postoji li ?
                mincost=permmincost
                af, bf = proposals[ty,tx,tl,0], proposals[ty,tx,tl,1]
                pastlabels[i][tl]=permminlabel

                #myb[:pnprop]=np.where(ksets[tl,:pnprop], dp[i-1,:pnprop]+purepsi(af,bf,proposals[ty-ystep,tx-xstep,:pnprop,0],proposals[ty-ystep,tx-xstep,:pnprop,1]), mincost)
                #dp[i,tl]=np.min(myb[:pnprop])+smallcosts
                
                for tk in range(pnprop):
                    if(ksets[tl,tk]):
                        mybcost=dp[i-1,tk]+purepsi(af,bf,proposals[ty-ystep,tx-xstep,tk,0],proposals[ty-ystep,tx-xstep,tk,1])
                        if(mybcost<mincost):
                            mincost=mybcost
                            pastlabels[i][tl]=tk
                dp[i,tl]=mincost+smallcosts
                         
                #dp[i,tl]=mincost+smallcosts

                #if(ty==50):
                #    print('oblik',np.shape(trues))
                
                #dp[i,tl]=mincost+smallcosts
                #nebitno:
                #psicosts=np.full((nprop[ty-1,tx],nprop[ty,tx]),tpsi)
                #ovo je vrv tacno:
                #psicosts=np.where(ksets4[0,:anprop,:bnprop],purepsi(proposals[ty-1,tx,mat1[:anprop,:bnprop],0],proposals[ty-1,tx,mat1[:anprop,:bnprop],1],proposals[ty,tx,mat2[:anprop,:bnprop],0],proposals[ty,tx,mat2[:anprop,:bnprop],1]),tpsi)
                #BUDI JAKO PAZLJIV SVE OVDE MOZE BITI NETACNO
                #for tl in range(nprop[ty-1,tx]):
                #    for tk in range(nprop[ty,tx]):
                #        if(ksets4[0,tl,tk]):\
                 
                #            psicosts[tl,tk]=purepsi(proposals[ty-1,tx,tl,0],proposals[ty-1,tx,tl,1],proposals[ty,tx,tk,0],proposals[ty,tx,tk,1])
        else:
            for tl in range(tnprop):
                smallcosts=lamda*lcosts[ty,tx,tl] + sidepsi(ty,tx,tl,ty+yside,tx+xside) + sidepsi(ty,tx,tl,ty-yside,tx-xside) #postoji li ?
                mincost=permmincost
                af, bf = proposals[ty,tx,tl,0], proposals[ty,tx,tl,1]
                pastlabels[i][tl]=permminlabel

                #myb[:pnprop]=np.where(ksets[:pnprop,tl], dp[i-1,:pnprop]+purepsi(af,bf,proposals[ty-ystep,tx-xstep,:pnprop,0],proposals[ty-ystep,tx-xstep,:pnprop,1]), mincost)
                #dp[i,tl]=np.min(myb[:pnprop])+smallcosts

                
                for tk in range(pnprop):
                    if(ksets[tk,tl]):
                        mybcost=dp[i-1,tk]+purepsi(af,bf,proposals[ty-ystep,tx-xstep,tk,0],proposals[ty-ystep,tx-xstep,tk,1])
                        if(mybcost<mincost):
                            mincost=mybcost
                            pastlabels[i][tl]=tk
                dp[i,tl]=mincost+smallcosts
                
                #RESETAVAJ TRUES
                
                #if(tl==0): print('oblik',np.shape(trues)) 
                #dp[i,tl]=mincost+smallcosts

    #sad rekonstrukcija
    #uzmem min dp[posl-1]
    ty-=ystep
    tx-=xstep
    i-=1
    mincost=10000.0
    minlabel=0
    for tl in range(nprop[ty,tx]):
        if(dp[i,tl]<mincost):
            mincost=dp[i,tl]
            minlabel=tl
    bestlabels[ty,tx]=minlabel
    pl=minlabel
    while(True):
        ty-=ystep
        tx-=xstep
        if(i<0): print('i je manje od 0',ty,tx,i,pl)
        if(pl==1000): 
            print('pl je 1000',ty,tx,i,pl)
            print(proposals[ty-1,tx-1])
            print(proposals[ty-1,tx])
            print(proposals[ty-1,tx+1])
            print(proposals[ty,tx-1])
            print(proposals[ty,tx])
            print(proposals[ty,tx+1])
            print(proposals[ty+1,tx-1])
            print(proposals[ty+1,tx])
            print(proposals[ty+1,tx+1])
            print(bestlabels[ty-1,tx-1])
            print(bestlabels[ty-1,tx])
            print(bestlabels[ty-1,tx+1])
            print(bestlabels[ty,tx-1])
            print(bestlabels[ty,tx])
            print(bestlabels[ty,tx+1])
            print(bestlabels[ty+1,tx-1])
            print(bestlabels[ty+1,tx])
            print(bestlabels[ty+1,tx+1])
        if(tx<0 or ty<0 or tx>=picw or ty>=pich): break
        pl=pastlabels[i][pl]
        i-=1
        bestlabels[ty,tx]=pl
    #print(bestlabels)
    #print(pastlabels)

    return dp
print(bestlabels[35])
for w in range(bcd_times):
    for xloc in range(0,picw,2):
        bcd(1,0,0,xloc)
        #print('a')
    #print(nprop[50])
    print(bestlabels[35])
    for yloc in range(0,pich,2):
        bcd(0,-1,yloc,picw-1)
    for xloc in range((picw//2)*2-1,-1,-2):   
        bcd(-1,0,pich-1,xloc)
    for yloc in range((pich//2)*2-1,-1,-2):
        bcd(0,1,yloc,0)

finalpic=np.zeros((pich,picw,2))
for ty in range(pich):
    for tx in range(picw):
        finalpic[ty,tx]=proposals[ty,tx,bestlabels[ty,tx]]
print('unpacking',unpacktime)
print('trues',truestime)
print(finalpic[25])
print(finalpic[:,25])
np.save('veliki_flow_nakon_1',finalpic)

print(dt.datetime.now())
