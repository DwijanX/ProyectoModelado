#%%
import cv2
from cv2 import INTER_AREA
import torch
import matplotlib.pyplot as pl
import os
import numpy
from PIL import Image
import h5py
# assign directory


def WhitePatch(Img):
    Img_mean = (Img*1.0 / Img.mean(axis=(0,1)))
    return Img_mean

def ProcessImg(PilImg,ApplyWhitePatch):
    PilImg = PilImg.resize((128, 128), Image.ANTIALIAS)
    npImg = numpy.array(PilImg) 
    if ApplyWhitePatch:
        npImg=WhitePatch(npImg)
    return npImg

def getArrayOfImagesFromDir(dir,label,Rotate=False,ApplyWhitePatch=True):
    count=0
    Images=[]
    for filename in os.listdir(dir):
        f = os.path.join(dir, filename)
        if os.path.isfile(f):
            img = Image.open(f)
            if Rotate:
                img = img.rotate(-90, Image.NEAREST, expand = 1)
            
            img=ProcessImg(img,ApplyWhitePatch)
            if img.shape!=(128,128,3):
                print(f)
                print(img.shape)
            #pl.imshow(img)
            #pl.show()
            count+=1
            Images.append(img)
    y=numpy.ones(count)
    y=label*y
    ImagesNp=numpy.array(Images)
    print(ImagesNp.shape)
    return ImagesNp,y

directory1 = '../Images/Alcohol/Beer'
directory2 = '../Images/Alcohol/botellas'
directory3 = '../Images/Alcohol/cans'
directory4 = '../Images/Alcohol/RotateImagesBottles'
directory5 = '../Images/Soda/M.Beer'
directory6 = '../Images/Soda/M.Diet'
directory7 = '../Images/Soda/P.cherry'
directory9 = '../Images/Soda/P.Orig'
directory8 = '../Images/Alcohol/Copas'
directory10 = '../Images/RandomImgs'

#0 nada
#1 soda
#2 tarro de cerveza
#3 botella de alcohol
#4 lata de alcohol
#5 copas de vino
X1,Y1=getArrayOfImagesFromDir(directory1,2)
X2,Y2=getArrayOfImagesFromDir(directory2,3)
X=numpy.append(X1,X2,axis=0)
Y=numpy.append(Y1,Y2,axis=0)
X3,Y3=getArrayOfImagesFromDir(directory3,4,True)
X=numpy.append(X,X3,axis=0)
Y=numpy.append(Y,Y3,axis=0)
X4,Y4=getArrayOfImagesFromDir(directory4,3,True)
X=numpy.append(X,X4,axis=0)
Y=numpy.append(Y,Y4,axis=0)

X5,Y5=getArrayOfImagesFromDir(directory5,1,ApplyWhitePatch=False)
X=numpy.append(X,X5,axis=0)
Y=numpy.append(Y,Y5,axis=0)
X6,Y6=getArrayOfImagesFromDir(directory6,1,ApplyWhitePatch=False)
X=numpy.append(X,X6,axis=0)
Y=numpy.append(Y,Y6,axis=0)
X7,Y7=getArrayOfImagesFromDir(directory7,1,ApplyWhitePatch=False)
X=numpy.append(X,X7,axis=0)
Y=numpy.append(Y,Y7,axis=0)

X8,Y8=getArrayOfImagesFromDir(directory8,5)
X=numpy.append(X,X8,axis=0)
Y=numpy.append(Y,Y8,axis=0)

X9,Y9=getArrayOfImagesFromDir(directory9,1,ApplyWhitePatch=False)
X=numpy.append(X,X9,axis=0)
Y=numpy.append(Y,Y9,axis=0)

X10,Y10=getArrayOfImagesFromDir(directory10,0)
X=numpy.append(X,X10,axis=0)
Y=numpy.append(Y,Y10,axis=0)


print(X.shape)
print(Y.shape)
#for x in X:
#    pl.imshow(x)
#    pl.show()
#print(X.shape)
#print(Y.shape)

t=h5py.File('../Images/DatasetAlcohol.h5','w')
t.create_dataset('Y_TrainSet',data=Y)
t.create_dataset('X_TrainSet',data=X)
t.close()
"""
Images=numpy.array(Images)


ImagesResized=[]
for img in Images:
    aux=resizeImg(img)
    ImagesResized.append(aux)
"""



""" 
imagen=cv2.imread("../Images/Alcohol/8c43d837650c46528921251de55e24f4.jpg")

"""  

