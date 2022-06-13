#%%
import cv2
import torch
from torch.autograd import Variable
from torchvision import datasets, transforms
import torch.nn.functional as F
import numpy
import h5py
from torch.utils.data import Dataset
import matplotlib.pyplot as pl
from PIL import Image

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
#model_ft = torch.load("Gatos\Files\mi_modeloDeGatos.pt")
model_ft = torch.load("Code\mi_modeloDeAlcohol.pt")

img=cv2.imread("Code\\test.jpg")
#img=cv2.imread("Code\\Test2.jpg")

def ProcessImg(Img):
        GrayImage=cv2.cvtColor(Img,cv2.COLOR_BGR2GRAY)
        thresh=150
        ret,thresh_img = cv2.threshold(GrayImage, thresh, 255, cv2.THRESH_BINARY)
        #pl.imshow(thresh_img)
        #pl.show()
        grupos,_=cv2.findContours(thresh_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        ventanas= [cv2.boundingRect(g) for g in grupos]
        for bots in ventanas:
            (x, y, w, h) = bots
            espacio=int(bots[3]*1.6)
            p1=int((bots[1]+bots[3]//2))-espacio//2
            p2=int((bots[0]+bots[2]//2))-espacio//2
            JustBottle = Img[y:y + h+espacio,x:x+w+espacio]
            if  w>30 and h>30:
                pilImg = Image.fromarray(JustBottle)
                JustBottle = pilImg.resize((128,128))
                cv2.rectangle(Img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                bottle = transform(JustBottle)
                bottle.unsqueeze_(dim=0)
                bottle = Variable(bottle)
                bottle = bottle.view(bottle.shape[0], -1)
                ansvec = F.log_softmax(model_ft(bottle))
                ans=ansvec.argmax().item()
                cv2.putText(Img,str(ans),(x,y+30),cv2.FONT_HERSHEY_SIMPLEX,2,(255,0,0))
        return Img


#pl.imshow(img)
#pl.show()
ScannedImg=ProcessImg(img)
pl.imshow(ScannedImg)
pl.show()
"""
pilImg = Image.fromarray(img)
JustBottle = pilImg.resize((128,128))
bottle = transform(JustBottle)
bottle.unsqueeze_(dim=0)
bottle = Variable(bottle)
bottle = bottle.view(bottle.shape[0], -1)
ansvec = F.log_softmax(model_ft(bottle))
ans=ansvec.argmax().item()

print(ans)
"""











