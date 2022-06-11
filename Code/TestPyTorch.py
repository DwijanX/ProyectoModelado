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

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
#model_ft = torch.load("Gatos\Files\mi_modeloDeGatos.pt")
model_ft = torch.load("Code\mi_modeloDeAlcohol.pt")


def ProbarLambda(FileDirection,XName,YNAME):
        ConfusionMatrix=numpy.array([[0,0],[0,0]])
        ErrorCount=0
        SuccessCount=0
        DataFile=h5py.File(FileDirection,'r')
        X=DataFile[XName][:]
        Y=DataFile[YNAME][:]
        for x in X:
            bottle = transform(x)
            bottle.unsqueeze_(dim=0)
            bottle = Variable(bottle)
            bottle = bottle.view(bottle.shape[0], -1)
            #pl.imshow(gato.reshape(64*3,64))
            #pl.show()
            ansvec = F.log_softmax(model_ft(bottle))
            ans=ansvec.argmax().item()
            
            if(ans!=Y[(ErrorCount+SuccessCount)] ):
                ConfusionMatrix[int(Y[(ErrorCount+SuccessCount)])][0]+=1
                ErrorCount+=1
            else:
                ConfusionMatrix[int(Y[(ErrorCount+SuccessCount)])][1]+=1
                SuccessCount+=1
        print(ConfusionMatrix)
        """Precision=ConfusionMatrix[1][1]/(ConfusionMatrix[1][1]+ConfusionMatrix[1][0])
        Recall=ConfusionMatrix[1][1]/(ConfusionMatrix[1][1]+ConfusionMatrix[0][0])
        F1=2*((Precision*Recall)/(Precision+Recall))
        Accuracy=(ConfusionMatrix[1][1]+ConfusionMatrix[0][1])/(ConfusionMatrix[1][1]+ConfusionMatrix[1][0]+ConfusionMatrix[0][1]+ConfusionMatrix[0][0])
        print(ConfusionMatrix)
        print("Precision",Precision*100)
        print("Recall",Recall*100)
        print("F1",F1*100)
        print("Accuracy",Accuracy*100)"""
        
        return SuccessCount/(ErrorCount+SuccessCount),ErrorCount/(ErrorCount+SuccessCount)




#Suc,Err=ProbarLambda("Gatos\Files\gatillos_test.h5","test_set_x","test_set_y")
Suc,Err=ProbarLambda("Images\DatasetAlcohol.h5","X_TrainSet","Y_TrainSet")

print("aciertos",Suc)
print("fallos",Err)





















#%%

