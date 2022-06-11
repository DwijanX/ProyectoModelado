#%%
import numpy
import joblib
from sklearn.svm import SVC
import matplotlib.pyplot as pl
import h5py
import cv2

def processImg(Img):
    GrayImage=cv2.cvtColor(Img,cv2.COLOR_BGR2GRAY)
    GaussianFilter=cv2.GaussianBlur(GrayImage,(5,5),0)
    return GaussianFilter.reshape(1,-1)
def ProbarLambda(FileDirection,XName,YNAME):
        ConfusionMatrix=numpy.array([[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]])
        ErrorCount=0
        SuccessCount=0
        DataFile=h5py.File(FileDirection,'r')
        X=DataFile[XName][:]
        Y=DataFile[YNAME][:]
        for x in X:
            ans=clasificador.predict(processImg(x))
            if Y[(ErrorCount+SuccessCount)]==0:
                print(ans)
                pl.imshow(x)
                pl.show()
            if(ans!=Y[(ErrorCount+SuccessCount)] ):
                ConfusionMatrix[int(Y[(ErrorCount+SuccessCount)])][0]+=1
                ErrorCount+=1
            else:
                ConfusionMatrix[int(Y[(ErrorCount+SuccessCount)])][1]+=1
                SuccessCount+=1
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

#clasificador=joblib.load('Code\ParametrosProyecto')
clasificador=joblib.load('./ParametrosProyecto')



#Suc,Err=ProbarLambda("Images\DatasetAlcohol1.h5","X_TrainSet","Y_TrainSet")
Suc,Err=ProbarLambda("../Images/DatasetAlcohol1.h5","X_TrainSet","Y_TrainSet")

print("aciertos",Suc)
print("fallos",Err)
