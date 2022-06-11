#%%
import numpy
import joblib
from sklearn.svm import SVC
import matplotlib.pyplot as pl
import h5py
import cv2

from PIL import Image

def ProcessImg(Img):
        GrayImage=cv2.cvtColor(Img,cv2.COLOR_BGR2GRAY)
        GaussianFilter=cv2.GaussianBlur(GrayImage,(5,5),0)
        ret, imagenBN = cv2.threshold(GaussianFilter, 127, 255, cv2.THRESH_BINARY_INV)
        grupos,_=cv2.findContours(imagenBN.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        ventanas= [cv2.boundingRect(g) for g in grupos]
        for bots in ventanas:
            (x, y, w, h) = bots
            espacio=int(bots[3]*1.6)
            p1=int((bots[1]+bots[3]//2))-espacio//2
            p2=int((bots[0]+bots[2]//2))-espacio//2
            JustBottle = imagenBN[y:y + h+espacio,x:x+w+espacio]
            if  p2>0 and p1>0 and espacio>80:
                pilImg = Image.fromarray(JustBottle)
                JustBottle = pilImg.resize((128,128))
                JustBottle = numpy.array(JustBottle)
                cv2.rectangle(Img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                identified=clasificador.predict(JustBottle.reshape(1,-1))
                print(identified)
                cv2.putText(Img,str(identified),(x,y-20),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0))
        return Img

#clasificador=joblib.load('Code\ParametrosProyecto')
clasificador=joblib.load('./ParametrosProyecto')



video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    
    frame=ProcessImg(frame)
    cv2.imshow('Video', frame)

    if cv2.waitKey(49) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
