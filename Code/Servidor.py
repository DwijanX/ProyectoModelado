from flask import Flask, request, Response,jsonify
import jsonpickle
import numpy
import cv2
import torch
from torch.autograd import Variable
from torchvision import datasets, transforms
import torch.nn.functional as F
from PIL import Image
import json
from flask_ngrok import run_with_ngrok
import base64
import matplotlib.pyplot as pl

app = Flask(__name__)
run_with_ngrok(app)
# instanciamos nuestra red
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
model_ft = torch.load("Code\mi_modeloDeAlcohol.pt")
#model_ft = torch.load("./mi_modeloDeAlcohol.pt")
# cargamos los parametros que entrenamos
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
            JustBottle = Img[y:y + h+espacio,x:x+w+espacio]
            if  p2>0 and p1>0 and espacio>60:
                pilImg = Image.fromarray(JustBottle)
                JustBottle = pilImg.resize((128,128))
                #JustBottle = numpy.array(JustBottle)
                cv2.rectangle(Img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                bottle = transform(JustBottle)
                bottle.unsqueeze_(dim=0)
                bottle = Variable(bottle)
                bottle = bottle.view(bottle.shape[0], -1)
                #pl.imshow(gato.reshape(64*3,64))
                #pl.show()
                ansvec = F.log_softmax(model_ft(bottle))
                ans=ansvec.argmax().item()
                #identified=clasificador.predict(JustBottle.reshape(1,-1))
                print(ans)
                cv2.putText(Img,str(ans),(x,y-20),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0))
        return Img

@app.route('/api/test', methods=['POST','GET'])
def test():
    r = request
    img = Image.frombuffer(r.data, numpy.uint8)
    
    print(type(img))
    print(img.shape)
    Ans=ProcessImg(img)
    data = {}
    data['img'] = base64.encodebytes(Ans).decode('utf-8')
    response_pickled = jsonpickle.encode(data)
    return Response(response=response_pickled, status=200, mimetype="application/json")


if __name__ == '__main__':
    app.run()

