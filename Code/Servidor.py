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
from io import BytesIO
from skimage.feature import hog

app = Flask(__name__)
run_with_ngrok(app)
# instanciamos nuestra red

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
model_ft = torch.load("Code\mi_modeloDeAlcoholHOG.pt")
#model_ft = torch.load("./mi_modeloDeAlcohol.pt")
# cargamos los parametros que entrenamos
def ProcessImg(Img):
        GrayImage=cv2.cvtColor(Img,cv2.COLOR_BGR2GRAY)
        thresh=150
        ret,thresh_img = cv2.threshold(GrayImage, thresh, 255, cv2.THRESH_BINARY)
        #pl.imshow(thresh_img)
        #pl.show()
        grupos,_=cv2.findContours(thresh_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        ventanas= [cv2.boundingRect(g) for g in grupos]
        fd, HogImg = hog(Img, orientations=9, pixels_per_cell=(8, 8),cells_per_block=(2, 2), visualize=True, channel_axis=-1)
        for bots in ventanas:
            (x, y, w, h) = bots
            espacio=int(bots[3]*1.6)
            p1=int((bots[1]+bots[3]//2))-espacio//2
            p2=int((bots[0]+bots[2]//2))-espacio//2

            JustBottle = HogImg[y:y + h+espacio,x:x+w+espacio]
            
            if  w>30 and h>30 and espacio>40:
                pilImg = Image.fromarray(JustBottle)
                JustBottleHog = pilImg.resize((128,128))
                cv2.rectangle(Img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                bottle = transform(JustBottleHog)
                bottle.unsqueeze_(dim=0)
                bottle = Variable(bottle)
                bottle = bottle.view(bottle.shape[0], -1)
                ansvec = F.alpha_dropout(model_ft(bottle))
                ans=ansvec.argmax().item()
                cv2.putText(Img,str(ans),(x,y+50),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0))
        return Img

@app.route('/api/test', methods=['POST','GET'])
def test():
    r = request
    Img64=r.form.get("img")
    im_bytes = base64.b64decode(Img64)
    im_arr = numpy.frombuffer(im_bytes, dtype=numpy.uint8)  # im_arr is one-dim Numpy array
    img = cv2.imdecode(im_arr, flags=cv2.IMREAD_COLOR)
    #print(img)
    #print(type(img))
    Ans=ProcessImg(img)
    pil_img = Image.fromarray(Ans)
    buff = BytesIO()
    pil_img.save(buff, format="JPEG")
    new_image_string = base64.b64encode(buff.getvalue()).decode("utf-8")
    data = {}
    data['img'] =new_image_string
    response_pickled = jsonpickle.encode(data)
    return Response(response=response_pickled, status=200, mimetype="application/json")
    
    print(type(img))
    print(img.shape)
    Ans=ProcessImg(img)
    data = {}
    data['img'] = base64.encodebytes(Ans).decode('utf-8')
    response_pickled = jsonpickle.encode(data)
    return Response(response=response_pickled, status=200, mimetype="application/json")


if __name__ == '__main__':
    app.run()

