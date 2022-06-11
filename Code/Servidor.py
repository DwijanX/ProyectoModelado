from flask import Flask, request, Response
import jsonpickle
import numpy
import cv2
import torch
from torch.autograd import Variable
from torchvision import datasets, transforms
import torch.nn.functional as F

app = Flask(__name__)

# instanciamos nuestra red
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
model_ft = torch.load("modelo/mi_modelo_digitos.pt")

# cargamos los parametros que entrenamos

@app.route('/api/test', methods=['POST'])
def test():
    r = request
    print("se ingresa a la api")
    # convert string of image data to uint8
    nparr = numpy.fromstring(r.data, numpy.uint8)
    # decode image
    imagen = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    # reducimos ruido aplicando suavizado gaussiano
    imagen_gris = cv2.GaussianBlur(imagen_gris, (5, 5,), 0)
    # convertimos a blanco y negro
    ret, imagenBN = cv2.threshold(imagen_gris, 90, 255, cv2.THRESH_BINARY_INV)
    # identificamos grupos en la imagen
    grupos, _ = cv2.findContours(imagenBN.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # enventanamos los grupos
    ventanas = [cv2.boundingRect(g) for g in grupos]
    # para cada ventana identificada......
    c = 0
    numero = 0
    ventanas.sort(reverse=True)
    for g in ventanas:
        # computamos desplazamiento para ampliar la ventana
        l = int(g[3] * 1.6)
        # ampliar ancho
        p1 = int(g[1] + g[3] // 2) - l // 2
        # ampliar largo
        p2 = int(g[0] + g[2] // 2) - l // 2
        # dibujamos un rectangulo
     #   cv2.rectangle(imagen, (p2, p1), (p2 + l, p1 + l), (255, 0, 0), 2)
        # capturamos la ventana en digito
        digito = imagenBN[p1: p1 + l, p2: p2 + l]
        # escalamos a imagen 20x20, que es la dimencion con la que entrenamos nuestra red
        digito = cv2.resize(digito, (28, 28), interpolation=cv2.INTER_AREA)
        digito = cv2.dilate(digito, (3, 3,))
        # transpuesta y aplanamos a un vector de 400
        aux = digito.T.flatten()
        # introducimos a nuetsro algoritmo front proagation para la prediccion
        digito = transform(digito)
        digito.unsqueeze_(dim=0)
        digito = Variable(digito)

        digito = digito.view(digito.shape[0], -1)
        recon = F.softmax(model_ft(digito), dim=1)

        # mostramos el numero reconocido en la imagen
        print(str(recon.argmax().item()), " * 10^",c)
        numero += recon.argmax().item() * 10 ** c
        print("resultante : ", numero)
        c += 1

    response = {'message': 'Se reconoce el numero {}'.format(numero)
                }
    response_pickled = jsonpickle.encode(response)
    return Response(response=response_pickled, status=200, mimetype="application/json")


if __name__ == '__main__':
    app.run()