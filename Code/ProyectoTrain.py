import h5py
from sklearn import model_selection
import cv2
import numpy
import matplotlib.pyplot as pl
from sklearn.compose import TransformedTargetRegressor
from sklearn.svm import SVC
import joblib
import pickle
from PIL import Image

def PILImagenGris(imagen, tam):
    imgArr = numpy.empty_like(tam)
    for i in range(tam):
        print(imagen[i].shape)
        pl.imshow(imagen[i])
        pl.show()
        img = Image.fromarray(imagen[i])
        img = img.convert('L')
        imgArr[i] = numpy.array(img)
    return imgArr


def convertImagenGris(imagen, tam):
    train_gris=[]
    for i in range (tam):
        test = imagen[i]
        imagenGris = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)
        imagenGris = cv2.GaussianBlur(imagenGris, (5, 5), 0)
        train_gris.append(imagenGris.reshape(1,-1))
    train_gris = numpy.array(train_gris)
    train_gris = train_gris.reshape(tam, 128*128)
    return train_gris


def calculoFnFpVpVn(test, arrayRes, tam):
    Fn = 0 
    Fp = 0
    Vp = 0
    Vn = 0
    for i in range(0, tam-1):
        if (test[i] != arrayRes[i]):
            if arrayRes[i] == 1:
                Fp += 1
            else:
                Fn += 1
        else:
            if arrayRes[i] == 1:
                Vp += 1
            else:
                Vn += 1
    Mat = [[Fn, Vn],
           [Fp, Vp]]
    
    #precision
    pre = Mat[1][1]/(Mat[1][1]+Mat[1][0])
    #recall
    rec = Mat[1][1]/(Mat[1][1]+Mat[0][0])
    #accuracy
    acc = (Mat[1][1]+Mat[0][1])/(Mat[1][1]+Mat[0][1]+Mat[1][0]+Mat[0][0])
    #f1 
    f1 = 2*(pre*rec)/(pre+rec)
    
    return pre, rec, acc, f1


alcohol  = h5py.File("Images\DatasetAlcohol1.h5")

print(alcohol.keys())

print(alcohol['X_TrainSet'][:].shape)

X_train, X_test, y_train, y_test = model_selection.train_test_split(alcohol['X_TrainSet'][:], alcohol['Y_TrainSet'][:], test_size= 0.3, shuffle= False)

tam = len(X_train[:])

print (X_train.shape)

tamTest = len(X_test[:])

alcoholToGris = convertImagenGris(X_train, tam)

#alcoholToGris = PILImagenGris(X_train[:], tam)

clasificador = SVC(kernel = "poly")

#X_train = X_train.reshape(1030, 128*128*3)

clasificador.fit(alcoholToGris, y_train)



alcoholToGrisTest = convertImagenGris(X_test, tamTest)

arrRes = numpy.ones(tamTest)
for i in range(tamTest):
    res  = clasificador.predict(alcoholToGrisTest[i].reshape(1,-1))
    arrRes[i] = res
    

pre, rec, acc, f1 =  calculoFnFpVpVn(y_test, arrRes, tamTest)

print(pre, rec, acc, f1)

file = 'ParametrosProyecto1'

"""archivo = h5py.File(file, "w")
archivo.create_dataset("Theta1", data=)
archivo.create_dataset("Theta2", data=)"""

joblib.dump(clasificador, file)