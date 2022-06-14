import torch
from time import time
from torchvision import transforms
from torch import nn, optim
from torch.utils.data import Dataset
import h5py


class H5DData(Dataset):
    
    def __init__(self, archivo,XName,YName, transform=None):
        self.archivo = h5py.File(archivo, 'r')
        self.labels = self.archivo[YName]
        self.data = self.archivo[XName]
        self.transform = transform

    def __getitem__(self, index):
        datum = self.data[index]
        if self.transform is not None:
            datum = self.transform(datum)
        return datum, self.labels[index]

    def __len__(self):
        return len(self.labels)

    def close(self):
        self.archivo.close()


#transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
#Para HOG
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_loader = torch.utils.data.DataLoader(H5DData("Images\DatasetAlcoholHOG.h5","X_TrainSet","Y_TrainSet",transform), batch_size=64, shuffle=True)
#test_loader = torch.utils.data.DataLoader(H5DData("../Files/gatillos_test.h5","test_set_x","test_set_y",transform), batch_size=64, shuffle=True)

capa_entrada = 128*128
capas_ocultas = [256, 128]
capa_salida = 2

modelo = nn.Sequential(nn.Linear(capa_entrada, capas_ocultas[0]), nn.ReLU(),
                       nn.Linear(capas_ocultas[0], capas_ocultas[1]), nn.ReLU(),
                       nn.Linear(capas_ocultas[1], capa_salida), nn.AlphaDropout())

j = nn.CrossEntropyLoss()

# entrenamiento de la red
optimizador = optim.Adam(modelo.parameters(),  lr=0.005)
tiempo = time()
epochs = 10
for e in range(epochs):
    costo = 0
    for imagen, etiqueta in train_loader:
        
        imagen = imagen.view(imagen.shape[0], -1)
        optimizador.zero_grad()
        h = modelo(imagen.float())
        error = j(h, etiqueta.long())
        error.backward()
        optimizador.step()
        costo += error.item()
    else:
        print("Epoch {} - Funcion costo: {}".format(e, costo / len(train_loader)))
print("\nTiempo de entrenamiento (en minutes) =", (time() - tiempo) / 60)
torch.save(modelo, 'Code/mi_modeloDeAlcoholHOG.pt')
#torch.save(modelo, '../Files/mi_modeloDeGatos.pt')


#%%