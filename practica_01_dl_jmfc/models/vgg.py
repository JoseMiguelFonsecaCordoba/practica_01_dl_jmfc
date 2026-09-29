import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base import BaseNN
from torchvision.models import (vgg11, VGG11_Weights)

class VGG11(BaseNN): #de aqui hasta el comentario de abajo es transfer learning
    def __init__(self, num_classes=10, pretrained=True):
        super(VGG11, self).__init__(
            name="vgg11",
        )
        self.network = vgg11(
            weights=VGG11_Weights.DEFAULT if pretrained else None
        )

        for parameter in self.network.features.parameters():
            parameter.requires_grad = False

        in_features = self.network.classifier[-1].in_features
        self.network.classifier[-1] = nn.Linear(
            in_features, 
            num_classes
        ) 
        #transfer learning, quitamos la ultima capa y agregamos una nueva 
        # capa de salida con el numero de clases que tenemos


        ## BUFFERS PARA NORMALIZACION DE IMAGENES ##
        self.register_buffer(
            "mnist_std",
            torch.tensor([0.3081]).view(1, 1, 1, 1)
        )
        self.register_buffer(
            "mnist_mean",
            torch.tensor([0.1307]).view(1, 1, 1, 1)
        )
        self.register_buffer(
            "imagenet_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1) #es la desviacion estandar de imagenet, es un vector de 3 porque imagenet es rgb
        )
        self.register_buffer(
            "imagenet_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1) #es la media de imagenet, es un vector de 3 porque imagenet es rgb
        )

    def forward(self, x):
        x = x * self.mnist_std + self.mnist_mean  # desnormalizar nlist
        x = x.repeat(1, 3, 1, 1) # capa extra para convertir de 1 canal a 3 canales, porque imagenet es rgb
        x = F.interpolate(
            x, 
            size=(224, 224), 
            mode='bilinear', 
            align_corners=False
            )
        #normalizamos usando la media y desviacion estandar de imagenet
        #esto es obligatorio porque el modelo fue entrenado con imagenet
        # y si no normalizamos con la media y desviacion estandar de imagenet, 
        # el modelo no va a funcionar bien
        x = (x - self.imagenet_mean) / self.imagenet_std

        return self.network(x)


    #en aprendizaje profundo rara vez se entrena un modelo de 0,
    #el modelo entra mas grande sea mas informacion necesita para entrenar
    # si no tenemos mucha informacion el modelono va a funcionar

    # para eso existe rasnfer learning , tomaomos un modelo que ya existe y los pesos de ese modelo
    #lo cargamos, una vez cargado el modelo con sus pesos
    #la parte de featurs congelarlo y me regreso a cnn

    #features para la parte de exrtaccion de caracteristicas
    # y classifier para la parte de clasificacion, la parte de features ya no la necesitamos

    #si no puedo propagar error, no puedo entrenar
    #todas las capas convulucionales ya no las entrenes, porque ya esta entrenados
    # pero este modelo vgg su cabeza calsificadora tiene 1000 neuronas a la salida
    # esa capa si la estoy forzando a que ya no tenga 1000 neuronas sino que tenga 10 neuronas,
    # porque tengo 10 clases y se adapte a mi problema, en este caso mnist

    #necesito que mis datos de entrada esten normzaliados, todo lo que meta a este modelo
    # tiene que estar normalizado

    # transfer learning si requiere que entrnemos, pero solo a ultima capa, 
    # porque la parte pesada ya esta pre-entrenada



    # podemos buscar en la documentacion oficial
    # desde https://pytorch.org



    # estandar abierto para redes neuronales
    # servidor de inferencia TRITON 
    
    # versionamiento, podemos guardar el modelo, si tenemos una actualizacion de nuestro modelo, 
    # podemos guardar la version anterior y la nueva version
    # SERVIDOR DE INFERENCIA   !!!!!1IMPORTANTE!!!!!!

    

