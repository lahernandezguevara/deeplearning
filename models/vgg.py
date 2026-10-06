import torch 
import torch.nn as nn
import torch.nn.functional as F
from models.base import BaseNN
from torchvision.models import (vgg11, VGG11_Weights)

# --------------------------------------------------------------------------
# TRANSFER LEARNING: Tomamos un modelo que ya existe y los pesos de ese modelo
class VGG11(BaseNN):
    def __init__(self, num_classes=10):
        super(VGG11, self).__init__(
            name = "vgg11",
        )
        # Creando el modelo y cargando los pesos
        self.network = vgg11(
            weights=VGG11_Weights.DEFAULT
        )

        # Ya no entrenes las capas convolucionales porque tus pesos
        # ya estan preentrenados
        for parameter in self.network.features.parameters():
            # A toda la parte de convolucionales desactivamos el gradiente
            parameter.requires_grad = False
            # No requiero un gradiente, no tengo retropropagacion del error
            # no actualizo parametros, no puedo entrenar

        in_features = self.network.classifier[-1].in_features
        # Aqui se acaba transfer learning
        self.network.classifier[-1] = nn.Linear(
            in_features, 
            # Forzo a que tenga 10 neuronas de salida para que se adapte a mnist
            num_classes
        )

# --------------------------------------------------------------------------------------

    def forward(self, x):
        # La conversión y desnormalización de MNIST ya no es necesaria
        # debido a que los DataLoaders de "tongpython/cat-and-dog" (y las peticiones 
        # en la interfaz de Triton) están preprocesando las imágenes a 224x224 RGB
        # usando las medias/std de ImageNet automáticamente.
        return self.network(x)




        
    # Que es onx, es un estandar abierto para redes neuronales, ahorita guardamos el 
    # modelo como pth, este es un screenshot del entrenamiento, vamos a exportaro como
    # .onx, modelo listo para produccion, una vez que sepamos como exportarlo y poderlo 
    # guardar, vamos a poder usar un servidor de inferencia llamado triton

    # Cuando la gente envuelve un modelo en algun backend(fastapi, django) ese backend carga el 
    # modelo en memoria y hace peticiones, el modelo puede hacer muchas cosas al mismo tiempo
    # Estoy desperdiciando totalmente mi gpu, un servidor de inferencia tiene una ventana
    # abierta de peticiones (222 milisegundos) las encapsula en un batch, y responde
    # El servidor de inferencia es mas eficiente, optimo en memoria, tenemos versionamiento,
    # version 1 del modelo, version 2 hace un hot-swap, carga version 2 del modelo automatico
    # Hot-swap de modelos en produccion, sagemaker en aws usa triton, en ia generativa se usa
    # triton o vvllm, enfocado a modelos de lenguaje largos