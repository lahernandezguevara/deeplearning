from abc import ABC, abstractclassmethod
import torch.nn as nn

class BaseNN(nn.Module, ABC):
    def __init__(self, name):
        super(BaseNN, self).__init__()
        self.name = name

    # Si yo le agrego un metodo abstracto se convierte en una
    # funcion abstracta, la clase abstracta si puede tener un metodo
    # definido, solo requiere una funcion miembro, entoncces es una
    # Interface
    @abstractclassmethod
    def forward(self, x):
        pass

# Implementando patron de disenio factory

# Vgg tiene 64 filtro sconvolucionales, un pooling
# mas filtro y otro pooling
# Vgg es una red convolucional profunda, vgg11 (11 capas) 
# existe 16 y 19

# Los modelo enorme no ddddd
# Entre ma grande el modelo neceitamo ma 
# datofrom abc import ABC, abstractclassmethod
# Tranfern learning, carga modelo original,
# congelamo parte de etraccion y decongelamo la 
# parte de 