import torch 
from functools import cache

# Decorador de cache, el dipositivo se guarda y no se tiene que 
# ejecutar otra vez 
@cache
def get_device():
    return torch.device("cpu")

