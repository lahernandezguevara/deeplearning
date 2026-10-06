import torch 
from functools import cache

# Decorador de cache, el dipositivo se guarda y no se tiene que 
# ejecutar otra vez 
@cache
def get_device():
    entrada = input("seleccione 1 para gpu y 0 para cpu: ")
    
    if int(entrada) == 1:
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            print(f"GPU is available. Using device: {device_name}")
            return torch.device("cuda")
        else:
            print("GPU selected but not available, using CPU instead.")
            return torch.device("cpu")
    else:
        print("Using CPU instead.")
        return torch.device("cpu")

