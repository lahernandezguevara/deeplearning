from models.cnn import CNN
from models.mlp import MLP
from models.vgg import VGG11

# Crear fabrica, aqui puedo definir todo 
# lo que quiero que contenga mis modelos
def create_model(
    model_name,
    num_classes=10,
):
    if model_name == "cnn":
        return CNN()
    if model_name == "mlp":
        return MLP()
    if model_name == "vgg11":
        return VGG11(
            # Recibe numero de clases
            num_classes
        )
    else:
        raise ValueError("Unknown model")