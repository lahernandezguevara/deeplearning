import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import kagglehub
import os

def get_cat_dog_loaders(batch_size: int):
    """
    Descarga (si no existe) y carga el dataset de Cat and Dog de Kaggle.
    Aplica las transformaciones obligatorias para VGG11.
    """
    print("Descargando/Verificando dataset de Kaggle: tongpython/cat-and-dog...")
    base_path = kagglehub.dataset_download("tongpython/cat-and-dog")
    print(f"Dataset disponible en: {base_path}")
    
    # Manejar las variantes clásicas de anidación del dataset en Kaggle
    train_dir = os.path.join(base_path, "training_set", "training_set")
    if not os.path.exists(train_dir):
        train_dir = os.path.join(base_path, "training_set")
        
    test_dir = os.path.join(base_path, "test_set", "test_set")
    if not os.path.exists(test_dir):
        test_dir = os.path.join(base_path, "test_set")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)
    
    num_classes = len(train_dataset.classes)
    
    # DataLoaders finalizados
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, num_classes
