import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def get_custom_loaders(data_dir: str, batch_size: int, split_ratios: tuple = (0.8, 0.1, 0.1)):
    """
    Load custom image dataset from data_dir using ImageFolder.
    Applies required transformations for VGG11.
    
    Returns: train_loader, val_loader, test_loader, num_classes
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    num_classes = len(dataset.classes)
    
    total_size = len(dataset)
    train_size = int(total_size * split_ratios[0])
    val_size = int(total_size * split_ratios[1])
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)  # reproducibilidad
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, num_classes
