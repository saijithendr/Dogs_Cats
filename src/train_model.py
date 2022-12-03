import os
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as T

def load_data(train_dir: str, test_dir: str, batch_size: int):
    
    transform = T.Compose(
        [T.Resize(150, 200), T.ToTensor()]
    )
    
    train_dataset = torchvision.datasets.ImageFolder(
        root= train_dir, transform= transform
    )
    
    test_dataset =  torchvision.datasets.ImageFolder(
        root= test_dir, transform= transform
    )
    
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)
    
    return train_dataloader, test_dataloader



train_dir = os.path.join('data/cats_dogs/PetImages', 'train')
test_dir = os.path.join('data/cats_dogs/PetImages', 'test')

# Hyperparameters
batch_size = 32
learning_rate = 0.001
num_epochs = 20



train_data, test_data = load_data(
    train_dir=train_dir, test_dir=test_dir, batch_size=batch_size
)

