import os
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as T
from utils import CNN

device = ('cuda' if torch.cuda.is_available() else 'cpu')

def load_data(train_dir: str, test_dir: str, batch_size: int):
    
    transform = T.Compose(
        [T.Resize((150, 150), T.InterpolationMode.BILINEAR), T.ToTensor()]
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
learning_rate = 0.01
num_epochs = 10



train_data, test_data = load_data(
    train_dir=train_dir, test_dir=test_dir, batch_size=batch_size
)

# Training 
model = CNN().to(device)

optimizer = torch.optim.Adam(
    params=model.parameters(), lr=learning_rate )

criterion = nn.CrossEntropyLoss()

n_total_steps = len(train_data)

#Loop 
for epoch in range(num_epochs):
    for id, (img, label) in enumerate(train_data):
        
        img = img.to(device)
        label = label.to(device)
        
        # forward pass
        output = model(img)
        loss = criterion(output, label)
        
        #Backward and loss
        optimizer.zero_grad()
        loss.backward()
        
        if (id+1) % 100 == 0:
            print(f'epoch {epoch+1} / {num_epochs}, step {id+1}/{n_total_steps}, loss = {loss.item():.4f}')
