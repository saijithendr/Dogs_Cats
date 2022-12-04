import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)     # 3-color, output-6, kernal-5
        self.pool = nn.MaxPool2d(2, 2)      # pooling---> kernal-2 maxpool-2
        self.conv2 = nn.Conv2d(6, 16, 4)    # Conv---> 6-input(must be same as output of prev conv), output-16, kernal-5
        self.fcn1 = nn.Linear(19600, 120)  # fully NN1---> input-16*5*5, hidden-120    32-5+
        self.fcn2 = nn.Linear(120, 90)      # fully NN2---> hidden_input-120, hidden_output-80 
        self.fcn3 = nn.Linear(90, 2)       # fully NN3---> input-80, output-2 (num target classes)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16*35*35)
        x = F.relu(self.fcn1(x))
        x = F.relu(self.fcn2(x))
        x = self.fcn3(x)
        return x
    