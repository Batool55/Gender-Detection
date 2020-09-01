import torch
import torch.nn as nn
import torch.nn.functional as F

class MultilayerPs(nn.Module):
    def __init__(self ):
        super(MultilayerPs, self).__init__()
        self.fc1 = torch.nn.Linear(12 ,600)
        self.out = torch.nn.Linear(600,1)
        
    def forward(self, x):
        x = F.tanh(self.fc1(x))
        x = F.sigmoid(self.out(x))
        return x
