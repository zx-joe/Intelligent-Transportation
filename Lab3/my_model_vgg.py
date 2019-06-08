import torch
from torchvision import datasets, transforms
import numpy as np
import torch.utils.data as utils
import time
import pdb
from torch.utils.data.sampler import SubsetRandomSampler
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import helper


torch.manual_seed(1)    # reproducible


class VGG(nn.Module):
    def __init__(self, n_input_channels=3, n_output=10):
        super().__init__()
        self.conv32 = torch.nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv64 = torch.nn.Conv2d(32, 64,kernel_size=3, padding=1)
        self.conv128_1 = torch.nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv128_2 = torch.nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv256_1 = torch.nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv256_2 = torch.nn.Conv2d(256, 256, kernel_size=5, padding=1)
        
        self.bn32 = torch.nn.BatchNorm2d(32)
        self.bn64 = torch.nn.BatchNorm2d(64)
        self.bn128 = torch.nn.BatchNorm2d(128)
        self.bn256 = torch.nn.BatchNorm2d(256)
        

        
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.pool_2 = torch.nn.AvgPool2d(1, 1)
        self.fc1 = torch.nn.Linear(256, 64)
        self.fc2 = torch.nn.Linear(64, 10)
        self.fc = torch.nn.Linear(256, 10)

    def forward(self, x):
        
        x = self.conv32(x)
        x = F.relu(self.bn32(x))
        x = self.pool(x)
        
        x = self.conv64(x)
        x = F.relu(self.bn64(x)) 
        x = self.pool(x)
        
        x = self.conv128_1(x)
        x = F.relu(self.bn128(x))
        x = self.conv128_2(x)
        x = F.relu(self.bn128(x))      
        x = self.pool(x)
        
        x = self.conv256_1(x)
        x = F.relu(self.bn256(x))
        x = self.conv256_2(x)
        x = F.relu(self.bn256(x))      
        x = self.pool(x)
        
        x = self.pool_2(x)
        x = x.view(-1, 256) 

        x = self.fc1(x)
        x = self.fc2(x)


        return x
    
    def predict(self, x):
        logits = self.forward(x)
        return F.softmax(logits)
 
