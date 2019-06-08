import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
import torch.utils.data as utils
import time
import pdb
from torch.utils.data.sampler import SubsetRandomSampler


torch.manual_seed(1)    # reproducible


class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        '''
        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(32, 32, kernel_size=5)
        self.conv3 = torch.nn.Conv2d(32, 64, kernel_size=2)
        self.fc1 = torch.nn.Linear(9 * 64, 200)
        self.fc2 = torch.nn.Linear(200, 10)
        '''
        self.conv1 = torch.nn.Conv2d(3, 16, 3)
        self.conv2 = torch.nn.Conv2d(16, 32, 3) 
        self.conv3 = torch.nn.Conv2d(32, 64, 5)
        #self.conv4 = torch.nn.Conv2d(64, 128, 3,padding=1)
        self.fc = torch.nn.Linear(64, 10)
        self.pool = torch.nn.MaxPool2d(2, 2)
        #self.fc2 = torch.nn.Linear(120, 84)
        #self.fc3 = torch.nn.Linear(84, 10)

    def forward(self, x):
        '''
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2))
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc1(x.view(-1, 3* 32*32)))
        x = self.fc2(x)
        '''
        x = self.conv1(x)
        x = F.relu(x)
        #x = F.relu(torch.nn.BatchNorm2d(x),implace=True)
        x = self.pool(x)
   
        x = self.conv2(x)
        x = F.relu(x)
        #x = F.relu(torch.nn.BatchNorm2d(x),implace=True)
        x = self.pool(x)

        x = self.conv3(x)
        x = F.relu(x)
        #x = F.relu(torch.nn.BatchNorm2d(x),implace=True)
        x = self.pool(x)

        
        x = x.view(-1, 64)       
        x = x = self.fc(x)

        return x
 
