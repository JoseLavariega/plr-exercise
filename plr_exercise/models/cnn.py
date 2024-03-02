from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR


class Net(nn.Module):
    '''
    Defines Neural Network architecture, for a Convolutional Neural Network
    '''
    def __init__(self):
        '''
        Initializes Neural network Layers. Read from top to bottom. 
        Convolutional Layer 1: 1 input, 32 output channels, Kernel 3x3
        Convolutional Layer 2: 32 input, 64 output, Kernel 3x3
        Droupout Layer 1: 0.25 Dropout Rate
        Dropout Layer 2: 0.75 Dropout Rate
        Fully Connected 1: 9216 inputs, 128 outputs
        Fully Connected 2: 128 inputs, 10 outputs (digits)
        '''
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        '''
        Specifies architecture of the forward pass (images to outputs)
        Applies convolutional and activation layers
        Max Pools and Dropout operations. 
        Flattens the intermediary output to pass through fully connected layers. 
        Activation and Dropout to the Final Layer
        Log-Softmax classification

        Output: torch.Tensor: Logarithm of softmax probabilities for each class
        '''
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output