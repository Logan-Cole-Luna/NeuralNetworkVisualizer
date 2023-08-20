# Importing necessary packages with comments
from sklearn.metrics import classification_report
# classification_report: Used to display a detailed classification report on our testing set
from torch.utils.data import random_split
# random_split: Constructs a random training/testing split from an input set of data
from torch.utils.data import DataLoader
# DataLoader: PyTorch’s awesome data loading utility that allows us to effortlessly build data pipelines to train our CNN
from torchvision.transforms import ToTensor
# ToTensor: A preprocessing function that converts input data into a PyTorch tensor for us automatically
from torchvision.datasets import KMNIST
# KMNIST: The Kuzushiji-MNIST dataset loader built into the PyTorch library
from torch.optim import Adam
# Adam: The optimizer we’ll use to train our neural network
from torch import nn
# nn: PyTorch’s neural network implementations
from torch.nn import Module
# can see how PyTorch implements neural networks using classes
from torch.nn import Conv2d
# Conv2d: PyTorch’s implementation of convolutional layers
from torch.nn import Linear
# Linear: Fully connected layers
from torch.nn import MaxPool2d
# MaxPool2d: Applies 2D max-pooling to reduce the spatial dimensions of the input volume
from torch.nn import ReLU
# ReLU: Our ReLU activation function
from torch.nn import LogSoftmax
# LogSoftmax: Used when building our softmax classifier to return the predicted probabilities of each class
from torch import flatten
import matplotlib.pyplot as plt
import numpy as np
import argparse
import torch
import time

# Inspiration: https://pyimagesearch.com/2021/07/19/pytorch-training-your-first-convolutional-neural-network-cnn/
# Importing necessary packages with comments
from sklearn.metrics import classification_report
from torch.utils.data import random_split, DataLoader

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder


# Define the CNN architecture
class AircraftCNN(nn.Module):
    def __init__(self, num_classes):
        super(AircraftCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 32 * 56 * 56)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Set up data loading and preprocessing
data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

data_dir = 'Airplane detection.v2i.coco'
dataset = ImageFolder(data_dir, transform=data_transform)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Create the CNN model
num_classes = len(dataset.classes)
model = AircraftCNN(num_classes)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    for images, labels in loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Save the trained model
torch.save(model.state_dict(), 'aircraft_model.pth')
