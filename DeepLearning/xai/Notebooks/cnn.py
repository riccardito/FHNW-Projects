# imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
import os
import seaborn as sns
from tqdm import tqdm

from collections import Counter

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms

from torchvision import transforms
import torchvision
from PIL import Image

random.seed(42)
import nltk
import json


class resNet152(nn.Module):
    def __init__(self, output):
        super(resNet152, self).__init__()
        self.inception = models.resnet152(pretrained=False)
        self.inception.fc = nn.Linear(self.inception.fc.in_features, output)

    def forward(self, images):
        features = self.inception(images)
        return features


class riciNet(nn.Module):
    def __init__(self, output):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.fc1 = nn.Linear(4 * 4 * 256, 128)
        self.fc2 = nn.Linear(128, 84)
        self.fc3 = nn.Linear(84, 10)
        self.fc4 = nn.Linear(10, output)

    def forward(self, x):
        x = self.bn1(self.conv1(x))
        x = self.pool1(self.relu1(x))
        x = self.bn2(self.conv2(x))
        x = self.pool2(self.relu2(x))
        x = self.bn3(self.conv3(x))
        x = self.pool3(self.relu3(x))
        s = x.view((x.shape[0], -1))
        s = self.fc1(s)
        s = self.fc2(s)
        s = self.fc3(s)
        s = self.fc4(s)
        return s
