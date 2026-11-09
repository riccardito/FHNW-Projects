# imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
import os
from tqdm import tqdm

from collections import Counter
from collections import OrderedDict

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
import json


class AlexNet(nn.Module):
    def __init__(self, num_classes: int = 2, dropout: float = 0.5) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4608),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4608, 1152),
            nn.ReLU(inplace=True),
            nn.Linear(1152, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class resNet18(nn.Module):
    def __init__(self, output):
        super(resNet18, self).__init__()
        self.inception = models.resnet18(pretrained=False)
        self.inception.fc = nn.Linear(self.inception.fc.in_features, output)

    def forward(self, images):
        features = self.inception(images)
        return features


class resNet50(nn.Module):
    def __init__(self, output, pretrained):
        super(resNet50, self).__init__()
        self.pretrained = pretrained
        self.inception = models.resnet50(pretrained=self.pretrained)
        self.inception.fc = nn.Linear(self.inception.fc.in_features, output)

    def forward(self, images):
        features = self.inception(images)
        return features


class resNet152(nn.Module):
    def __init__(self, output, pretrained):
        super(resNet152, self).__init__()
        self.pretrained = pretrained
        self.inception = models.resnet152(pretrained=self.pretrained)
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

        self.fc1 = nn.Linear(200704, 128)
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


class riciNet_small(nn.Module):
    def __init__(self, output):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=7)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=4)
        self.fc1 = nn.Linear(46656, 128)
        self.fc2 = nn.Linear(128, output)
        self.drop1 = nn.Dropout()
        self.drop2 = nn.Dropout()

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(self.relu1(x))
        s = x.view((x.shape[0], -1))
        s = self.drop1(s)
        s = self.fc1(s)
        s = self.drop2(s)
        s = self.fc2(s)
        return s


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class Dense_Block(nn.Module):
    def __init__(self, in_channels):
        super(Dense_Block, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(num_features=in_channels)
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=96, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=32, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        bn = self.bn(x)
        conv1 = self.relu(self.conv1(bn))
        conv2 = self.relu(self.conv2(conv1))
        # Concatenate in channel dimension
        c2_dense = self.relu(torch.cat([conv1, conv2], 1))
        conv3 = self.relu(self.conv3(c2_dense))
        c3_dense = self.relu(torch.cat([conv1, conv2, conv3], 1))

        conv4 = self.relu(self.conv4(c3_dense))
        c4_dense = self.relu(torch.cat([conv1, conv2, conv3, conv4], 1))

        conv5 = self.relu(self.conv5(c4_dense))
        c5_dense = self.relu(torch.cat([conv1, conv2, conv3, conv4, conv5], 1))

        return c5_dense


class Transition_Layer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Transition_Layer, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=False)
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        bn = self.bn(self.relu(self.conv(x)))
        out = self.avg_pool(bn)
        return out


class DenseNet(nn.Module):
    def __init__(self, nr_classes):
        super(DenseNet, self).__init__()

        self.lowconv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, padding=3, bias=False)
        self.relu = nn.ReLU()

        # Make Dense Blocks
        self.denseblock1 = self._make_dense_block(Dense_Block, 64)
        self.denseblock2 = self._make_dense_block(Dense_Block, 128)
        # self.denseblock3 = self._make_dense_block(Dense_Block, 128)    # Make transition Layers
        self.transitionLayer1 = self._make_transition_layer(Transition_Layer, in_channels=160, out_channels=128)
        self.transitionLayer2 = self._make_transition_layer(Transition_Layer, in_channels=160, out_channels=64)
        # self.transitionLayer3 = self._make_transition_layer(Transition_Layer, in_channels = 160, out_channels = 64)
        # Classifier
        self.bn = nn.BatchNorm2d(num_features=64)
        self.pre_classifier = nn.Linear(200704, 112)
        self.drop = nn.Dropout()
        self.classifier = nn.Linear(112, nr_classes)

    def _make_dense_block(self, block, in_channels):
        layers = []
        layers.append(block(in_channels))
        return nn.Sequential(*layers)

    def _make_transition_layer(self, layer, in_channels, out_channels):
        modules = []
        modules.append(layer(in_channels, out_channels))
        return nn.Sequential(*modules)

    def forward(self, x):
        out = self.relu(self.lowconv(x))
        out = self.denseblock1(out)
        out = self.transitionLayer1(out)
        out = self.denseblock2(out)
        out = self.transitionLayer2(out)

        # out = self.denseblock3(out)
        # out = self.transitionLayer3(out)

        out = self.bn(out)
        out = out.view((out.shape[0], -1))

        out = self.pre_classifier(out)
        out = self.drop(out)
        out = self.classifier(out)
        return out