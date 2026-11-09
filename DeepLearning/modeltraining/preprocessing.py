from abc import ABC
from PIL import Image
import torch
from torch.utils.data.dataset import Dataset

import os
import pandas as pd
from torchvision.io import read_image
import numpy as np
from torch.utils.data import DataLoader, SubsetRandomSampler


class Imageloader(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


class custom_metrics:
    def __init__(self, truth, pred):
        self.truth = truth
        self.pred = pred
        self.calc_n = self.truth - self.pred
        self.calc_p = self.truth + self.pred

    def all_values(self):
        fp = sum(self.calc_n == -1).item()
        fn = sum(self.calc_n == 1).item()
        tp = sum(self.calc_p == 2).item()
        tn = sum(self.calc_p == 0).item()
        return [fp, fn, tp, tn]

    @staticmethod
    def all_metrics(l):
        total = sum(l)
        fp, fn, tp, tn = l
        if tp + fp > 0:
            precision = tp / (tp + fp)
        else:
            precision = 0
        if tp + fp > 0:
            recall = tp / (tp + fn)
        else:
            recall = 0

        accuracy = (tp + tn) / total

        if precision + recall > 0:
            f1 = 2 * ((precision * recall) / (precision + recall))

        else:
            f1 = 0
        return {"accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1
                }


class data_split:
    def __init__(self, data):
        self.data = data

    def splitter(self, validation_split=0.2, shuffle_dataset=True):
        # Creating data indices for training and validation splits:
        dataset_size = len(self.data)
        indices = list(range(dataset_size))
        split = int(np.floor(validation_split * dataset_size))

        if shuffle_dataset:
            np.random.seed(42)
            np.random.shuffle(indices)

        train_indices, val_indices = indices[split:], indices[:split]

        # Creating data samplers
        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)

        return train_sampler, valid_sampler