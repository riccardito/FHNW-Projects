# imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision.transforms import transforms
from tqdm import tqdm
import numpy as np
import json
import matplotlib.pyplot as plt
import torchvision.models as models

from flashtorch.utils import apply_transforms, load_image
from flashtorch.saliency import Backprop

from preprocessing import Imageloader, custom_metrics
from cnn import resNet152, riciNet

with open("../modeltraining/modeltemplates/params.json") as json_file:
    params = json.load(json_file)
annotations_file = "/scratch/snx3000/rnef/Chest/allLabels.csv"
img_dir = "/scratch/snx3000/rnef/Chest"
model_dir = "../models/ranAt_2022_04_26__08_30_19/EpNr1vL0.5311"
seed = 42
torch.manual_seed(seed)
wb = False
pretrained = False
torch.cuda.empty_cache()


def main():
    outputsize = 2

    # load model
    cNet = riciNet(outputsize)
    cNet.load_state_dict(torch.load(model_dir))
    cNet.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose(
        [transforms.ToPILImage(),
         transforms.Grayscale(num_output_channels=1),
         transforms.Resize((224, 224)),
         transforms.ToTensor(),
         transforms.Normalize((0.5), (0.5)),
         ]
    )

    dataset = Imageloader(annotations_file, img_dir, transform)
    test_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=params["batchsize"],
                                              pin_memory=True,
                                              num_workers=8)

    with torch.no_grad():
        total = 0
        l = np.array([0, 0, 0, 0])
        for i, data in enumerate(tqdm(test_loader, 0)):
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = cNet(images)

            backprop = Backprop(cNet)
            backprop.visualize(images, 84, guided=True)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            cm_test = custom_metrics(labels, predicted)
            l_temp = cm_test.all_values()
            l += l_temp

        res_dict = cm_test.all_metrics(l)
        print(f'\nAccuracy of the network on test images: {res_dict["accuracy"]} %')
        print(res_dict)

    print('Finished Training')

if __name__ == "__main__":
    main()
