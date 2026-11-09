# imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler, WeightedRandomSampler
from torchvision.transforms import transforms
from tqdm import tqdm
import numpy as np
import json
import torchxrayvision as xrv

from preprocessing import Imageloader, custom_metrics, data_split
import wandb
from cnn2 import AlexNet
from save_model import modelSaver

with open("../modeltraining/modeltemplates/params.json") as json_file:
    params = json.load(json_file)
# annotations_file = "/scratch/snx3000/rnef/Chest/allLabels.csv"
# annotations_file = "/scratch/snx3000/rnef/Chest/sampleLabels.csv"
# annotations_file = "/scratch/snx3000/rnef/Chest/sampleLabels10k.csv"
# annotations_file = "/scratch/snx3000/rnef/Chest/sampleLabels2.csv"
# annotations_file = "/scratch/snx3000/rnef/Chest/sampleLabelshope50.csv"
annotations_file = "../data/Chest/BBox_List_2017_pneumo_cardio_lean.csv"

img_dir = "../data/Chest"
model_dir = "../models/"

seed = 333
torch.manual_seed(seed)
wb = False
pretrained = False
saveModel = True
torch.cuda.empty_cache()


def main():
    num_classes = 2
    model = AlexNet(num_classes)

    # W&B initialising
    if wb:
        wandb.init(project="ip6-3", entity="riccci", name="AlexNet_3d_cp_s333")
        wandb.config = params

    if saveModel:
        modelsaver = modelSaver(model_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose(
        [transforms.ToPILImage(),
         transforms.Grayscale(num_output_channels=3),
         transforms.Resize(224),
         transforms.ToTensor(),
         transforms.Normalize(mean=0.5, std=0.5)
         ]
    )

    dataset = Imageloader(annotations_file, img_dir, transform)

    # Split dataset into train and validation
    splitt = data_split(dataset)
    train_sampler, valid_sampler = splitt.splitter()

    train_loader = torch.utils.data.DataLoader(dataset,
                                               batch_size=params["batch_size"],
                                               sampler=train_sampler,
                                               pin_memory=True,
                                               num_workers=8)

    test_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=params["batch_size"],
                                              sampler=valid_sampler,
                                              pin_memory=True,
                                              num_workers=8)

    model = model.to(device)
    # model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16_bn', pretrained=False).to(device)
    # model = torch.hub.load('pytorch/vision:v0.10.0', 'densenet121', pretrained=True).to(device)
    # model = xrv.models.get_model(weights="densenet121-res224-nih").to(device)
    # model = xrv.models.ResNet(weights="resnet50-res512-all").to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=params["learning_rate"])

    for epoch in range(params["epochs"]):
        model.train()
        print(f"*************Training*************** \n epoch {epoch + 1} from {params['epochs']}")
        total_train = 0.0
        correct_train = 0.0
        # training

        # batch_index, (faces, labels) in enumerate(train_loader)
        for i, data in enumerate(tqdm(train_loader, 0)):

            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize

            outputs = model(inputs)
            # print(outputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if pretrained:
                # Only finetune the CNN, only works for resnet
                for name, param in model.inception.named_parameters():
                    if "fc.weight" in name or "fc.bias" in name:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False

            predicted_train = torch.argmax(outputs.data, dim=1)
            # print(f"pred={predicted_train}\ntruth={labels}")
            total_train += labels.size(0)
            correct_train += (predicted_train == labels).sum().item()

            # logging results
            if wb:
                wandb.log({"loss": loss})

        train_accuracy = correct_train / total_train
        if wb:
            wandb.log({"train accuracy": train_accuracy})
        print(f"Training ended with {train_accuracy * 100}%")

        # testing
        with torch.no_grad():
            model.eval()
            print(f"*************Testing*************** \n epoch {epoch + 1} from {params['epochs']}")
            total_test = 0.0
            l = np.array([0, 0, 0, 0])
            for i, data in enumerate(tqdm(test_loader, 0)):
                images_test, labels_test = data
                images_test, labels_test = images_test.to(device), labels_test.to(device)

                outputs_test = model(images_test)

                # x = nn.Sigmoid()
                val_loss = criterion(outputs_test, labels_test)
                predicted_test = torch.argmax(outputs_test.data, dim=1)
                # print(f"pred={predicted_test}\ntruth={labels_test}")
                total_test += labels_test.size(0)
                cm_test = custom_metrics(labels_test, predicted_test)
                l_temp = cm_test.all_values()
                l += l_temp

                #if wandb:
                    #wandb.log({"conf_mat": wandb.plot.confusion_matrix(probs=None, y_true=labels_test.cpu().numpy(),
                                                                       #preds=predicted_test.cpu().numpy(),
                                                                       #class_names=["cardiomaly", "pneumothorax"])})

                # print(f"truth : {labels.cpu().numpy()},\npredicted: {predicted.cpu().numpy()}")
                # wandb.log({"pr": wandb.plot.pr_curve(labels.numpy(), predicted.numpy())})
                # wandb.log({"roc": wandb.plot.roc_curve(labels, predicted,
                # labels=None, classes_to_plot=None)})

        res_dict = cm_test.all_metrics(l)
        print(f'\nAccuracy of the network on test images: {res_dict["accuracy"] * 100} %')
        if wb:
            wandb.log(res_dict)

        print(res_dict)

        # save model
        if saveModel:
            modelsaver.save_model(model, res_dict["accuracy"])
    if saveModel:
        modelsaver.get_best_moodel()
    print('Finished Training')


if __name__ == "__main__":
    main()
