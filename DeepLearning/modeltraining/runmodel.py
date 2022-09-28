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
from cnn import resNet50, riciNet, AlexNet
from save_model import modelSaver

with open("../modeltraining/modeltemplates/params.json") as json_file:
    params = json.load(json_file)
# annotations_file = "/scratch/snx3000/rnef/Chest/allLabels.csv"
# annotations_file = "/scratch/snx3000/rnef/Chest/sampleLabels.csv"
annotations_file = "../data/Chest/BBox_List_2017_pneumo_cardio_lean.csv"

img_dir = "../data/Chest"
model_dir = "../models/"

seed = 42
torch.manual_seed(seed)
wb = False
pretrained = False
saveModel = True
torch.cuda.empty_cache()


def main():
    num_classes = 1
    cNet = AlexNet(num_classes)

    # W&B initialising
    if wb:
        wandb.init(project="ip6", entity="riccci", name="S20k_resnet50_res512")
        wandb.config = params

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose(
        [transforms.ToPILImage(),
         transforms.Grayscale(num_output_channels=3),
         transforms.Resize((224, 224)),
         # transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize((0.5), (0.5)),
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

    # initialize model and loss
    model = cNet.to(device)
    # model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16_bn', pretrained=False).to(device)
    # model = xrv.models.get_model(weights="densenet121-res224-all").to(device)
    # model = xrv.models.ResNet(weights="resnet50-res512-all").to(device)
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=params["learning_rate"])

    # activate drop-ofs
    model.train()

    for epoch in range(params["epochs"]):
        print(f"*************Training*************** \n epoch {epoch + 1} from {params['epochs']}")
        total_train = 0.0
        correct_train = 0.0
        # training
        for i, data in enumerate(tqdm(train_loader, 0)):

            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize

            outputs = model(inputs)
            # print(outputs)
            # x = nn.Sigmoid()
            loss = criterion(outputs, labels.view((labels.shape[0], -1)).float())
            loss.backward()
            optimizer.step()

            if pretrained:
                # Only finetune the CNN, only works for resnet
                for name, param in model.inception.named_parameters():
                    if "fc.weight" in name or "fc.bias" in name:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
            _, predicted_train = torch.max(outputs.data, 1)

            total_train += labels.size(0)
            correct_train += (predicted_train == labels).sum().item()

            # logging results
            if wb:
                wandb.log({"loss": loss})

        train_accuracy = correct_train / total_train
        if wb:
            wandb.log({"train accuracy": train_accuracy})

        # testing
        with torch.no_grad():
            model.eval()
            print(f"*************Testing*************** \n epoch {epoch + 1} from {params['epochs']}")
            total = 0
            l = np.array([0, 0, 0, 0])
            for i, data in enumerate(tqdm(test_loader, 0)):
                images, labels = data
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)

                # x = nn.Sigmoid()
                val_loss = criterion(outputs, labels.view((labels.shape[0], -1)).float())

                _, predicted = torch.max(outputs.data, 1)

                total += labels.size(0)
                cm_test = custom_metrics(labels, predicted)
                l_temp = cm_test.all_values()
                l += l_temp
                # wandb.log({"conf_mat" : wandb.plot.confusion_matrix(probs=None,y_true=labels.cpu().numpy(), preds=predicted.cpu().numpy(), class_names=["normal","abnormal"])})
                # print(f"truth : {labels.cpu().numpy()},\npredicted: {predicted.cpu().numpy()}")
                # wandb.log({"pr": wandb.plot.pr_curve(labels.numpy(), predicted.numpy())})
                # wandb.log({"roc": wandb.plot.roc_curve(labels, predicted,
                # labels=None, classes_to_plot=None)})

        res_dict = cm_test.all_metrics(l)
        print(f'\nAccuracy of the network on test images: {res_dict["accuracy"]} %')
        if wb:
            wandb.log(res_dict)

        print(res_dict)

        # save model
        if saveModel:
            modelsaver = modelSaver(model_dir)
            modelsaver.save_model(model, val_loss)

    print('Finished Training')


if __name__ == "__main__":
    main()
