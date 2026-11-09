import torch
from time import gmtime, strftime
import os


# init modelSaver object before training
class modelSaver:
    def __init__(self, path):
        self.path = path + strftime("/ranAt_%Y_%m_%d__%H_%M_%S", gmtime())
        self.vllist = []
        self.epochcounter = 1
        # create temp directory
        os.mkdir(self.path)

    # save model after testing
    def save_model(self, model, vl):
        self.vllist.append((self.epochcounter, vl))
        filename = self.path + "/" + "testAcc" + str(round(vl.item(), 8)).replace(".", "_") + ".pth"
        torch.save(model.state_dict(), filename)
        self.epochcounter += 1

    # deletes all model with higher validation loss, than the lowest
    def get_best_moodel(self):
        # gets epochnr of lowest validationloss model
        bestEp = min(self.vllist, key=lambda item: item[1])
        allfiles = os.listdir(self.path)
        allfiles.remove(bestEp)
        for i in allfiles:
            os.remove(i)
