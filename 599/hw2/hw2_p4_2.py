import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision
from torchvision.datasets import MNIST
from torchvision import transforms

import numpy as np
import pandas as pd
import seaborn as sns

DEBUG_MODE = True 
def pp(s):
    if(DEBUG_MODE):
        print(s)

class Net(nn.Module):

    def __init__(self, depth, alpha, width, initVariance):
        super(Net, self).__init__()

        self.alpha = alpha
        self.preLayers = [nn.Linear(width, width) for _ in range(depth - 1)]
        self.outputLayer = nn.Linear(width, 1)

        for L, layer in self.modules():
            if isinstance(layer, nn.Linear):
                layer.weight.data.normal_(mean=0, std=initVariance **0.5)

    def forward(self, x):
        x = "" # TODO: initialize x to the correct type? Python is weird. 
        for l in self.preLayers:
            x = F.leaky_relu(l, self.alpha) 
        x = F.sigmoid(self.outputLayer)
        return x


    def fit(self, batches, learningRate, epochs_n=10): # TODO: change back to 10
        loss_per_epoch = []

        optimizer = optim.SGD(self.parameters(), lr=learningRate),
        lossFn = nn.BCELoss(),

        for i in range(epochs_n):
            batch_loss = 0
            pp(f"### Epoch #{i}")

            self.train()
            for image, label in batches:
                optimizer.zero_grad()
                net_output = self(image)
                loss_output = lossFn(net_output, label)
                loss_output.backward()
                optimizer.step()
                batch_loss += loss_output.item()

            loss_per_epoch.append(batch_loss)
            pp(f"   loss for this epoch: {loss_per_epoch[i]}" )

        return loss_per_epoch 

# setup data
train_data_raw = MNIST(
    root='data', 
    train=True, 
    transform=torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(32),
            torchvision.transforms.ToTensor(),
        ]), 
    download=True
)


def addToDf(df: pd.DataFrame, history, actType):
    for i, loss in enumerate(history):
        new_row = pd.DataFrame(columns=df.columns)
        new_row.loc[0] = [i, loss, actType]
        df = pd.concat([df, new_row], ignore_index=True)

    return df

def plotResults(df, d, a, lr, title):
    sns.color_palette("Set2")
    plot = sns.lineplot(
        data=df, x="Epoch", y="Loss", hue="ActivationType", ci=None
    ).set_title(title)
    plot.figure.savefig(f"{title}_{d}_{a}.png")
    plot.figure.clf()

########### DO THE STUFF ###########
trainBatches = DataLoader(train_data_raw, batch_size=64, shuffle=True)
# TODO: filter 

width = 256
depths = [10] #[10, 20, 30, 40]
alphas = [2] #[2, 1, 0.5, 0.1]

initFuncs = {
    "relu": lambda alpha: 2/width,
    "prelu": lambda alpha: 2/(width(1 - alpha**2))
}

history = []
learningRate = 0.2

for depth in depths:
    for alpha in alphas:

        historyDf = pd.DataFrame(columns=[
            "Loss", "Epoch", "ActivationType"
        ])

        for activatinType, initFunc in initFuncs.items():
            net = Net(depth, alpha, width, initFunc(alpha))
            losses = net.fit(trainBatches, learningRate, epochs_n=5)

            historyDf = addToDf()
        
        plotResults(historyDf, d=depth, a=alpha, lr=learningRate) 
