import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

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

        #self.conv1 = nn.Conv2d(1, 6, 5)
        #self.conv2 = nn.Conv2d(6, 16, 5)

        self.preLayers = nn.ModuleList()
        for _ in range(depth):
            newLayer = nn.Linear(width, width)
            #pp(f"initVariance: {initVariance}")
            nn.init.normal_(newLayer.weight, 0, np.sqrt(initVariance))
            self.preLayers.append(newLayer)

        self.outputLayer = nn.Linear(width, 1)
        self.outSig = nn.Sigmoid()

    def forward(self, x):

        x = torch.flatten(x, 2, 3)

        for l in self.preLayers:
            x = F.leaky_relu(l(x), self.alpha) 

        x = F.leaky_relu(self.outputLayer(x), self.alpha)

        return torch.squeeze(self.outSig(x))


    def fit(self, batches, learningRate, epochs_n=5): # TODO: change back to 10
        loss_per_epoch = []

        optimizer = optim.SGD(self.parameters(), lr=learningRate)
        lossFn = nn.BCELoss()

        for i in range(epochs_n):
            batch_loss = 0

            self.train()
            for image, label in batches:
                optimizer.zero_grad()
                y_hat = self(image)
                #pp(f"y_hat: {y_hat}" )
                #pp(f"label: {label}" )
                loss = lossFn(y_hat, label.float())
                loss.backward()
                optimizer.step()
                #pp(f"loss: {loss}")
                batch_loss += loss.item()

            loss_per_epoch.append(batch_loss)

        return loss_per_epoch 

# setup data
trainDataRaw = MNIST(
    root='data', 
    train=True, 
    transform=transforms.Compose([
        transforms.Resize(16),
        transforms.ToTensor(),
    ]),
    download=True
)


def addToDf(df: pd.DataFrame, history, actType):
    for i, loss in enumerate(history):
        new_row = pd.DataFrame(columns=df.columns)
        new_row.loc[0] = [i, loss, actType]
        df = pd.concat([df, new_row], ignore_index=True)

    return df

def plotResults(df, d, a, lr):
    sns.color_palette("Set2")

    df['Loss'] = np.log(df['Loss'].astype(float))

    title = f"Loss per Epoch with alpha={a} and depth={d}"
    plot = sns.lineplot(
        data=df, x="Epoch", y="Loss", hue="ActivationType", ci=None
    ).set_title(title)
    plot.figure.savefig(f"{title}_{d}_{a}.png")
    plot.figure.clf()

########### DO THE STUFF ###########
batchSize = 64
width = 256
depths = [10, 20, 30, 40]
alphas = [2, 1, 0.5, 0.1]

keepIdxs = (trainDataRaw.targets==0) | (trainDataRaw.targets==1)
trainDataRaw.targets = trainDataRaw.targets[keepIdxs]
trainDataRaw.data = trainDataRaw.data[keepIdxs]

trainBatches = DataLoader(trainDataRaw, batch_size=batchSize, shuffle=True)

initFuncs = {
    "relu": (lambda alpha: 2/width),
    "prelu": (lambda alpha: 2/(width*(1 + alpha**2))),
}

history = []
learningRate = 0.01

for depth in depths:
    for alpha in alphas:

        historyDf = pd.DataFrame(columns=[
            "Epoch", "Loss", "ActivationType"
        ])

        for activationType, initFunc in initFuncs.items():
            net = Net(depth, alpha, width, initFunc(alpha))
            losses = net.fit(trainBatches, learningRate, epochs_n=5)
            pp(f"Losses: {losses}")
            historyDf = addToDf(historyDf, losses, activationType) # TODO
        
        print(historyDf.head())
        plotResults(historyDf, d=depth, a=alpha, lr=learningRate) 
