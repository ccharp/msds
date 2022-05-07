import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.kernel_ridge import KernelRidge

import numpy as np
import pandas as pd
import seaborn as sns

DEBUG_MODE = True 
def pp(s):
    if(DEBUG_MODE):
        print(s)

class Net(nn.Module):

    def __init__(self, depth, width, initVariance):
        super(Net, self).__init__()

        self.preLayers = nn.ModuleList()
        for _ in range(depth):
            newLayer = nn.Linear(width, width)
            nn.init.normal_(newLayer.weight, 0, np.sqrt(initVariance))
            self.preLayers.append(newLayer)

        self.outputLayer = nn.Linear(width, 1)

    def forward(self, x):
        x = torch.flatten(x, 2, 3)

        for l in self.preLayers:
            x = F.relu(l(x))

        y = F.relu(self.outputLayer(x))
        return torch.squeeze(y)


    def fit(self, batches, loss_fn, optimizer, epochs_n=10): # TODO: change back to 10
        loss_per_epoch = []

        for i in range(epochs_n): # TODO no batches
            batch_loss = 0
            pp(f"### Epoch #{i}")

            self.train()
            for image, label in batches:
                optimizer.zero_grad()
                net_output = self(image)
                loss_output = loss_fn(net_output, label)
                loss_output.backward()
                optimizer.step()
                batch_loss += loss_output.item()

            loss_per_epoch.append(batch_loss)
            pp(f"   loss for this epoch: {loss_per_epoch[i]}" )

        return loss_per_epoch 

def genData(n, m):
    def relu(num):
        return 0 if num < 0 else num
    def genNSphere(r, d):
        v = np.random.normal(0, r, d)  
        d = np.sum(v**2) **(0.5)
        return v/d

    xss = [genNSphere(1, 9) for _ in range(n)]
    ys = []
    for xs in xss:
        ys.append(sum(map(relu, xs))/m)

    return (np.array(xss), np.array(ys))

############ BEGIN APPLICATION ############ 
ns = [20, 40, 80, 160]
d = 10
m = 5

def calcError(preds, targets):
    return 0 # TODO

def arcKernel(x1: np.array, x2: np.array):
    x1Tx2 = np.transpose(x1) * x2
    return x1Tx2*(np.pi - np.arccos(x1Tx2))/(2*np.pi)

def plotResults(df):
    sns.color_palette("Set2")

    df['Loss'] = np.log(df['Loss'].astype(float))

    title = f"Loss per Epoch with"
    plot = sns.lineplot(
        data=df, x="N", y="Loss", hue="ModelType", ci=None
    ).set_title(title)
    plot.figure.savefig(f"{title}_{d}_{a}.png")
    plot.figure.clf()

historyDf = pd.DataFrame(columns=[
            "N", "Loss", "ModelType"
])

def addHistoryRow(df, loss, modelType):
    new_row = pd.DataFrame(columns=df.columns)
    new_row.loc[0] = [loss, modelType]
    df = pd.concat([df, new_row], ignore_index=True)
    return df

for n in ns:
    trainXs, trainYs = genData(n, m)
    testXs, testYs = genData(1000, m)

    net = Net(depth=m, width=d)
    _ = net.fit(trainXs, trainYs)
    nnError = calcError(net(testXs, testYs))
    historyDf = addHistoryRow(historyDf, nnError, "Neureal Network")

    krr = KernelRidge(kernel=arcKernel)
    krr.fit(trainXs, trainYs)
    krrError = calcError(krr.predict(testXs))
    historyDf = addHistoryRow(historyDf, krrError, "Kernel Regression")


