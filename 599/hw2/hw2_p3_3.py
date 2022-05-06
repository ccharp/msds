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

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5*5 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square, you can specify with a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


    def fit(self, batches, loss_fn, optimizer, epochs_n=10): # TODO: change back to 10
        loss_per_epoch = []

        for i in range(epochs_n):
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

# TODO: all of these *Experiment functions can be generalized further
def sgdExperiments(learning_rates, batch_sizes):
    experiments = []
    for batch_size in batch_sizes:
        for lr in learning_rates:
            net = Net()
            experiments.append({
                "net": net,
                "optimizer": optim.SGD(net.parameters(), lr=lr),
                "batch_size": batch_size,
            })

    def plot(exps_data):
        do_plot(
            exps_data,
            batch_sizes,
            learning_rates,
            "Batch Size", "Learning Rate",
            "SGD",
        )

    return {"plot_fn": plot, "exps": experiments}


def gen_data(n):
    def gen_d_sphere(r, d):
        v = np.random.normal(0, r, d)  
        d = np.sum(v**2) **(0.5)
        return v/d

    xs = [gen_d_sphere(1, 9) for _ in range(n)]
    ys = "TODO"

    return "TODO"

# Declare the experiments we're going to run. This is the program! 
experiments = [ # [Hypers]
    sgdExperiments([0.001, 0.01, 0.1], [60000]),
    sgdExperiments([0.001, 0.01, 0.1], [128, 512]),
]

for exp_class in experiments:
    history = []

    for exp in exp_class["exps"]:
        net = exp["net"]

        train_batches = DataLoader(train_data_raw, batch_size=exp["batch_size"], shuffle=True)

        history.append(net.fit(
            train_batches, 
            nn.CrossEntropyLoss(),
            exp["optimizer"],
        ))

    exp_class["plot_fn"](history)
