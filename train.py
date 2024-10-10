from torch.utils.data import DataLoader
import torch
from UNet3D import UNet3D
from torch.nn import CrossEntropyLoss
from torch.optim import Adam

from dataloader import BratsDataset
import os

def train():
    device = torch.device("cuda")

    train_dataloader = DataLoader(BratsDataset(), batch_size=1, shuffle=True)

    model = UNet3D(1, 1)

    torch.nn.DataParallel(model)
    model.to(device)

    criterion = CrossEntropyLoss()
    optim = Adam(model.parameters())

    for e in range(200):
        print("Epoch ", e + 1, " / 200")

        training_loss = 0

        model.train()

        for data, labels in train_dataloader:
            data, labels = data.to(device), labels.to(device)

            print(data.shape)

            optim.zero_grad()

            target = model(data)

            loss = criterion(target, labels)
            training_loss += loss.item()

            loss.backward()
            optim.step()

        print("Training loss = ", (training_loss / len(train_dataloader)))


train()