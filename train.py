from torch.utils.data import DataLoader
import torch
from UNet2D import UNet2D
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from tqdm import tqdm
from Brats_2D_DS import BratsDataset
import os

def train():
    device = torch.device("mps")

    train_dataloader = DataLoader(BratsDataset(), batch_size=128, shuffle=True)

    model = UNet2D(True)

    torch.nn.DataParallel(model)
    model.to(device)

    criterion = CrossEntropyLoss()
    optim = Adam(model.parameters())

    for e in range(200):
        print("Epoch ", e + 1, " / 200")

        training_loss = 0

        model.train()

        for data, labels in tqdm(train_dataloader):
            data, labels = data.to(device=device, dtype=torch.float), labels.to(device=device, dtype=torch.float)

            optim.zero_grad()

            target = model(data)

            loss = criterion(target, labels)
            training_loss += loss.item()

            loss.backward()
            optim.step()

        print("Training loss = ", (training_loss / len(train_dataloader)))


train()