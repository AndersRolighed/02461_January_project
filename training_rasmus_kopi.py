# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 12:23:29 2022

@author: Rasmus
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from model import UNet, double_convolution
from data_load import BraTS_dataset
from utilities import plot_2d_tensor

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def training():
    # parameters
    learning_rate = 0.002
    num_epochs = 1
    model = UNet()
    out = 0
    mask = 0
    plot_in_training_loop = True

    # Set til True hvis du gerne vil køre videre med en loaded model.
    load_model = False

    # Så det i en video, ved ikke om vi får brug for det i forhold til HPC
    device = "cuda" if torch.cuda.is_available() else torch.device("cpu")

    # initialization
    model.to(device)

    # Loss function
    loss = torch.nn.BCELoss()

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Get the data
    train_data = BraTS_dataset(image_dir="./BraTS2020/train_valid_data/train/images",
                               mask_dir="./BraTS2020/train_valid_data/train/masks")

    dataloader = DataLoader(train_data, batch_size=2, shuffle=True)

    # Training mode initialized
    model.train()

    if load_model:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model, optimizer)

    # Training loop
    all_loss = []
    for epoch in range(num_epochs):
        print(f"Epoch number: {epoch}")

        checkpoint = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
        if epoch == 5:
            save_checkpoint(checkpoint)

        for i, (feature, mask) in enumerate(dataloader):
            feature, mask = feature.float(), mask.float()
            out = model(feature)
            # print(out)
            # print(type(mask))

            l = loss(out, mask)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            
            all_loss += [l.detach().numpy()]
            plt.plot(all_loss)
            plt.show()
            
            if plot_in_training_loop == True:
                out_np = out.detach().numpy()
                mask_np = mask.detach().numpy()
    
                plot_2d_tensor(mask_np, out_np, 75)
                
            print(f"loss = {l}")

    # Save the model

    # FILE = "model.pth"
    # torch.save(model.state_dict(), FILE)

    # print("Model's state_dict:")
    # for param_tensor in model.state_dict():
        # print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    # print("Optimizer's state_dict:")
    # for var_name in optimizer.state_dict():
        # print(var_name, "\t", optimizer.state_dict()[var_name])

    out_np = out.detach().numpy()
    mask_np = mask.detach().numpy()

    plot_2d_tensor(mask_np, out_np, 75)

    return all_loss


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print('=> Saving checkpoint')
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])


if __name__ == "__main__":
    loss_list = training()
    plt.plot(np.array(loss_list))
    plt.show()
