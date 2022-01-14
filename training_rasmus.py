# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 13:13:23 2022

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
from utilities import plot_prediction_mask
from utilities import save_checkpoint
from utilities import load_checkpoint
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def training():
    # parameters
    learning_rate = 0.002
    num_epochs = 1
    model = UNet()
    out = 0
    mask = 0

    #Hvis du gerne vil plotte efter hver loss
    plot_in_training_loop = False

    # Sæt til True hvis du gerne vil køre videre med en loaded model,
    # skriv navnet på checkpointet du gerne vil loade her:
    load_model = True
    checkpoint_name = "16_epochs_BCEL_model.pth.tar"

    # Så det i en video, ved ikke om vi får brug for det i forhold til HPC
    device = "cuda" if torch.cuda.is_available() else torch.device("cpu")

    # initialization
    model.to(device)

    # Loss function
    # loss = torch.nn.CrossEntropyLoss()
    loss = torch.nn.BCELoss()

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Get the data
    train_data = BraTS_dataset(image_dir="./BraTS2020/train_valid_data/train/images",
                               mask_dir="./BraTS2020/train_valid_data/train/masks")
    # images/short_edit
    # masks/short_edit

    dataloader = DataLoader(train_data, batch_size=2, shuffle=True)

    # Training mode initialized
    model.train()

    # Choose the file / checkpoint you wish to load, and input its name here:
    if load_model:
        load_checkpoint(torch.load(checkpoint_name), model, optimizer)

    # Training loop
    all_loss = []
    for epoch in range(num_epochs):
        print(f"Epoch number: {epoch}")
        epoch_loss = []

        checkpoint = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
        if epoch == 1:
            save_checkpoint(checkpoint, "16_epochs_BCEL_model.pth.tar")

        for i, (feature, mask) in enumerate(dataloader):
            feature, mask = feature.float(), mask.float()
            out = model(feature)

            l = loss(out, mask)

            optimizer.zero_grad()
            l.backward()
            optimizer.step()

            epoch_loss += [l.detach().numpy()]
            all_loss += [l.detach().numpy()]

            if plot_in_training_loop:
                out_np = out.detach().numpy()
                mask_np = mask.detach().numpy()

                plot_2d_tensor(mask_np, out_np, 75)

            print(f"loss = {l}")
        print(f"Average loss at epoch {epoch} is : ", np.mean(epoch_loss))

    # out_np = out.detach().numpy()
    # mask_np = mask.detach().numpy()

    test_output = out
    print(type(test_output), np.shape(test_output))
    print(test_output[0, :, 75, 75, 75])

    plot_prediction_mask(out, mask)
    out_np = out.detach().numpy()
    mask_np = mask.detach().numpy()
    plot_2d_tensor(mask_np, out_np)

    return all_loss


if __name__ == "__main__":
    loss_list = training()
    plt.plot(np.array(loss_list))
    plt.show()
