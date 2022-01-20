# -*- coding: utf-8 -*-


import numpy as np
import torch
# import torch.nn as nn
from torch.utils.data import DataLoader
# import torch.nn.functional as F

from model import UNet
from data_load_fixed import BraTS_dataset


# Hyperparameters
LOAD_MODEL = False
LOADPATH = "model_hpc_ba2-lr0001-ep100-bce.pth.tar"
SAVEPATH = "model_hpc_ba2-lr0001-ep300-bce.pth.tar"
LOSS_PATH = "loss_array_ba2-lr0001-ep300-bce.npy"

LEARNING_RATE = 0.0001
BATCH_SIZE = 2
NUM_EPOCHS = 300


def training():
    # Parameters
    model = UNet()
    out = 0
    mask = 0
    
    # Set device, send model to device
    device = "cuda" if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    # Loss function
    loss_func = torch.nn.BCELoss()

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Dataload
    train_data = BraTS_dataset(image_dir = "/work3/s214633/train_data/train/images",
                               mask_dir = "/work3/s214633/train_data/train/masks")
    dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

    # Load model after model and optimizer initialization
    if LOAD_MODEL:
        checkpoint = torch.load(LOADPATH)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch_nr = checkpoint['epoch']
        loss = checkpoint['loss']
        total_loss = checkpoint['total_loss']
        loss_pr_epoch = checkpoint['loss_pr_epoch']
        
        # print("Model's state_dict:")
        # for param_tensor in model.state_dict():
        #     print(param_tensor, "\t", model.state_dict()[param_tensor].size())
        print(f"Loaded model: {LOADPATH}")
        
    else:
        epoch_nr = 1
        total_loss = []
        loss_pr_epoch = []

    # Training mode initialized
    model.train()

    # Training loop
    for epoch in range(NUM_EPOCHS):
        epoch_loss = []
        for i, (feature, mask) in enumerate(dataloader):
            # Dataload. Send to device. 
            feature, mask = feature.float(), mask.float()
            feature, mask = feature.to(device), mask.to(device)
            
            # Data to model
            out = model(feature)
            
            # Calculate loss, do backpropagation and step optimizer
            loss = loss_func(out, mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Save loss
            loss_np = loss.detach().cpu()
            loss_np = loss_np.numpy()
            total_loss.append(loss_np)
            epoch_loss.append(loss_np)
            
            # Print loss
            print(f"Epoch {epoch_nr}, batch {i}: loss = {float(loss)}")
            
        # Save average loss for each epoch
        epoch_loss = np.mean(epoch_loss)
        loss_pr_epoch.append(epoch_loss)
        print(f"Loss for epoch {epoch_nr}: {epoch_loss}")
        
        # Epoch counter
        epoch_nr += 1
        
        # Save model at 100 and 200 epochs
        if epoch == 100:
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch_nr,
                'loss': loss,
                'total_loss': total_loss,
                'loss_pr_epoch': loss_pr_epoch
                }, "model_hpc_ba2-lr00001-ep100of300-bce.pth.tar")            
        if epoch == 200:
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch_nr,
                'loss': loss,
                'total_loss': total_loss,
                'loss_pr_epoch': loss_pr_epoch
                }, "model_hpc_ba2-lr00001-ep200of300-bce.pth.tar")
        
        
    # Save the model in the end
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch_nr,
        'loss': loss,
        'total_loss': total_loss,
        'loss_pr_epoch': loss_pr_epoch
        }, SAVEPATH)

    # Save array of loss
    loss_arr = np.array(total_loss)
    np.save(LOSS_PATH, loss_arr)

    return


if __name__ == "__main__":
    training()
    
    print("-"*40)
    print("All Done")
    
