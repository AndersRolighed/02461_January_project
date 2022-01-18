# -*- coding: utf-8 -*-


import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

from model import UNet
from data_load import BraTS_dataset


# Hyperparameters
LOAD_MODEL = False
# LOADPATH = "model_hpc_lr0001_ep50_1.pth.tar"
# LOADPATH = "15_epochs_BCEL_model.pth.tar"
LOADPATH = "model_hpc_batch1-0001-20.pth.tar"
SAVEPATH = "model_hpc_batch2-0001-300.pth.tar"
LOSS_PATH = "loss_array_batch2-0001-50.npy"

# LEARNING_RATE = 0.0001
LEARNING_RATE = 0.0001

BATCH_SIZE = 2
NUM_EPOCHS = 300


# ALPHA = 0.8
# GAMMA = 2

# class FocalLoss(nn.Module):
#     def __init__(self, weight=None, size_average=True):
#         super(FocalLoss, self).__init__()

#     def forward(self, inputs, targets, alpha=ALPHA, gamma=GAMMA, smooth=1):
        
#         #comment out if your model contains a sigmoid or equivalent activation layer
#         # inputs = F.sigmoid(inputs)       
        
#         #flatten label and prediction tensors
#         inputs = inputs.view(-1)
#         targets = targets.view(-1)
        
#         #first compute binary cross-entropy 
#         BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
#         BCE_EXP = torch.exp(-BCE)
#         focal_loss = alpha * (1-BCE_EXP)**gamma * BCE
                       
#         return focal_loss


def training():
    # parameters
    model = UNet()
    out = 0
    mask = 0
    
    # Så det i en video, ved ikke om vi får brug for det i forhold til HPC
    device = "cuda" if torch.cuda.is_available() else torch.device("cpu")

    # Initialization
    model.to(device)

    # Loss function
    loss_func = torch.nn.BCELoss()
    # loss_func = FocalLoss()

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


    for epoch in range(NUM_EPOCHS):
        epoch_loss = []
        for i, (feature, mask) in enumerate(dataloader):
            feature, mask = feature.float(), mask.float()
            feature, mask = feature.to(device), mask.to(device)
            
            out = model(feature)

            loss = loss_func(out, mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_np = loss.detach().cpu()
            loss_np = loss_np.numpy()
            total_loss.append(loss_np)
            epoch_loss.append(loss_np)
            
            print(f"Epoch {epoch_nr}, batch {i}: loss = {loss}")
            
        
        epoch_loss = np.mean(epoch_loss)
        loss_pr_epoch.append(epoch_loss)
        print(f"Loss for epoch {epoch_nr}: {epoch_loss}")
        epoch_nr += 1
        
        if epoch == 100:
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch_nr,
                'loss': loss,
                'total_loss': total_loss,
                'loss_pr_epoch': loss_pr_epoch
                }, "model_hpc_batch2-0001-100.pth.tar")
            
        if epoch == 200:
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch_nr,
                'loss': loss,
                'total_loss': total_loss,
                'loss_pr_epoch': loss_pr_epoch
                }, "model_hpc_batch2-0001-200.pth.tar")
        

    # Save the model
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch_nr,
        'loss': loss,
        'total_loss': total_loss,
        'loss_pr_epoch': loss_pr_epoch
        }, SAVEPATH)

    # Save loss array
    loss_arr = np.array(total_loss)
    np.save(LOSS_PATH, loss_arr)

    return



# def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
#     print('=> Saving checkpoint')
#     torch.save(state, filename)


# def load_checkpoint(checkpoint, model, optimizer):
#     print("=> Loading checkpoint")
#     model.load_state_dict(checkpoint['state_dict'])
#     optimizer.load_state_dict(checkpoint['optimizer'])


if __name__ == "__main__":
    training()
    
    print("-"*40)
    print("All Done")
    
