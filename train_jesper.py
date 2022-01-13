#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


"""

from data_load import BraTS_dataset
from test_model import UNet

import os, os.path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader


import numpy as np
import matplotlib.pyplot as plt

#%%

class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        # inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss
        
        return Dice_BCE
    
class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        # inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice


# def train_one_epoch(epoch_index, tb_writer):
#     running_loss = 0.
#     last_loss = 0.

#     # Here, we use enumerate(training_loader) instead of
#     # iter(training_loader) so that we can track the batch
#     # index and do some intra-epoch reporting
#     for i, data in enumerate(training_loader):
#         # Every data instance is an input + label pair
#         inputs, labels = data

#         # Zero your gradients for every batch!
#         optimizer.zero_grad()

#         # Make predictions for this batch
#         outputs = model(inputs)

#         # Compute the loss and its gradients
#         loss = loss_fn(outputs, labels)
#         loss.backward()

#         # Adjust learning weights
#         optimizer.step()

#         # Gather data and report
#         running_loss += loss.item()
#         if i % 1000 == 999:
#             last_loss = running_loss / 1000 # loss per batch
#             print('  batch {} loss: {}'.format(i + 1, last_loss))
#             tb_x = epoch_index * len(training_loader) + i + 1
#             tb_writer.add_scalar('Loss/train', last_loss, tb_x)
#             running_loss = 0.

#     return last_loss

def train_epochs(NETWORK, NUM_EPOCHS, BATCH_SIZE, LEARN_RATE):
    network = NETWORK
    network = network.float()
    
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # print(f"Using {device} device")

    # Set model to test - ie. train=false
    # model.train(mode=False)
    
    # test_data = BraTS_dataset(image_dir = "./BraTS2020/train_valid_data/train/images", 
    #                           mask_dir = "./BraTS2020/train_valid_data/train/masks")
    test_data = BraTS_dataset(image_dir = "/home/jesperdlau/BraTS_Training_Data/train_data/train/images",
                              mask_dir = "/home/jesperdlau/BraTS_Training_Data/train_data/train/masks")
    
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)
    optimizer = optim.Adam(network.parameters(), lr=LEARN_RATE)
    loss_func = torch.nn.CrossEntropyLoss()
    # loss_func = DiceBCELoss()
    loss_func = DiceLoss()
    
    all_loss = []
    
    for epoch in range(NUM_EPOCHS):
    
        epoch_loss = []
        # total_loss = 0
        # total_correct = 0
    
        for idx, (images, labels) in enumerate(test_loader): # Get Batch
            
            # Send to device
            # images, labels = images.to(device), labels.to(device)
        
            preds = network(images.float()) # Pass Batch
            # loss = F.cross_entropy(preds, labels) # Calculate Loss
            loss = loss_func(preds, labels.float())
    
            optimizer.zero_grad()
            loss.backward() # Calculate Gradients
            optimizer.step() # Update Weights
            
            run_loss = 0.
            run_loss = loss.item()
            epoch_loss.append(run_loss)
            
            # total_loss += run_loss
            # total_correct += get_num_correct(preds, labels)
            # all_loss += [loss.detach().numpy()]
            
            # Diagnostics
            print(f"{idx}: {run_loss}")
            
            out_np = preds.detach().numpy()
            mask_np = labels.detach().numpy()
            mask_argmax = np.argmax(mask_np, axis=1)
            pred_argmax = np.argmax(out_np, axis=1)
            
            # import matplotlib.pyplot as plt
            plt.figure()
            plt.subplot(121)
            plt.imshow(pred_argmax[0,:,:,75])
            plt.title("Predicted mask")
            plt.subplot(122)
            plt.imshow(mask_argmax[0,:,:,75])
            plt.title("Original mask")
            
            plt.suptitle(f"Index: {idx}, Loss: {run_loss:.4f}")
            plt.show()
            
        # print(
        #     "epoch", epoch, 
        #     # "total_correct:", total_correct, 
        #     "loss:", total_loss
        # )
        all_loss.append(epoch_loss)
        
        
    
    # flat_list = [item for sublist in all_loss for item in sublist] 
    
    return network, all_loss
    


if __name__ == '__main__':
    # test_run()
    NETWORK = UNet()
    NUM_EPOCHS = 1
    BATCH_SIZE = 1
    LEARN_RATE = 0.001
    
    
    
    network, all_loss = train_epochs(NETWORK, NUM_EPOCHS, BATCH_SIZE, LEARN_RATE)
    loss_arr = np.array(all_loss).flatten()
    
    
    # Save model
    torch.save(network.state_dict(), "model_test.pth")
    print("Saved PyTorch Model State to model.pth")

    # Load model
    # model = NeuralNetwork()
    # model.load_state_dict(torch.load("model_test.pth"))

    # Plot
    plt.plot([val for val in loss_arr if val < 10.])
    plt.show()