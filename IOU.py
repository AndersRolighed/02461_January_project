# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 09:30:24 2022

@author: Rasmus
"""

# fra - https://www.kaggle.com/iezepov/fast-iou-scoring-metric-in-pytorch-and-numpy

import torch
import numpy as np

def iou_score(outputs: torch.Tensor, labels: torch.Tensor):
    SMOOTH = 1e-6
    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most probably
    # be with the BATCH x 1 x H x W shape
    outputs = outputs.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W
    
    intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum((1, 2))         # Will be zero if both are 0
    
    iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0
    
    thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds
    
    return thresholded.mean()  # Or thresholded.mean() if you are interested in average across the batch

if __name__ =="__main__":
    
    from utilities import plot_prediction_mask, plot_2d_tensor, plot_2d_data
    
    num_1 = 0
    num_2 = 2
    mod_index = 1
    n_slice = 70
    
    mask_1 = torch.from_numpy(np.load(f"./BraTS2020/train_valid_data/train/masks/mask_{num_1}.npy"))
    mask_2 = torch.from_numpy(np.load(f"./BraTS2020/train_valid_data/train/masks/mask_{num_2}.npy"))
    
    mask_1_mod = mask_1[mod_index,:,:,:]
    mask_2_mod = mask_2[mod_index,:,:,:]   
    
    print(iou_score(mask_1_mod,mask_2_mod))
    plot_2d_data(mask_1, mask_2,n_slice)