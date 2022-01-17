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

def dice_score(outputs: torch.Tensor, labels: torch.Tensor):
    SMOOTH = 1e-6
    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most probably
    # be with the BATCH x 1 x H x W shape
    outputs = outputs.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W
    
    intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum((1, 2))         # Will be zero if both are 0
    
    dice = (2*intersection + SMOOTH) / (union + intersection + SMOOTH)  # We smooth our devision to avoid 0/0
    
    thresholded = torch.clamp(20 * (dice - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds
    
    return thresholded.mean()  # Or thresholded.mean() if you are interested in average across the batch

def iou_score_allmod(prediction, label):
    pred_mod1 = prediction[0,:,:,:]
    label_mod1 = label[0,:,:,:]
    
    pred_mod2 = prediction[1,:,:,:]
    label_mod2 = label[1,:,:,:]
    
    pred_mod3 = prediction[2,:,:,:]
    label_mod3 = label[2,:,:,:]
    
    pred_mod4 = prediction[3,:,:,:]
    label_mod4 = label[3,:,:,:]
    
    iou_mod1 = iou_score(pred_mod1,label_mod1)
    iou_mod2 = iou_score(pred_mod2,label_mod2)
    iou_mod3 = iou_score(pred_mod3,label_mod3)
    iou_mod4 = iou_score(pred_mod4,label_mod4)
    
    iou_mean = (iou_mod1 + iou_mod2 + iou_mod3 + iou_mod4)/4
    iou_allmod = iou_score(prediction, label)
    
    print("_"*48)
    print(f"Iou score of mod 1 = {iou_mod1:.3f}")
    print(f"Iou score of mod 2 = {iou_mod2:.3f}")
    print(f"Iou score of mod 3 = {iou_mod3:.3f}")
    print(f"Iou score of mod 4 = {iou_mod4:.3f}")
    print(f"Mean of iou score of the 4 modalities = {iou_mean:.3f}")
    print(f"Iou score of all mods = {iou_allmod:.3f}")
    
def dice_score_allmod(prediction, label):
    pred_mod1 = prediction[0,:,:,:]
    label_mod1 = label[0,:,:,:]
    
    pred_mod2 = prediction[1,:,:,:]
    label_mod2 = label[1,:,:,:]
    
    pred_mod3 = prediction[2,:,:,:]
    label_mod3 = label[2,:,:,:]
    
    pred_mod4 = prediction[3,:,:,:]
    label_mod4 = label[3,:,:,:]
    
    dice_mod1 = dice_score(pred_mod1,label_mod1)
    dice_mod2 = dice_score(pred_mod2,label_mod2)
    dice_mod3 = dice_score(pred_mod3,label_mod3)
    dice_mod4 = dice_score(pred_mod4,label_mod4)
    dice_mean = (dice_mod1 + dice_mod2 + dice_mod3 + dice_mod4)/4
    dice_allmod = dice_score(prediction, label)
    
    print("_"*48)
    print(f"Dice score of mod 1 = {dice_mod1:.3f}")
    print(f"Dice score of mod 2 = {dice_mod2:.3f}")
    print(f"Dice score of mod 3 = {dice_mod3:.3f}")
    print(f"Dice score of mod 4 = {dice_mod4:.3f}")
    print(f"Mean of dice score of the 4 modalities = {dice_mean:.3f}")
    print(f"Dice score of all mods = {dice_allmod:.3f}")
    print("_"*48)
    
def full_evaluation(prediction, label):
    iou_score_allmod(prediction, label)
    dice_score_allmod(prediction, label)

if __name__ =="__main__":
    
    from utilities import plot_prediction_mask, plot_2d_tensor, plot_2d_data
    
    num_1 = 0
    num_2 = 4
    mod_index = 1
    n_slice = 70
    
    mask_1 = torch.from_numpy(np.load(f"./BraTS2020/train_valid_data/train/masks/mask_{num_1}.npy"))
    mask_2 = torch.from_numpy(np.load(f"./BraTS2020/train_valid_data/train/masks/mask_{num_2}.npy"))
    
    full_evaluation(mask_1, mask_2)
    
    # iou_score_allmod(mask_1, mask_2)
    # dice_score_allmod(mask_1, mask_2)
    
    #mask_1_mod = mask_1[mod_index,:,:,:]
    #mask_2_mod = mask_2[mod_index,:,:,:]   
    
    #print(iou_score(mask_1_mod,mask_2_mod))
    #plot_2d_data(mask_1, mask_2,n_slice)