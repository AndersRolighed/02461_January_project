# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 09:30:24 2022

@author: Rasmus
"""

# fra - https://www.kaggle.com/iezepov/fast-iou-scoring-metric-in-pytorch-and-numpy

import torch
import numpy as np
import torch.nn as nn

def pixel_accuracy_func(prediction, label, print_stats):
    
    prediction = prediction.flatten()
    label = label.flatten()
    
    correct_pixels = 0
    
    for p, l in zip(prediction, label):
        if p==l:
            correct_pixels+=1

    total_num_pixels = len(prediction)
    pixel_accuracy = correct_pixels/total_num_pixels
    
    if print_stats:
        print(f"Pixel accuracy = {pixel_accuracy*100:.3f}%")
    
    return pixel_accuracy            

def iou_score(outputs: torch.Tensor, labels: torch.Tensor):
    SMOOTH = 1e-6
    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most probably
    # be with the BATCH x 1 x H x W shape
    #outputs = outputs.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W
    
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
    #outputs = outputs.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W
    
    intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum((1, 2))         # Will be zero if both are 0
    
    dice = (2*intersection + SMOOTH) / (union + intersection + SMOOTH)  # We smooth our devision to avoid 0/0
    
    thresholded = torch.clamp(20 * (dice - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds
    
    return thresholded.mean()  # Or thresholded.mean() if you are interested in average across the batch

def iou_score_allmod(prediction, label, print_stats):
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
    iou_allmod = float(iou_score(prediction, label))
    
    if print_stats:
        print("_"*48)
        print(f"Iou score of mod 1 = {iou_mod1:.7f}")
        print(f"Iou score of mod 2 = {iou_mod2:.7f}")
        print(f"Iou score of mod 3 = {iou_mod3:.7f}")
        print(f"Iou score of mod 4 = {iou_mod4:.7f}")
        print(f"Mean of iou score of the 4 modalities = {iou_mean:.7f}")
        print(f"Iou score of all mods = {iou_allmod:.7f}")
    
    return float(iou_allmod.detach().numpy())
    
def dice_score_allmod(prediction, label, print_stats):
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
    
    if print_stats:
        print("_"*48)
        print(f"Dice score of mod 1 = {dice_mod1:.7f}")
        print(f"Dice score of mod 2 = {dice_mod2:.7f}")
        print(f"Dice score of mod 3 = {dice_mod3:.7f}")
        print(f"Dice score of mod 4 = {dice_mod4:.7f}")
        print(f"Mean of dice score of the 4 modalities = {dice_mean:.7f}")
        print(f"Dice score of all mods = {dice_allmod:.7f}")
        print("_"*48)
    
    return dice_allmod.detach().numpy()

def full_evaluation(prediction, label, print_stats):
    
    # print("HER!!!!!!!!")
    # print(np.shape(prediction))
    # print(np.shape(label))
    
    prediction = prediction[0,:,:,:,:]
    label = label[0,:,:,:,:]
    
    # print("HER!!!!!!!!")
    # print(np.shape(prediction))
    # print(np.shape(label))
    # print(label)
    
    prediction = np.argmax(prediction,axis=0)
    label = np.argmax(label, axis=0)
    
    # print("HER!!!!!!!!")
    # print(np.shape(prediction))
    # print(np.shape(label))
    # print(label)
    
    prediction = np.eye(4, dtype='uint8')[prediction]
    label = np.eye(4, dtype='uint8')[label]
    
    # print("HER!!!!!!!!")
    # print(np.shape(prediction))
    # print(np.shape(label))
    # print(label)
    
    # # prediction = torch.nn.functional.one_hot(prediction.to(torch.int64))
    # # label = torch.nn.functional.one_hot(label.to(torch.int64))
    
    # # print(prediction)
    # # print(label)
    # print(prediction.size())
    
    pixel_accuracy = pixel_accuracy_func(prediction, label, print_stats)
    
    prediction = torch.tensor(prediction, dtype=torch.int8)
    label = torch.tensor(label, dtype=torch.int8)
    
    iou_allmod = iou_score_allmod(prediction, label, print_stats)
    dice_allmod = dice_score_allmod(prediction, label, print_stats)
    
    return pixel_accuracy, iou_allmod, dice_allmod

if __name__ =="__main__":
    
    from utilities import plot_prediction_mask
    
    num_1 = 0
    num_2 = 1
    mod_index = 1
    n_slice = 70
    directory = "./BraTS2020/train_valid_data/train/"
    fileindex_1 = 1
    fileindex_2 = 2
    
    # prediction = torch.from_numpy(np.load(f"{directory}/masks/mask_{num_1}.npy"))
    # mask = torch.from_numpy(np.load(f"{directory}/masks/mask_{num_2}.npy"))
    # prediction = np.expand_dims(prediction, axis=0)
    # mask = np.expand_dims(mask, axis=0)
    # prediction = torch.from_numpy(prediction)
    # mask = torch.from_numpy(mask)
    # plot_prediction_mask(prediction, mask)
    
    # prediction = prediction[0,:,:,:,:]
    # mask = mask[0,:,:,:,:]

    
    prediction = np.load("./test_prediction.npy")
    mask = np.load("./test_prediction_mask.npy")
    
    pixel_score, iou_score, dice_score = full_evaluation(prediction,mask,False)
    
    # plot_prediction_mask(prediction, mask)
    # dice_score_allmod(prediction, mask, True)
    
    # mask_1_mod = mask_1[mod_index,:,:,:]
    # mask_2_mod = mask_2[mod_index,:,:,:]   
    
    # print(iou_score(mask_1_mod,mask_2_mod))
    