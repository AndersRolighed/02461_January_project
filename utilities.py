#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 13:28:39 2022

Utilities

"""

#%% Import

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import glob



#%% 
# Tager numpy ndarray eller torch tensor som input i test_image og test_mask
# Plotter den givne slice (n_slice)

def plot_2d_data(test_image,test_mask,n_slice):
    plt.figure(figsize=(12, 8))
    
    plt.subplot(241)
    plt.imshow(test_image[0,:,:,n_slice], cmap='gray')
    plt.title('Modality 1')
    plt.subplot(242)
    plt.imshow(test_image[1,:,:,n_slice], cmap='gray')
    plt.title('Modality 2')
    plt.subplot(243)
    plt.imshow(test_image[2,:,:,n_slice], cmap='gray')
    plt.title('Modality 3')
    plt.subplot(244)
    plt.imshow(test_image[3,:,:,n_slice], cmap='gray')
    plt.title('Modality 4')
    plt.subplot(245)
    plt.imshow(test_mask[0,:,:,n_slice], cmap='gray')
    plt.title('Mask 1')
    plt.subplot(246)
    plt.imshow(test_mask[1,:,:,n_slice], cmap='gray')
    plt.title('Mask 2')
    plt.subplot(247)
    plt.imshow(test_mask[2,:,:,n_slice], cmap='gray')
    plt.title('Mask 3')
    plt.subplot(248)
    plt.imshow(test_mask[3,:,:,n_slice], cmap='gray')
    plt.title('Mask 4')
    
    plt.show()
    

#%%
# Tager numpy ndarray eller torch tensor som input i test_image og test_mask
# Plotter den givne slice (n_slice)

def plot_2d_tensor(test_image, test_mask):

    max = 0
    for i in range(128):
        mask_sum = np.sum(test_mask[0,1,:,:,i])+np.sum(test_mask[0,2,:,:,i])+np.sum(test_mask[0,3,:,:,i])
        if mask_sum > max:
            max = mask_sum
            n_slice = i

    plt.figure(figsize=(12, 8))
    
    plt.subplot(241)
    plt.imshow(test_image[0,0,:,:,n_slice], cmap='gray')
    plt.title('Modality 1')
    plt.subplot(242)
    plt.imshow(test_image[0,1,:,:,n_slice], cmap='gray')
    plt.title('Modality 2')
    plt.subplot(243)
    plt.imshow(test_image[0,2,:,:,n_slice], cmap='gray')
    plt.title('Modality 3')
    plt.subplot(244)
    plt.imshow(test_image[0,3,:,:,n_slice], cmap='gray')
    plt.title('Modality 4')
    plt.subplot(245)
    plt.imshow(test_mask[0,0,:,:,n_slice], cmap='gray')
    plt.title('Mask 1')
    plt.subplot(246)
    plt.imshow(test_mask[0,1,:,:,n_slice], cmap='gray')
    plt.title('Mask 2')
    plt.subplot(247)
    plt.imshow(test_mask[0,2,:,:,n_slice], cmap='gray')
    plt.title('Mask 3')
    plt.subplot(248)
    plt.imshow(test_mask[0,3,:,:,n_slice], cmap='gray')
    plt.title('Mask 4')
    
    plt.show()


#%% Batch load single
# Will load a single image and mask but as a batch of 1. 
def batchLoad_single(directory, fileindex):
    
    image_list = sorted(glob.glob(str(directory) + 'images/*'))
    mask_list  = sorted(glob.glob(str(directory) + 'masks/*'))
    
    image = np.load(image_list[fileindex])
    mask  = np.load(mask_list[fileindex])
    
    image_batch = np.expand_dims(image, axis=0)
    mask_batch  = np.expand_dims(mask, axis=0)
    
    image_batch_tensor = torch.from_numpy(image_batch)
    mask_batch_tensor = torch.from_numpy(mask_batch)
    
    # print(image_batch, image_batch_tensor)
    
    return image_batch_tensor.float(), mask_batch_tensor.float()
    
# Example:
# directory = '/home/jesperdlau/Documents/Intro_Intelligente_Systemer/Januarprojekt/02461_January_project/BraTS2020/train_valid_data/train/'
# test_image, test_mask = batchLoad_single(directory, 0)


#%% 
def plot_prediction_mask_max(prediction, mask):
    mask_np = mask.detach().numpy()
    max = 0
    for i in range(128):
        mask_sum = np.sum(mask_np[0,1,:,:,i])+np.sum(mask_np[0,2,:,:,i])+np.sum(mask_np[0,3,:,:,i])
        if mask_sum > max:
            max = mask_sum
            idx = i

    pred_argmax = np.argmax(prediction.detach().numpy(), axis=1)
    mask_argmax = np.argmax(mask.detach().numpy(), axis=1)
    
    plt.figure()
    plt.subplot(121)
    plt.imshow(pred_argmax[0,:,:,idx])
    plt.title("Predicted mask")
    plt.subplot(122)
    plt.imshow(mask_argmax[0,:,:,idx])
    plt.title("Original mask")
    
    # plt.suptitle("")
    plt.show()


def save_checkpoint(state, filename):
    print('=> Saving checkpoint')
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    
    

#%% 
def plot_prediction_mask(prediction, mask, idx):
    pred_argmax = np.argmax(prediction.detach().numpy(), axis=1)
    mask_argmax = np.argmax(mask.detach().numpy(), axis=1)
    
    print(np.shape(pred_argmax), np.shape(mask_argmax))
    plt.figure()
    plt.subplot(121)
    plt.imshow(pred_argmax[0,:,:,idx])
    plt.title("Predicted mask")
    plt.subplot(122)
    plt.imshow(mask_argmax[0,:,:,idx])
    plt.title("Original mask")
    
    # plt.suptitle("")
    plt.show()

#%%
# Input: prediction as tensor, mask as tensor, image number as int, index as int, and extra info as string (fx. )
def plot_combined(prediction, mask, img_n, idx, info):
    prediction = prediction.detach().numpy()
    mask = mask.detach().numpy()
    pred_argmax = np.argmax(prediction, axis=1)
    mask_argmax = np.argmax(mask, axis=1)
    
    plt.figure(figsize=(10, 4))
    plt.suptitle(f"Image: {img_n}  -  Slice: {idx}  -  Info: {info}")
    
    ax1 = plt.subplot2grid((2,5), (0,0))
    ax2 = plt.subplot2grid((2,5), (0,1))
    ax3 = plt.subplot2grid((2,5), (0,2))
    ax4 = plt.subplot2grid((2,5), (0,3))
    ax5 = plt.subplot2grid((2,5), (0,4))
    ax6 = plt.subplot2grid((2,5), (1,0))
    ax7 = plt.subplot2grid((2,5), (1,1))
    ax8 = plt.subplot2grid((2,5), (1,2))
    ax9 = plt.subplot2grid((2,5), (1,3))
    ax10 = plt.subplot2grid((2,5), (1,4))
    
    ax1.imshow(prediction[0,0,:,:,idx], cmap='gray')
    ax1.set_title('Modality 1')
    ax2.imshow(prediction[0,1,:,:,idx], cmap='gray')
    ax2.set_title('Modality 2')
    ax3.imshow(prediction[0,2,:,:,idx], cmap='gray')
    ax3.set_title('Modality 3')
    ax4.imshow(prediction[0,3,:,:,idx], cmap='gray')
    ax4.set_title('Modality 4')
    
    ax6.imshow(mask[0,0,:,:,idx], cmap='gray')
    ax6.set_title('Mask 1')
    ax7.imshow(mask[0,1,:,:,idx], cmap='gray')
    ax7.set_title('Mask 2')
    ax8.imshow(mask[0,2,:,:,idx], cmap='gray')
    ax8.set_title('Mask 3')
    ax9.imshow(mask[0,3,:,:,idx], cmap='gray')
    ax9.set_title('Mask 4')
    
    ax5.imshow(pred_argmax[0,:,:,idx])
    ax5.set_title("Argmax pred")
    ax10.imshow(mask_argmax[0,:,:,idx])
    ax10.set_title("Argmax mask")
    
    for num in range(1,11):
        exec('ax'+str(num)+'.set_yticklabels([])')
        exec('ax'+str(num)+'.set_xticklabels([])')
        exec('ax'+str(num)+'.set_xticks([])')
        exec('ax'+str(num)+'.set_yticks([])')

    plt.show()
    

# Input: raw as tensor, prediction as tensor, mask as tensor, image number as int, index as int, and extra info as string (fx. )
def plot_combined_withRaw(raw, prediction, mask, img_n, idx, info):
    raw = raw.detach().numpy()
    prediction = prediction.detach().numpy()
    mask = mask.detach().numpy()
    pred_argmax = np.argmax(prediction, axis=1)
    mask_argmax = np.argmax(mask, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.suptitle(f"Image: {img_n}  -  Slice: {idx}  -  Info: {info}")
    
    ax1 = plt.subplot2grid((3,5), (0,0))
    ax2 = plt.subplot2grid((3,5), (0,1))
    ax3 = plt.subplot2grid((3,5), (0,2))
    ax4 = plt.subplot2grid((3,5), (0,3))
    ax5 = plt.subplot2grid((3,5), (0,4))
    
    ax6 = plt.subplot2grid((3,5), (1,0))
    ax7 = plt.subplot2grid((3,5), (1,1))
    ax8 = plt.subplot2grid((3,5), (1,2))
    ax9 = plt.subplot2grid((3,5), (1,3))
    ax10 = plt.subplot2grid((3,5), (1,4))
    
    ax11 = plt.subplot2grid((3,5), (2,0))
    ax12 = plt.subplot2grid((3,5), (2,1))
    ax13 = plt.subplot2grid((3,5), (2,2))
    ax14 = plt.subplot2grid((3,5), (2,3))
    # ax15 = plt.subplot2grid((3,5), (2,4))
    
    # Preds
    ax1.imshow(prediction[0,0,:,:,idx], cmap='gray')
    ax1.set_title('Pred 1')
    ax2.imshow(prediction[0,1,:,:,idx], cmap='gray')
    ax2.set_title('Pred 2')
    ax3.imshow(prediction[0,2,:,:,idx], cmap='gray')
    ax3.set_title('Pred 3')
    ax4.imshow(prediction[0,3,:,:,idx], cmap='gray')
    ax4.set_title('Pred 4')
    
    # Masks
    ax6.imshow(mask[0,0,:,:,idx], cmap='gray')
    ax6.set_title('Mask 1')
    ax7.imshow(mask[0,1,:,:,idx], cmap='gray')
    ax7.set_title('Mask 2')
    ax8.imshow(mask[0,2,:,:,idx], cmap='gray')
    ax8.set_title('Mask 3')
    ax9.imshow(mask[0,3,:,:,idx], cmap='gray')
    ax9.set_title('Mask 4')
    
    # Raw
    ax11.imshow(raw[0,0,:,:,idx], cmap='gray')
    ax11.set_title('Raw 1')
    ax12.imshow(raw[0,1,:,:,idx], cmap='gray')
    ax12.set_title('Raw 2')
    ax13.imshow(raw[0,2,:,:,idx], cmap='gray')
    ax13.set_title('Raw 3')
    ax14.imshow(raw[0,3,:,:,idx], cmap='gray')
    ax14.set_title('Raw 4')
    
    # Argmax
    ax5.imshow(pred_argmax[0,:,:,idx])
    ax5.set_title("Argmax pred")
    ax10.imshow(mask_argmax[0,:,:,idx])
    ax10.set_title("Argmax mask")
    
    
    for num in range(1,15):
        exec('ax'+str(num)+'.set_yticklabels([])')
        exec('ax'+str(num)+'.set_xticklabels([])')
        exec('ax'+str(num)+'.set_xticks([])')
        exec('ax'+str(num)+'.set_yticks([])')

    plt.show()
    
# Input: raw as tensor, bce as tensor, focal as tensor, mask as tensor, image number as int, index as int, learning rate as float, epoch as int
# Remaining accuracy data as float
def plot_bce_focal_raw(raw, bce, focal, mask, img_n, idx, LR, epoch, pix_bce, iou_bce, dice_bce, pix_focal, iou_focal, dice_focal):
    # Prepare data
    raw = raw.detach().numpy()
    bce = bce.detach().numpy()
    focal = focal.detach().numpy()
    mask = mask.detach().numpy()
    bce_argmax = np.argmax(bce, axis=1)
    focal_argmax = np.argmax(focal, axis=1)
    mask_argmax = np.argmax(mask, axis=1)
    
    # Initialize figure
    plt.figure(figsize=(10, 9))
    
    # Set info in top
    TITLE1 = f"Test_Image: {img_n}  -  Slice: {idx}  -  Learning Rate: {LR}  -  Epoch: {epoch}"
    TITLE2 = f"\nPix_bce:   {pix_bce}  -   IOU_bce:   {iou_bce}  -   Dice_bce:   {dice_bce} "
    TITLE3 = f"\nPix_focal: {pix_focal}  -   IOU_focal: {iou_focal}  -   Dice_focal: {dice_focal}"
    plt.suptitle(t=TITLE1+TITLE2+TITLE3,
                 x=0.13,
                 y=0.98,
                 ha="left")
    
    # Axies
    ax1 = plt.subplot2grid((4,5), (0,0))
    ax2 = plt.subplot2grid((4,5), (0,1))
    ax3 = plt.subplot2grid((4,5), (0,2))
    ax4 = plt.subplot2grid((4,5), (0,3))
    ax5 = plt.subplot2grid((4,5), (0,4))
    
    ax6 = plt.subplot2grid((4,5), (2,0))
    ax7 = plt.subplot2grid((4,5), (2,1))
    ax8 = plt.subplot2grid((4,5), (2,2))
    ax9 = plt.subplot2grid((4,5), (2,3))
    ax10 = plt.subplot2grid((4,5), (2,4))
    
    ax11 = plt.subplot2grid((4,5), (3,0))
    ax12 = plt.subplot2grid((4,5), (3,1))
    ax13 = plt.subplot2grid((4,5), (3,2))
    ax14 = plt.subplot2grid((4,5), (3,3))
    ax15 = plt.subplot2grid((4,5), (3,4))
    
    ax16 = plt.subplot2grid((4,5), (1,0))
    ax17 = plt.subplot2grid((4,5), (1,1))
    ax18 = plt.subplot2grid((4,5), (1,2))
    ax19 = plt.subplot2grid((4,5), (1,3))
    ax20 = plt.subplot2grid((4,5), (1,4))
    
    # BCE
    ax1.imshow(bce[0,0,:,:,idx], cmap='gray')
    ax1.set_title('BCE 1')
    ax2.imshow(bce[0,1,:,:,idx], cmap='gray')
    ax2.set_title('BCE 2')
    ax3.imshow(bce[0,2,:,:,idx], cmap='gray')
    ax3.set_title('BCE 3')
    ax4.imshow(bce[0,3,:,:,idx], cmap='gray')
    ax4.set_title('BCE 4')
    
    # FOCAL
    ax16.imshow(focal[0,0,:,:,idx], cmap='gray')
    ax16.set_title('Focal 1')
    ax17.imshow(focal[0,1,:,:,idx], cmap='gray')
    ax17.set_title('Focal 2')
    ax18.imshow(focal[0,2,:,:,idx], cmap='gray')
    ax18.set_title('Focal 3')
    ax19.imshow(focal[0,3,:,:,idx], cmap='gray')
    ax19.set_title('Focal 4')
    
    # Masks
    ax6.imshow(mask[0,0,:,:,idx], cmap='gray')
    ax6.set_title('Mask: No tumor')
    ax7.imshow(mask[0,1,:,:,idx], cmap='gray')
    ax7.set_title('Mask: Core')
    ax8.imshow(mask[0,2,:,:,idx], cmap='gray')
    ax8.set_title('Mask: PTE')
    ax9.imshow(mask[0,3,:,:,idx], cmap='gray')
    ax9.set_title('Mask: GD')
    
    # Raw
    ax11.imshow(raw[0,0,:,:,idx], cmap='gray')
    ax11.set_title('T1')
    ax12.imshow(raw[0,1,:,:,idx], cmap='gray')
    ax12.set_title('T1Gd')
    ax13.imshow(raw[0,2,:,:,idx], cmap='gray')
    ax13.set_title('T2')
    ax14.imshow(raw[0,3,:,:,idx], cmap='gray')
    ax14.set_title('T2-FLAIR')
    
    # Argmax
    ax5.imshow(bce_argmax[0,:,:,idx])
    ax5.set_title("Argmax BCE")
    ax10.imshow(mask_argmax[0,:,:,idx])
    ax10.set_title("Argmax mask")
    ax20.imshow(focal_argmax[0,:,:,idx])
    ax20.set_title("Argmax focal")
    
    # Remove ticks and labels for each subplot
    for num in range(1,21):
        exec('ax'+str(num)+'.set_yticklabels([])')
        exec('ax'+str(num)+'.set_xticklabels([])')
        exec('ax'+str(num)+'.set_xticks([])')
        exec('ax'+str(num)+'.set_yticks([])')

    plt.show()
    
    
    
    
    
    
    
    