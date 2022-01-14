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
def plot_prediction_mask(prediction, mask):
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