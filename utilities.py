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


#%% Batch load single
# Will load a single image and mask but as a batch of 1. 
def batchLoad_single(directory, fileindex):
    
    image_list = sorted(glob.glob(str(directory) + '/images/*'))
    mask_list  = sorted(glob.glob(str(directory) + '/masks/*'))
    
    image = np.load(image_list[fileindex])
    mask  = np.load(mask_list[fileindex])
    
    image_batch = np.expand_dims(image, axis=0)
    mask_batch  = np.expand_dims(mask, axis=0)
    
    return image_batch, mask_batch
    
# Example:
# directory = '/home/jesperdlau/Documents/Intro_Intelligente_Systemer/Januarprojekt/02461_January_project/BraTS2020/train_valid_data/train'
# test_image, test_mask = batchLoad_zero(directory, 0)