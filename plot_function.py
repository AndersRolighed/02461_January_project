# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 13:13:47 2022

@author: Rasmus
"""
import matplotlib.pyplot as plt
import numpy as np
import torch 
#funktionen tager numpy 

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

test_image = torch.from_numpy(np.load("C:/Users/Rasmus/Desktop/MRI-projekt/02461_January_project/BraTS2020/train_valid_data/train/images/image_0.npy"))

test_mask  = torch.from_numpy(np.load("C:/Users/Rasmus/Desktop/MRI-projekt/02461_January_project/BraTS2020/train_valid_data/train/masks/mask_0.npy"))

print(type(test_image))
print(type(test_mask))
plot_2d_data(test_image,test_mask,75)


