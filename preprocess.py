#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""
#%% Import

import numpy as np
import glob
import nibabel as nib
from sklearn.preprocessing import MinMaxScaler

#%% Raw data
# Set location to your own location for the RAW data, if available. 
t1_list    = sorted(glob.glob('/home/jesperdlau/BraTS_Training_Data/MICCAI_BraTS2020_TrainingData/*/*t1.nii'))
t1ce_list  = sorted(glob.glob('/home/jesperdlau/BraTS_Training_Data/MICCAI_BraTS2020_TrainingData/*/*t1ce.nii'))
t2_list    = sorted(glob.glob('/home/jesperdlau/BraTS_Training_Data/MICCAI_BraTS2020_TrainingData/*/*t2.nii'))
flair_list = sorted(glob.glob('/home/jesperdlau/BraTS_Training_Data/MICCAI_BraTS2020_TrainingData/*/*flair.nii'))
mask_list   = sorted(glob.glob('/home/jesperdlau/BraTS_Training_Data/MICCAI_BraTS2020_TrainingData/*/*seg.nii'))

#%%
# Set pwd (place of working directory) to your own specific location
pwd = '/home/jesperdlau/Documents/Intro_Intelligente_Systemer/Januarprojekt/02461_January_project/'
scaler = MinMaxScaler()

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='uint8')[y]


# for indx in range(len(t1_list)):   #Using t1_list as all lists are of same size
for indx in range(20):
    print("Preprocessing image and masks number: ", indx)
      
    # Load image data from each modality and normalize
    temp_image_t1    = nib.load(t1_list[indx]).get_fdata()
    temp_image_t1    = scaler.fit_transform(temp_image_t1.reshape(-1, temp_image_t1.shape[-1])).reshape(temp_image_t1.shape)
    temp_image_t1ce  = nib.load(t1ce_list[indx]).get_fdata()
    temp_image_t1ce  = scaler.fit_transform(temp_image_t1ce.reshape(-1, temp_image_t1ce.shape[-1])).reshape(temp_image_t1ce.shape)
    temp_image_t2    = nib.load(t2_list[indx]).get_fdata()
    temp_image_t2    = scaler.fit_transform(temp_image_t2.reshape(-1, temp_image_t2.shape[-1])).reshape(temp_image_t2.shape)
    temp_image_flair = nib.load(flair_list[indx]).get_fdata()
    temp_image_flair = scaler.fit_transform(temp_image_flair.reshape(-1, temp_image_flair.shape[-1])).reshape(temp_image_flair.shape)
        
    # Stack each modality to one array and crop to shape 128*128*128
    temp_combined_images = np.stack([temp_image_t1, temp_image_t1ce, temp_image_t2, temp_image_flair], axis=0)
    temp_combined_images = temp_combined_images[:, 56:184, 56:184, 13:141] 

    # Load mask data as integers (0, 1, 2, 4) and reassign mask value 4 to 3 to make it simpler. Then, crop to shape 128*128*128. 
    temp_mask         = nib.load(mask_list[indx]).get_fdata()
    temp_mask         = temp_mask.astype(np.uint8)
    temp_mask[temp_mask==4] = 3
    temp_mask = temp_mask[56:184, 56:184, 13:141]
     
    # Embed the 4 values to categorical 4-dimensinal, binary values. Finally stack the values to similar shape as with images. 
    temp_mask = to_categorical(temp_mask, num_classes=4)
    temp_mask = np.stack([temp_mask[:,:,:,0], temp_mask[:,:,:,1], temp_mask[:,:,:,2], temp_mask[:,:,:,3]], axis=0)
    
    # Save image and mask
    np.save(pwd + 'BraTS2020/input_data_4channels/images/image_' + str(indx) + '.npy', temp_combined_images)
    np.save(pwd + 'BraTS2020/input_data_4channels/masks/mask_' + str(indx) + '.npy', temp_mask)

#%% Plot 2*4 
import matplotlib.pyplot as plt

n_slice = 75
n_image = 0

# Data location
test_image = np.load(pwd + 'BraTS2020/input_data_4channels/images/image_' + str(n_image) + '.npy')
test_mask   = np.load(pwd + 'BraTS2020/input_data_4channels/masks/mask_' + str(n_image) + '.npy')

# PyPlot figure
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

#%% Split folder into training and validation
import splitfolders

input_folder  = pwd + 'BraTS2020/input_data_4channels/'
output_folder = pwd + 'BraTS2020/train_valid_data/'

splitfolders.ratio(input_folder, output=output_folder, ratio=(.75, .25), group_prefix=None) # default values
