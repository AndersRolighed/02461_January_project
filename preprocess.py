#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preprocess


Partial credit to youtube user 'DigitalSreeni'
https://www.youtube.com/watch?v=oB35sV1npVI
https://github.com/bnsreenu/python_for_microscopists/blob/master/231_234_BraTa2020_Unet_segmentation/232_brats2020_get_data_ready.py
"""
#%% Import

import numpy as np
import glob
import nibabel as nib
from sklearn.preprocessing import MinMaxScaler

#%% Raw data
# Set location to the RAW data.
t1_list    = sorted(glob.glob('/home/jesperdlau/BraTS_Training_Data/MICCAI_BraTS2020_TrainingData/*/*t1.nii'))
t1ce_list  = sorted(glob.glob('/home/jesperdlau/BraTS_Training_Data/MICCAI_BraTS2020_TrainingData/*/*t1ce.nii'))
t2_list    = sorted(glob.glob('/home/jesperdlau/BraTS_Training_Data/MICCAI_BraTS2020_TrainingData/*/*t2.nii'))
flair_list = sorted(glob.glob('/home/jesperdlau/BraTS_Training_Data/MICCAI_BraTS2020_TrainingData/*/*flair.nii'))
mask_list   = sorted(glob.glob('/home/jesperdlau/BraTS_Training_Data/MICCAI_BraTS2020_TrainingData/*/*seg.nii'))

#%%
# Set pwd (place of working directory) to your own specific location
# pwd = '/home/jesperdlau/Documents/Intro_Intelligente_Systemer/Januarprojekt/02461_January_project/'
pwd = '/home/jesperdlau/BraTS_Training_Data/'


# Function to 1-hot encode
def to_categorical(y, num_classes):
    return np.eye(num_classes, dtype='uint8')[y]


for indx in range(len(t1_list)):   #Using t1_list as all lists are of same size
# for indx in range(20, 50):
    print("Preprocessing image and masks number:", indx)
    
    # Load .nii data for each modality and min-max normalize
    scaler = MinMaxScaler()
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
    
    # 1-hot encode the 4 values of each mask. Finally stack the values to similar shape as with images. 
    temp_mask = to_categorical(temp_mask, num_classes=4)
    temp_mask = np.stack([temp_mask[:,:,:,0], temp_mask[:,:,:,1], temp_mask[:,:,:,2], temp_mask[:,:,:,3]], axis=0)
    
    # Save image and mask
    # np.save(pwd + 'BraTS2020/input_data_4channels/images/image_' + str(indx) + '.npy', temp_combined_images)
    # np.save(pwd + 'BraTS2020/input_data_4channels/masks/mask_' + str(indx) + '.npy', temp_mask)
    np.save(pwd + 'input_data_4channels/images/image_' + str(indx) + '.npy', temp_combined_images)
    np.save(pwd + 'input_data_4channels/masks/mask_' + str(indx) + '.npy', temp_mask)


#%% Split folder into training and validation
import splitfolders

input_folder  = pwd + 'input_data_4channels/'
output_folder = pwd + 'train_data_sample/'

splitfolders.ratio(input_folder, output=output_folder, ratio=(.75, .25), group_prefix=None) # default values
