#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 25 18:03:02 2021

@author: jesperdlau
"""
#%% Import

import numpy as np
import glob
import nibabel as nib
from sklearn.preprocessing import MinMaxScaler


#%% Function

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='uint8')[y]

#%%
pwd = '/home/jesperdlau/Documents/Intro_Intelligente_Systemer/Januarprojekt/'


t1_list    = sorted(glob.glob('/home/jesperdlau/BraTS_Training_Data/MICCAI_BraTS2020_TrainingData/*/*t1.nii'))
t1ce_list  = sorted(glob.glob('/home/jesperdlau/BraTS_Training_Data/MICCAI_BraTS2020_TrainingData/*/*t1ce.nii'))
t2_list    = sorted(glob.glob('/home/jesperdlau/BraTS_Training_Data/MICCAI_BraTS2020_TrainingData/*/*t2.nii'))
flair_list = sorted(glob.glob('/home/jesperdlau/BraTS_Training_Data/MICCAI_BraTS2020_TrainingData/*/*flair.nii'))
seg_list   = sorted(glob.glob('/home/jesperdlau/BraTS_Training_Data/MICCAI_BraTS2020_TrainingData/*/*seg.nii'))

# for n in range(10):
#     print(t2_list[n])


#%%

scaler = MinMaxScaler()

# for indx in range(len(t1_list)):   #Using t1_list as all lists are of same size
for indx in range(3):   #Using t1_list as all lists are of same size
    print("Now preparing image and masks number: ", indx)
      
    temp_image_t1    = nib.load(t1_list[indx]).get_fdata()
    temp_image_t1    = scaler.fit_transform(temp_image_t1.reshape(-1, temp_image_t1.shape[-1])).reshape(temp_image_t1.shape)
  
    temp_image_t1ce  = nib.load(t1ce_list[indx]).get_fdata()
    temp_image_t1ce  = scaler.fit_transform(temp_image_t1ce.reshape(-1, temp_image_t1ce.shape[-1])).reshape(temp_image_t1ce.shape)
    
    temp_image_t2    = nib.load(t2_list[indx]).get_fdata()
    temp_image_t2    = scaler.fit_transform(temp_image_t2.reshape(-1, temp_image_t2.shape[-1])).reshape(temp_image_t2.shape)
   
    temp_image_flair = nib.load(flair_list[indx]).get_fdata()
    temp_image_flair = scaler.fit_transform(temp_image_flair.reshape(-1, temp_image_flair.shape[-1])).reshape(temp_image_flair.shape)
        
    temp_seg         = nib.load(seg_list[indx]).get_fdata()
    temp_seg         = temp_seg.astype(np.uint8)
    temp_seg[temp_seg==4] = 3  #Reassign mask values 4 to 3
    #print(np.unique(temp_seg))
    
    # print("seg_shape =  " , np.shape(temp_seg))
    
    # temp_combined_images = np.stack([temp_image_flair, temp_image_t1, temp_image_t1ce, temp_image_t2], axis=3)
    # temp_combined_images = np.stack([temp_image_flair, temp_image_t1ce, temp_image_t2, temp_image_t1], axis=3)
    # temp_combined_images = np.stack([temp_image_t1, temp_image_t1ce, temp_image_t2, temp_image_flair], axis=3)
    temp_combined_images = np.stack([temp_image_t1, temp_image_t1ce, temp_image_t2, temp_image_flair], axis=0)


    #Crop to a size to be divisible by 64 so we can later extract 64x64x64 patches. 
    #cropping x, y, and z
    # temp_combined_images = temp_combined_images[56:184, 56:184, 13:141]
    temp_combined_images = temp_combined_images[:, 56:184, 56:184, 13:141]

    temp_seg = temp_seg[56:184, 56:184, 13:141]
    
    temp_seg = to_categorical(temp_seg, num_classes=4)
    # np.save('test.npy', temp_combined_images)
    # np.save('/home/jesperdlau/BraTS_Training_Data/BraTS2020_TrainingData/input_data_4channels/images/image_' + str(indx) + '.npy', temp_combined_images)
    # np.save('/home/jesperdlau/BraTS_Training_Data/BraTS2020_TrainingData/input_data_4channels/segmentations/seg_' + str(indx) + '.npy', temp_seg)
    np.save(pwd + 'BraTS2020/input_data_4channels_1/images/image_' + str(indx) + '.npy', temp_combined_images)
    np.save(pwd + 'BraTS2020/input_data_4channels_1/segmentations/seg_' + str(indx) + '.npy', temp_seg)
    
    
#%%

test_image_1 = np.load(pwd + 'BraTS2020/input_data_4channels_1/images/image_3.npy')
test_image = np.load(pwd + 'BraTS2020/input_data_4channels/images/image_3.npy')

print(np.shape(test_image))
print(np.shape(test_image_1))
#%% Test

import matplotlib.pyplot as plt

# test_image = np.load(pwd + 'BraTS2020/input_data_4channels/images/image_3.npy')

# plt.imshow(test_image[:,:,64,1], cmap='gray')


test_image = np.load(pwd + 'BraTS2020/input_data_4channels_1/images/image_3.npy')

plt.imshow(test_image[1,:,:,64], cmap='gray')

#%% Test segmentation repadding

test_seg = np.load(pwd + 'BraTS2020/input_data_4channels/segmentations/seg_8.npy')
print(np.shape(test_seg))

# Original seg shape = (240, 240, 155)
# Crop = [56:184, 56:184, 13:141]
#constant_values=0
test_seg_pad = np.pad(test_seg, ((56,56),(56,56),(13,14),(0,0)), 'constant')
print(np.shape(test_seg_pad))

n_slice = 75

plt.figure()
plt.subplot(121)
plt.imshow(test_seg[:,:,n_slice,2], cmap='gray')
plt.subplot(122)
plt.imshow(test_seg_pad[:,:,(n_slice + 13),2], cmap='gray')
plt.show()




#%% Test segmentation repadding 2 - ikke det samme med seg/mask...

test_seg = np.load(pwd + 'BraTS2020/input_data_4channels_1/segmentations/seg_2.npy')
print(np.shape(test_seg))

# Original seg shape = (240, 240, 155)
# Crop = [56:184, 56:184, 13:141]
# constant_values=0
test_seg_pad = np.pad(test_seg, ((0,0),(56,56),(56,56),(13,14)), 'constant')
print(np.shape(test_seg_pad))

n_slice = 75

plt.figure()
plt.subplot(121)
plt.imshow(test_seg[2,:,:,n_slice], cmap='gray')
plt.subplot(122)
plt.imshow(test_seg_pad[2,:,:,(n_slice + 13)], cmap='gray')
plt.show()


#%% Plot 2*6
n_slice = 75
n_image = 3

test_image = np.load(pwd + 'BraTS2020/input_data_4channels/images/image_' + str(n_image) + '.npy')
test_seg = np.load(pwd + 'BraTS2020/input_data_4channels/segmentations/seg_' + str(n_image) + '.npy')


plt.figure(figsize=(12, 8))

plt.subplot(241)
plt.imshow(test_image[:,:,n_slice,0], cmap='gray')
plt.title('Image ')
plt.subplot(242)
plt.imshow(test_image[:,:,n_slice,1], cmap='gray')
plt.title('Image ')
plt.subplot(243)
plt.imshow(test_image[:,:,n_slice,2], cmap='gray')
plt.title('Image ')
plt.subplot(244)
plt.imshow(test_image[:,:,n_slice,3], cmap='gray')
plt.title('Image ')
plt.subplot(245)
plt.imshow(test_seg[:,:,n_slice,0])
plt.title('Segmentation')
plt.subplot(246)
plt.imshow(test_seg[:,:,n_slice,1])
plt.title('Segmentation')
plt.subplot(247)
plt.imshow(test_seg[:,:,n_slice,2])
plt.title('Segmentation')
plt.subplot(248)
plt.imshow(test_seg[:,:,n_slice,3])
plt.title('Segmentation')

plt.show()

#%% Plot 2*6 - 2
n_slice = 75
n_image = 3

test_image = np.load(pwd + 'BraTS2020/input_data_4channels_1/images/image_' + str(n_image) + '.npy')
test_seg = np.load(pwd + 'BraTS2020/input_data_4channels_1/segmentations/seg_' + str(n_image) + '.npy')


plt.figure(figsize=(12, 8))

plt.subplot(241)
plt.imshow(test_image[:,:,n_slice,0], cmap='gray')
plt.title('Image ')
plt.subplot(242)
plt.imshow(test_image[:,:,n_slice,1], cmap='gray')
plt.title('Image ')
plt.subplot(243)
plt.imshow(test_image[:,:,n_slice,2], cmap='gray')
plt.title('Image ')
plt.subplot(244)
plt.imshow(test_image[:,:,n_slice,3], cmap='gray')
plt.title('Image ')
plt.subplot(245)
plt.imshow(test_seg[:,:,n_slice,0])
plt.title('Segmentation')
plt.subplot(246)
plt.imshow(test_seg[:,:,n_slice,1])
plt.title('Segmentation')
plt.subplot(247)
plt.imshow(test_seg[:,:,n_slice,2])
plt.title('Segmentation')
plt.subplot(248)
plt.imshow(test_seg[:,:,n_slice,3])
plt.title('Segmentation')

plt.show()

#%% Split folder into validation
import splitfolders  # or import split_folders

input_folder  = pwd + 'BraTS2020/input_data_4channels/'
output_folder = pwd + 'BraTS2020/data_128_4channels/'

# Split with a ratio.
# To only split into training and validation set, set a tuple to `ratio`, i.e, `(.8, .2)`.
splitfolders.ratio(input_folder, output=output_folder, ratio=(.75, .25), group_prefix=None) # default values