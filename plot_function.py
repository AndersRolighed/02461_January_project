# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 13:13:47 2022

@author: Rasmus
"""
import matplotlib.pyplot as plt

def plot_data():
    
    n_slice = 75
    n_image = 0
    
    # Data location
    test_image = np.load(pwd + 'BraTS2020/input_data_4channels/images/image_' + str(n_image) + '.npy')
    test_seg   = np.load(pwd + 'BraTS2020/input_data_4channels/masks/mask_' + str(n_image) + '.npy')
    
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
    plt.imshow(test_seg[0,:,:,n_slice], cmap='gray')
    plt.title('Mask 1')
    plt.subplot(246)
    plt.imshow(test_seg[1,:,:,n_slice], cmap='gray')
    plt.title('Mask 2')
    plt.subplot(247)
    plt.imshow(test_seg[2,:,:,n_slice], cmap='gray')
    plt.title('Mask 3')
    plt.subplot(248)
    plt.imshow(test_seg[3,:,:,n_slice], cmap='gray')
    plt.title('Mask 4')
    
    plt.show()