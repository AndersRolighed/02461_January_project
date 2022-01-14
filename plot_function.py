# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 13:13:47 2022

@author: Rasmus
"""
import matplotlib.pyplot as plt
import numpy as np
import torch 
#funktionen tager numpy 
#from count_nonzeros import count_nonzeros_in_tensor

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

image_num = 4
test_image = np.load(f"./BraTS2020/train_valid_data/train/images/image_{image_num}.npy")

test_mask = torch.from_numpy(np.load(f"./BraTS2020/train_valid_data/train/masks/mask_{image_num}.npy"))

m_1_test_mask = test_mask[0,:,:,:]
m_2_test_mask = test_mask[1,:,:,:]
m_3_test_mask = test_mask[2,:,:,:]
m_4_test_mask = test_mask[3,:,:,:]

#np_test_mask = np.load(f"./BraTS2020/train_valid_data/train/masks/mask_{image_num}.npy")

#flat_tensor = torch.flatten(test_mask)

#flat_array = np.flatten(np_test_mask)
# print(m_1_test_mask.sum())
# print(m_2_test_mask.sum())
# print(m_3_test_mask.sum())
# print(m_4_test_mask.sum())


#print(torch.sum(flat_tensor))
#print(torch.cumsum(test_mask,dim=2))
#print(test_mask)
#print(torch.sum(test_mask, keepdim = True))
plot_2d_data(test_image,test_mask,70)



