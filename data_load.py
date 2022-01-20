# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 09:36:24 2022

@author: Rasmus
"""
import os, os.path
# from torchvision.io import read_image
import torch
from torch.utils.data import Dataset
# from torchvision import datasets
from torchvision.transforms import ToTensor
import numpy as np
from torch.utils.data import DataLoader
# import matplotlib as plt

#%%
class BraTS_dataset(Dataset):
    
    def __init__(self, image_dir, mask_dir):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = ToTensor()
        self.image_list = os.listdir(image_dir)
    
    def __len__(self):
        return len([name for name in os.listdir(self.image_dir)])
    
    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_list[idx])
        mask_path = os.path.join(self.mask_dir, self.image_list[idx].replace("image", "mask"))
        
        image = np.load(image_path)
        mask = np.load(mask_path)
        
        image = torch.from_numpy(image)
        mask = torch.from_numpy(mask)
        
        return image, mask
    
    
    
#%%
if __name__ == "__main__":
    # from utilities import plot_2d_tensor
    # test_data = BraTS_dataset(image_dir = ".\BraTS2020\train_valid_data\train\images", 
    #                           mask_dir = ".\BraTS2020\train_valid_data\train\masks")
    # test_data = BraTS_dataset(image_dir = "./BraTS2020/train_valid_data/train/images", 
                              # mask_dir = "./BraTS2020/train_valid_data/train/masks")
    # test_data = BraTS_dataset(image_dir = "~/home/jesperdlau/Documents/Intro_Intelligente_Systemer/Januarprojekt/02461_January_project/BraTS2020/train_valid_data/train/images",
    #                           mask_dir  = "~/home/jesperdlau/Documents/Intro_Intelligente_Systemer/Januarprojekt/02461_January_project/BraTS2020/train_valid_data/train/masks")
    test_data = BraTS_dataset(image_dir="/home/jesperdlau/BraTS_Training_Data/train_data/train/images",
                               mask_dir="/home/jesperdlau/BraTS_Training_Data/train_data/train/masks")
    # train_dir = '/home/jesperdlau/BraTS_Training_Data/train_data/train/images'

    
    test_dataloader = DataLoader(test_data, batch_size=2, shuffle=True)
    
    # print(test_data)
    
    test_feature, test_label = next(iter(test_dataloader))
    print(np.shape(test_feature), np.shape(test_label))
    # print(f"Feature batch shape: {test_feature.size()}")
    # print(f"Labels batch shape: {test_label.size()}")
    
    # img = test_features[0].squeeze()
    # label = test_labels[0]
    
    # plot_2d_tensor(test_feature, test_label, 75)
    
    # plt.imshow(img, cmap="gray")
    # plt.show()
    # print(f"Label: {label}")
    
    
    
    
    