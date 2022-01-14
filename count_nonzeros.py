# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 13:16:48 2022

@author: Rasmus
"""

import torch 
import numpy as np

#from plot_function import plot_2d_data

def count_nonzeros_in_tensor(mask):
    
    count = torch.count_nonzero(mask)
    #count = torch.bincount(mask)
    
    return count


if __name__ == "__main__":
    num = 2
    test_image = torch.from_numpy(np.load(f"./BraTS2020/train_valid_data/train/images/image_{num}.npy"))
    
    test_mask = torch.from_numpy(np.load(f"./BraTS2020/train_valid_data/train/masks/mask_{num}.npy"))
   # plot_2d_data(test_image, test_mask, 75)
    print(count_nonzeros_in_tensor(test_mask))
