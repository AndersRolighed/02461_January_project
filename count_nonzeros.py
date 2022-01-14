# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 13:16:48 2022

@author: Rasmus
"""

import torch 
import numpy as np

# Tager vores masks som input, og tæller 1 tallerne i mask 2,3 og 4, da den første
# ikke opfører sig konsistent med sine værdier. 

def count_nonzeros_in_masks(mask):
    
    m_2_mask_sum = mask[1,:,:,:].sum()
    m_3_mask_sum = mask[2,:,:,:].sum()
    m_4_mask_sum = mask[3,:,:,:].sum()
    
    count = m_2_mask_sum + m_3_mask_sum + m_4_mask_sum
    
    return count

if __name__ == "__main__":
    from plot_function import plot_2d_data
    
    num = 0
    test_image = torch.from_numpy(np.load(f"./BraTS2020/train_valid_data/train/images/image_{num}.npy"))
    
    test_mask = torch.from_numpy(np.load(f"./BraTS2020/train_valid_data/train/masks/mask_{num}.npy"))
    
    plot_2d_data(test_image, test_mask, 75)
    print(count_nonzeros_in_masks(test_mask))
