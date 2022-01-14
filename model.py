#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  2 14:21:19 2022

@author: jesperdlau
"""

#%% Import

import numpy as np
import torch
import torch.nn as nn

#%% Functions

# input_conv og output_conv er integers som angiver dimensionen af hhv. input og output
def double_convolution(input_conv, output_conv):
    convolution = nn.Sequential(
        nn.Conv3d(input_conv, output_conv, kernel_size=3, padding="same"),
        nn.ReLU(inplace=True),
        nn.Conv3d(output_conv, output_conv, kernel_size=3, padding="same"),
        nn.ReLU(inplace=True),
        )
    return convolution

#%% Class

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        
        self.max_pool = nn.MaxPool3d(kernel_size=2, stride=2)
        # kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=Fals
        
        self.down_conv_1 = double_convolution(4, 16)
        self.down_conv_2 = double_convolution(16, 32)
        self.down_conv_3 = double_convolution(32, 64)
        self.down_conv_4 = double_convolution(64, 128)
        self.down_conv_5 = double_convolution(128, 256)
        
        self.up_trans_1 = nn.ConvTranspose3d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.up_trans_2 = nn.ConvTranspose3d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.up_trans_3 = nn.ConvTranspose3d(in_channels=64, out_channels=32, kernel_size=2, stride=2)
        self.up_trans_4 = nn.ConvTranspose3d(in_channels=32, out_channels=16, kernel_size=2, stride=2)
        self.up_trans_5 = nn.ConvTranspose3d(in_channels=16, out_channels=8, kernel_size=2, stride=2)
        
        self.up_conv_1 = double_convolution(256, 128)
        self.up_conv_2 = double_convolution(128, 64)
        self.up_conv_3 = double_convolution(64, 32)
        self.up_conv_4 = double_convolution(32, 16)
        
        self.out = nn.Conv3d(in_channels=16, out_channels=4, kernel_size=1)

    def forward(self, input_data):
        # Encoder 
        x1 = self.down_conv_1(input_data)
        x2 = self.max_pool(x1)
        x3 = self.down_conv_2(x2)
        x4 = self.max_pool(x3)
        x5 = self.down_conv_3(x4)
        x6 = self.max_pool(x5)
        x7 = self.down_conv_4(x6)
        x8 = self.max_pool(x7)
        x9 = self.down_conv_5(x8)
        
        # Decoder (Uses unnessesary memory by storing each y1..8)
        y = self.up_trans_1(x9)
        y = self.up_conv_1(torch.cat([y,x7], 1))
        y = self.up_trans_2(y)
        y = self.up_conv_2(torch.cat([y,x5], 1))
        y = self.up_trans_3(y)
        y = self.up_conv_3(torch.cat([y,x3], 1))
        y = self.up_trans_4(y)
        y = self.up_conv_4(torch.cat([y,x1], 1))
        # y9 = self.up_trans_5(y8)
        
        y = self.out(y)
        y_out = torch.sigmoid(y)

        return y_out
        
    
#%% test


if __name__ == '__main__':
    from utilities import batchLoad_single, plot_2d_tensor
    
    directory = "./BraTS2020/train_valid_data/train/"
    test_image, test_mask = batchLoad_single(directory, 0)

    # input_data = torch.rand((1, 4, 128, 128, 128))
    input_data = test_image
    
    model = UNet()
    # print(model(input_data))
    test_output = model(input_data)
    
    print(type(test_output), np.shape(test_output))
    
    # Plot test
    out_np = test_output.detach().numpy()
    mask_np = test_mask.detach().numpy()

    plot_2d_tensor(out_np, mask_np, 75)
    

