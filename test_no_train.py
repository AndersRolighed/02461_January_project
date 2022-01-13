#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 17:44:27 2022

@author: jesperdlau
"""


from data_load import BraTS_dataset
from test_model import UNet
from utilities import plot_prediction_mask, batchLoad_single, plot_prediction_mask


import os, os.path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader


import numpy as np
import matplotlib.pyplot as plt


def no_train(model, image):
   model = model.float()
   model.train(mode=False)
   output = model(image)
   return output


if __name__ == '__main__':
    
    # Model load
    model_path = "./model.pth"
    model = UNet()
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    
    # Print model state
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())
    
    # Image load
    directory = '/home/jesperdlau/Documents/Intro_Intelligente_Systemer/Januarprojekt/02461_January_project/BraTS2020/train_valid_data/train/'
    image, mask = batchLoad_single(directory, 5)
    
    # Run through model
    output = no_train(model, image)
    
    # Plot original image, mask and output mask
    plot_prediction_mask(output, mask, 75)
    