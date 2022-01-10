# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 13:13:23 2022

@author: Rasmus
"""


import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from model import UNet 
from data_load import BraTS_dataset


def training():
    # parametrer
    learning_rate = 0.01
    num_epochs = 20 
    model = UNet()
    
    # Så det i en video, ved ikke om vi får brug for det i forhold til HPC
    device = "cuda" if torch.cuda.is_available() else torch.device("cpu")
    
    # initialisering 
    model.to(device)
    
    # Loss function
    loss = torch.nn.CrossEntropyLoss() 
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
    
    # Få dataen
    test_data = BraTS_dataset(image_dir = "./BraTS2020/train_valid_data/train/images", 
                              mask_dir = "./BraTS2020/train_valid_data/train/masks")
    
    dataloader = DataLoader(test_data, batch_size=2, shuffle=True)
    
    # Sætter i trænings mode
    model.train()
    
    # Trænings loop
    for epoch in range(num_epochs):
        
        for feature, mask in dataloader:
            out = model(feature)
            l = loss(out, mask)    
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            
            print(f"loss = {l}")
    
if __name__ == "__main__":
    training()