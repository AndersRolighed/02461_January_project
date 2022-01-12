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
from model import UNet, double_convolution 
from data_load import BraTS_dataset
from utilities import plot_2d_tensor

def training():
    # parametrer
    learning_rate = 0.002
    num_epochs = 50
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
    train_data = BraTS_dataset(image_dir = "./BraTS2020/train_valid_data/train/images", 
                              mask_dir = "./BraTS2020/train_valid_data/train/masks")
    
    dataloader = DataLoader(train_data, batch_size=2, shuffle=True)
    
    # Sætter i trænings mode
    model.train()
    
    # Trænings loop
    all_loss = []
    for epoch in range(num_epochs):
        print(f"Epoch number: {epoch}")
        
        for i, (feature, mask) in enumerate(dataloader):
            feature, mask = feature.float(), mask.float()
            out = model(feature)
            # print(out)
            # print(type(mask))
            
            out_np = out.detach().numpy()
            mask_np = mask.detach().numpy()
            
            plot_2d_tensor(mask_np, out_np, 75)
            #plt.imshow(out[0,:,:,75], cmap='gray')
            
            l = loss(out, mask)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            
            all_loss += [l.detach().numpy()]
            
            plt.plot(np.array(all_loss))
            plt.show()
            
            print(f"loss = {l}")
            
            if l>2:
                print("fuck...")
            elif l>1:
                print("Terrible")
            elif l>0.3:
                print("Not great")
            elif l<0.2:
                print("Fine")
            elif l<0.05:
                print("On the right track, buddy")
            elif l<0.02:
                print("yay :D")
    
    # Gemme netværket
    
    FILE = "model2.pth"
    
    torch.save(model.state_dict(),FILE)
            
if __name__ == "__main__":
    training()