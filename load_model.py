# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 12:12:38 2022

@author: Rasmus
"""

import torch
import torch.nn as nn 

def load_model(file):

    loaded_model = Model(n_input_features=????)
    loaded_model.load_state_dict(torch.load(file))
    loaded_model.eval()


if __name__=="__main__":
    load_model()