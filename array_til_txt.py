# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 10:53:09 2022

@author: Rasmus
"""

import numpy as np 

BCE_data = np.load("test_bce_arr_2.npy")
focal_data = np.load("test_focal_arr_2.npy")

BCE_PIXEL = BCE_data[:,0]
BCE_IOU = BCE_data[:,1]
BCE_DICE = BCE_data[:,2] 

focal_pixel = focal_data[:,0]
focal_iou = focal_data[:,1]
focal_dice = focal_data[:,2]

np.savetxt('BCE_PIXEL.csv', BCE_PIXEL, delimiter=",")
np.savetxt('BCE_IOU.csv', BCE_IOU, delimiter=",")
np.savetxt('BCE_DICE.csv', BCE_DICE, delimiter=",")

np.savetxt('focal_pixel.csv', focal_pixel, delimiter=",")
np.savetxt('focal_iou.csv', focal_iou, delimiter=",")
np.savetxt('focal_dice.csv', focal_dice, delimiter=",")

