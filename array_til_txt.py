# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 10:53:09 2022

@author: Rasmus
"""

import numpy as np 

arr1_data = np.load("train_focal_arr_100.npy")
arr2_data = np.load("test_focal_arr_100.npy")

arr1_pixel = arr1_data[:,0]
arr1_iou = arr1_data[:,1]
arr1_dice = arr1_data[:,2]    

arr2_pixel = arr2_data[:,0]
arr2_iou = arr2_data[:,1]
arr2_dice = arr2_data[:,2]

np.savetxt('train_focal_pixel_100.csv', arr1_pixel, delimiter=",")
np.savetxt('train_focal_iou_100.csv', arr1_iou, delimiter=",")
np.savetxt('train_focal_dice_100.csv', arr1_dice, delimiter=",")

np.savetxt('test_focal_pixel_100.csv', arr2_pixel, delimiter=",")
np.savetxt('test_focal_iou_100.csv', arr2_iou, delimiter=",")
np.savetxt('test_focal_dice_100.csv', arr2_dice, delimiter=",")

