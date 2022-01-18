# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 14:55:29 2022

@author: Rasmus
"""

from evaluation import full_evaluation
import glob 
import numpy as np
import torch

def statistics(directory, save_data, print_stats):
    predictions_list = sorted(glob.glob(str(directory) + 'predictions/*'))
    mask_list  = sorted(glob.glob(str(directory) + 'masks/*'))
    
    pixel_scores = np.array([])
    iou_scores = np.array([])
    dice_scores = np.array([])
    
    for i in range(len(predictions_list)):
    
        prediction = np.load(predictions_list[i])
        mask  = np.load(mask_list[i])
        
        # prediction = np.expand_dims(prediction, axis=0)
        # mask = np.expand_dims(mask, axis=0)
        pixel_score, iou_score, dice_score = full_evaluation(prediction,mask,print_stats)
        
        pixel_scores = np.append(pixel_scores, pixel_score)
        iou_scores = np.append(iou_scores, iou_score)
        dice_scores = np.append(dice_scores, dice_score)

    if save_data:
        np.savetxt('pixel_data.csv', pixel_scores, delimiter=",")
        np.savetxt('iou_data.csv', iou_scores, delimiter=",")
        np.savetxt('dice_data.csv', dice_scores, delimiter=",")        
        
if __name__ =="__main__":
    directory = "SKRIV DIRECTORY"
    statistics(directory, True, True)

        