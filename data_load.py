# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 09:36:24 2022

@author: Rasmus
"""
import os, os.path
from torchvision.io import read_image
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import numpy as np
from torch.utils.data import DataLoader
import matplotlib as plt


class BraTS_dataset(Dataset):

    def __init__(self, image_dir, mask_dir):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = ToTensor()

    def __len__(self):
        return len([name for name in os.listdir(self.image_dir)])

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, os.listdir(self.image_dir)[idx])
        mask_path = os.path.join(self.mask_dir, os.listdir(self.mask_dir)[idx])

        image = np.load(image_path)
        mask = np.load(mask_path)

        image = torch.from_numpy(image)
        mask = torch.from_numpy(mask)

        return image, mask


if __name__ == "__main__":
    from utilities import plot_2d_tensor, plot_2d_tensor

    train_data = BraTS_dataset(image_dir="./BraTS2020/train_valid_data/train/images",
                               mask_dir="./BraTS2020/train_valid_data/train/masks")

    train_dataloader = DataLoader(train_data, batch_size=2, shuffle=True)

    print(train_data)

    train_feature, train_label = next(iter(train_dataloader))
    print(f"Feature batch shape: {train_feature.size()}")
    print(f"Labels batch shape: {train_label.size()}")
    # img = test_features[0].squeeze()
    # label = test_labels[0]

    plot_2d_tensor(train_feature, train_label, 75)

    # plt.imshow(img, cmap="gray")
    # plt.show()
    # print(f"Label: {label}")
