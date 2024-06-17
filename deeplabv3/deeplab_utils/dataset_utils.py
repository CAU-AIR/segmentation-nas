import os
import cv2
import time
import random
import numpy as np
from PIL import Image

import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, Subset

__all__ = ['load_data', 'train_test_split', 'ImageDataset']

def load_data(data_dir):
    # data -> list of folders ('data/1/crop/', 'data/2/crop/', ...)
    data_list = []
    for folder in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, folder)
        if os.path.isdir(folder_path):
            # data_sub_dir = os.path.join(data_dir, folder, "image")
            # data_sub_dir = os.path.join(data_dir, folder)
            data_list.append(folder_path)

    return data_list

def train_test_split(data, train_size=0.8, test_size=0.2):
    # split data into train, val, test
    total_size = len(data)
    train_size = int(train_size * total_size)
    test_size = int(test_size * total_size)

    # shuffle list
    np.random.shuffle(data)
    train_data = data[:train_size]
    test_data = data[train_size:]
    return train_data, test_data

class ImageDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data = []
        self.data_dir = data_dir
        self.transform = transform

        for folder in self.data_dir:
            for file in os.listdir(folder):
                if file.endswith(".jpg"):
                    self.data.append(os.path.join(folder, file))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # check if data and label are the same name after ../(dir)/
        label_dir = self.data[idx].replace("image", "target")
        
        image = cv2.imread(self.data[idx])
        label = cv2.imread(label_dir, cv2.IMREAD_GRAYSCALE)

        # # label to binary
        # label[label > 0] = 1
        # label = label.astype(np.float32)

        label[label >= 128] = 255
        label[label < 128] = 0
        
        if self.transform:
            image = self.transform(image)
            label = self.transform(label)


        mask = np.zeros((2, 128, 128), dtype=np.float32)
        mask[0, :, :] = np.where(label == 0, 1.0, 0.0)
        mask[1, :, :] = np.where(label == 1, 1.0, 0.0)

        label = torch.from_numpy(mask)

        # # Convert mask to binary (0: background, 1: target)
        # label = mask  # Assuming non-zero values are target
        
        return image, label
