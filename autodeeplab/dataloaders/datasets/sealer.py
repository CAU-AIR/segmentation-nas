import os
import cv2
import numpy as np

from torchvision import transforms
from torch.utils.data import Dataset

def load_data(data_dir):
    # data -> list of folders ('data/1/crop/', 'data/2/crop/', ...)
    data_list = []
    for folder in os.listdir(data_dir):
        if os.path.isdir(os.path.join(data_dir, folder)):
            # data_sub_dir = os.path.join(data_dir, folder, "image")
            data_sub_dir = os.path.join(data_dir, folder)
            data_list.append(data_sub_dir)

    return data_list

def set_transforms(size=128):
    # set up transforms
    transform = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Resize((size, size)),
                                    ])
    return transform

def train_val_test_split(data, train_size=0.8, val_size=0.1, test_size=0.1):
    # split data into train, val, test
    total_size = len(data)
    train_size = int(train_size * total_size)
    val_size = int(val_size * total_size)
    test_size = int(test_size * total_size)

    # shuffle list
    np.random.shuffle(data)
    train_data = data[:train_size]
    val_data = data[train_size : train_size + val_size]
    test_data = data[train_size + val_size :]
    return train_data, val_data, test_data

def twoTrainSeg(data, transform=None):
    number_images = len(data)
    half_size = int(0.5 * number_images)
    indices_1 = data[:half_size]
    indices_2 = data[half_size:]
    # if len(indices_1) % 2 != 0 or len(indices_2) % 2 != 0:
    #     raise Exception('indices lists need to be even numbers for batch norm')

    return Sealer(indices_1, transform), Sealer(indices_2, transform)

class Sealer(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data = []
        self.data_dir = data_dir
        self.transform = transform
        self.NUM_CLASSES = 1

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

        # label to binary
        label[label > 0] = 1
        label = label.astype(np.float32)

        if self.transform:
            image = self.transform(image)
            label = self.transform(label)
            
        return image, label