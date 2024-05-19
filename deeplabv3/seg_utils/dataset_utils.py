import os
import cv2
import time
import random
import numpy as np
from PIL import Image

import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, Subset

__all__ = ['load_data', 'train_test_split', 'ImageDataset', 'dataset', 'split_dataset']

def load_data(data_dir):
    # data -> list of folders ('data/1/crop/', 'data/2/crop/', ...)
    data_list = []
    for folder in os.listdir(data_dir):
        if os.path.isdir(os.path.join(data_dir, folder)):
            # data_sub_dir = os.path.join(data_dir, folder, "image")
            data_sub_dir = os.path.join(data_dir, folder)
            data_list.append(data_sub_dir)

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

        # label to binary
        label[label > 0] = 1
        label = label.astype(np.float32)

        if self.transform:
            image = self.transform(image)
            label = self.transform(label)
        
        return image, label

    def get_sample_size(self, idx):
        image = cv2.imread(self.data[idx])
        return image.shape

    def get_original_image(self, idx):
        image = cv2.imread(self.data[idx])
        name = self.data[idx]
        return image, name


class dataset(Dataset):
    def __init__(self, args, root_dir, mode, transform=None):
        self.args = args
        self.root_dir = root_dir
        self.transform = transform

        self.image_paths = []
        self.label_paths = []

        target = 'ori_target' if mode == 'original' else 'target'

        # 하위 폴더 탐색하여 이미지 파일 경로 수집
        for root, _, files in os.walk(root_dir):
            if mode in root:
                for file in files:
                    if file.endswith(".tif"):
                        image_path = os.path.join(root, file)
                        self.image_paths.append(image_path)
            
            if target == os.path.split(root)[-1]:
                for file in files:
                    if file.endswith(".tif"):
                        label_path = os.path.join(root, file)
                        self.label_paths.append(label_path)


    def __getitem__(self, index):
        # 이미지와 레이블 로드
        image = Image.open(self.image_paths[index])
        label = Image.open(self.label_paths[index])

        # 전처리 적용
        if self.transform is not None:
            if self.args.model == 'ESPNet_Encoder':
                label_transform = transforms.Compose([
                    transforms.Resize((int(self.args.input_size[0]/8), int(self.args.input_size[1]/8))),
                    transforms.ToTensor(), #scaleIn = 8
                ])

                image = self.transform(image)
                label = label_transform(label)[0]

            else:
                start_time = time.time()

                image = self.transform(image)

                end_time = time.time() - start_time

                label = self.transform(label)[0]

        return image, label, end_time

    def __len__(self):
        return len(self.image_paths)
    

def split_dataset(dataset, train_ratio):
    dataset_size = len(dataset)
    train_size = int(train_ratio * dataset_size)

    # 데이터셋의 인덱스를 랜덤하게 섞음
    indices = list(range(dataset_size))
    random.shuffle(indices)

    # 학습 세트와 테스트 세트로 분할
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)

    return train_dataset, test_dataset

    
if __name__ == '__main__':
    # 데이터 전처리를 위한 변환기 정의
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        ])

    # 커스텀 데이터셋 객체 생성
    root_dir = "train_data"  # 데이터셋이 있는 루트 폴더 경로
    datasets = dataset(root_dir, transform=transform)

    train_ratio = 0.8  # 학습 세트의 비율 (0.8은 80%의 데이터가 학습 세트로 사용됨을 의미)
    train_dataset, test_dataset = split_dataset(datasets, train_ratio)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)