import os
import random
import numpy as np
from PIL import Image
import time

import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, Subset

__all__ = ['dataset', 'split_dataset']

class Sealer(Dataset):
    def __init__(self, args, root_dir, transform=None):
        self.args = args
        self.root_dir = root_dir
        self.transform = transform
        self.NUM_CLASSES = 1

        self.image_paths = []

        for root, dirs, files in os.walk(root_dir):
            for file in files:
                if file.endswith(".jpg"):
                    self.image_paths.append(os.path.join(root, file))


    def __getitem__(self, index):
        # 이미지와 레이블 로드
        data_dir = self.image_paths[index]
        label_dir = self.image_paths[index].replace("image", "target")

        image = Image.open(data_dir)
        label = Image.open(label_dir)

        # 전처리 적용
        if self.transform is not None:
            image = self.transform(image)
            label = self.transform(label)[0].unsqueeze(0)

        return image, label

    def __len__(self):
        return len(self.image_paths)
    

def split_dataset(dataset, train_size=0.8, val_size=0.1, test_size=0.1):
    dataset_size = len(dataset)
    train_size = int(train_size * dataset_size)
    val_size = int(val_size * dataset_size)
    test_size = int(test_size * dataset_size)

    # 데이터셋의 인덱스를 랜덤하게 섞음
    indices = list(range(dataset_size))
    random.shuffle(indices)

    # 학습 세트와 테스트 세트로 분할
    train_indices = indices[:train_size]
    val_indices = indices[train_size : train_size + val_size]
    test_indices = indices[train_size + val_size :]

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    return train_dataset, val_dataset, test_dataset

    
if __name__ == '__main__':
    # 데이터 전처리를 위한 변환기 정의
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        ])

    # 커스텀 데이터셋 객체 생성
    root_dir = "train_data"  # 데이터셋이 있는 루트 폴더 경로
    datasets = Sealer(root_dir, transform=transform)

    train_ratio = 0.8  # 학습 세트의 비율 (0.8은 80%의 데이터가 학습 세트로 사용됨을 의미)
    train_dataset, test_dataset = split_dataset(datasets, train_ratio)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)