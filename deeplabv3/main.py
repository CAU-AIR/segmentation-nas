import random
import argparse
import numpy as np

import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms
from torch.utils.data import DataLoader

from dataset_utils import load_data, train_test_split, ImageDataset


parser = argparse.ArgumentParser()
# General Settings
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--device', type=str, default='0')
# Dataset Settings
parser.add_argument('--root', type=str, default='../dataset/image')
parser.add_argument('--input_size', nargs='+', type=int, default=[128, 128])    # ori size = [512, 672] / crop size = [128, 128]
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--split', type=int, default=0.8)
# Model Settings
parser.add_argument('--model', type=str, default='DeepLabv3', choices=['DeepLabv3', 'ESPNet', 'STDC'])
parser.add_argument('--epoch', type=int, default=30)
parser.add_argument('--lr', '--learning_rate', type=float, default=0.01)

args = parser.parse_args()

def main():
    # logging
    import wandb
    logs = wandb
    login_key = '1623b52d57b487ee9678660beb03f2f698fcbeb0'
    logs.login(key=login_key)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cudnn.deterministic = True
    cudnn.benchmark = True

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(args.input_size),
    ])

    data = load_data(args.root)
    train_data, test_data = train_test_split(data)

    train_dataset = ImageDataset(train_data, transform)
    test_dataset = ImageDataset(test_data, transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    logs.init(config=args, project='Segmentation NAS', name="DeepLabv3+_F" + str(len(data)))

    for i, j in test_loader:
        print(i.shape, j.shape, j.dtype)
        break