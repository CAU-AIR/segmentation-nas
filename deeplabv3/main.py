import os
import sys
import random
import argparse
import numpy as np
from datetime import datetime

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.models import segmentation
import torch.optim as optim
from torchmetrics import JaccardIndex

from deeplab_utils.dataset_utils import load_data, train_test_split, ImageDataset
from deeplab_utils.utils import AverageMeter, get_iou_score

# set job name
import setproctitle
setproctitle.setproctitle('hs_park/hyundai')

def train(model, device, train_loader, optimizer, criterion):
    iou = AverageMeter()
    losses = AverageMeter()

    # mIoU calculation using torchmetrics
    miou_metric = JaccardIndex(task="multiclass", num_classes=2).to(device)
    miou_metric.reset()

    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        output = model(data)['out']
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        miou = get_iou_score(output, target)
        # miou_metric.update(output[:, 1, :, :], target[:, 1, :, :])

        batch_size = data.size(0)
        losses.update(loss.item(), batch_size)
        iou.update(miou, batch_size)

    # miou = miou_metric.compute()
    # iou.update(miou)

    return losses.avg, iou.avg


def test(model, device, test_loader):
    iou = AverageMeter()
    latency = AverageMeter()

    model.eval()
    with torch.no_grad():
        # warm_up for latency calculation
        rand_img = torch.rand(1, 3, 128, 128).to(device)
        for _ in range(10):
            _ = model(rand_img)

        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

        for batch_idx, (data, target) in enumerate(test_loader):
            torch.cuda.synchronize()
            torch.cuda.synchronize()
            starter.record()

            data, target = data.to(device), target.to(device)

            output = model(data)['out']

            miou = get_iou_score(output, target)
    
            batch_size = data.size(0)
            iou.update(miou, batch_size)

            ender.record()
            torch.cuda.synchronize()
            torch.cuda.synchronize()
            latency_time = starter.elapsed_time(ender) / data.size(0)    # μs ()
            torch.cuda.empty_cache()

            latency.update(latency_time)

    return iou.avg, latency.avg

parser = argparse.ArgumentParser()
# General Settings
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--device', type=str, default='0')
# Dataset Settings
parser.add_argument('--root', type=str, default='../dataset/image')
parser.add_argument('--input_size', nargs='+', type=int, default=[128, 128])    # ori size = [512, 672] / crop size = [128, 128]
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--split', type=int, default=0.8)
# Model Settings
parser.add_argument('--model', type=str, default='DeepLabv3', choices=['DeepLabv3', 'ESPNet', 'STDC'])
parser.add_argument('--epoch', type=int, default=50)
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
    
    logs.init(config=args, project='Segmentation NAS', name="DeepLabv3_F" + str(len(data)))

    # load model
    model = segmentation.deeplabv3_resnet101(weights='DEFAULT')
    model.classifier[4] = nn.Conv2d(256, 2, kernel_size=(1, 1), stride=(1, 1))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = nn.DataParallel(model, device_ids=[0, 1, 2])
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    # criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    best_test_iou = -float('inf')  # Initialize the best IoU with a very low number
    timestamp = "/" + datetime.now().strftime("%H_%M_%S")  + "/"
    save_dir="output/" + str(datetime.now().date()) + timestamp
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for epoch in range(1, args.epoch + 1):
        train_loss, train_iou = train(model, device, train_loader, optimizer, criterion)
        logs.log({"Architecture test mIoU": train_iou})

        test_iou, latency = test(model, device, test_loader)
        logs.log({"Test mIoU": test_iou})

        if test_iou > best_test_iou:
            best_test_iou = test_iou  # Update the best IoU
            save_path = os.path.join(save_dir, f'best_model.pt')
            torch.save(model.state_dict(), save_path)  # Save the model

        sys.stdout.write('\r')
        sys.stdout.write('Epoch [%3d/%3d] Train loss: %.4f, Train mIoU: %.4f, Test mIoU: %.4f\n' % (epoch, args.epoch, train_loss, train_iou, test_iou))
        sys.stdout.flush()

    logs.log({"Best mIoU": best_test_iou})

    print("FPS:{:.2f}".format(1000./latency))
    print("Latency:{:.2f}ms / {:.4f}s".format(latency, (latency/1000.)))
    logs.log({"GPU time": latency})



if __name__ == '__main__':
    main()