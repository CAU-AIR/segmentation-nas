import os
import sys
import random
import argparse
import numpy as np
from datetime import datetime

from seg_utils.utils import AverageMeter, get_iou_score
from seg_utils.loss import DiceBCELoss
from seg_utils.dataset_utils import load_data, train_test_split, ImageDataset

import torch
import torch.nn as nn
from torchvision import transforms
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

import segmentation_models_pytorch as smp
from seg_models.models import load_model

def train(model, device, train_loader, optimizer, criterion, epoch):
    iou = AverageMeter()
    losses = AverageMeter()

    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        output = model(data)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # iou_score = get_iou_score(output, target)
        target = target.long()
        tp, fp, fn, tn = smp.metrics.get_stats(output, target, 'binary', threshold=0.5)
        iou_score = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")

        batch_size = data.size(0)
        losses.update(loss.item(), batch_size)
        iou.update(iou_score, batch_size)

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

            output = model(data)

            # iou_score = get_iou_score(output, target)
            target = target.long()
            tp, fp, fn, tn = smp.metrics.get_stats(output, target, 'binary', threshold=0.5)
            iou_score = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
    
            batch_size = data.size(0)
            iou.update(iou_score, batch_size)

            # ori_image = data * 255.0
            # output_image = output * 255.0

            # ori_image[:, 0] = output_image[:, 0]

            # # 두 배열을 RGB 이미지로 결합
            # for n, (ori_img, out_img) in enumerate(zip(ori_image, output_image)):
            #     index = batch_idx * (len(ori_image))
            #     ori_img = TF.to_pil_image(ori_img.squeeze().byte(), mode='RGB')
            #     ori_img.save("overlap/output_image" + str(index + n) + ".jpg")

            #     out_img = TF.to_pil_image(out_img.squeeze().byte(), mode='L')
            #     out_img.save("results/output_image" + str(index + n) + ".jpg")

            ender.record()
            torch.cuda.synchronize()
            torch.cuda.synchronize()
            latency_time = starter.elapsed_time(ender) / data.size(0)    # μs ()
            torch.cuda.empty_cache()

            latency.update(latency_time)

    return iou.avg, latency.avg


parser = argparse.ArgumentParser()
# General Settings
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--device', type=str, default='0')
# Dataset Settings
parser.add_argument('--root', type=str, default='../data/image')
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
    logs.init(config=args, project='Segmentation NAS', name="DeepLabv3+_Adam")

    torch.multiprocessing.set_start_method('spawn')

    device = 'cuda:' + args.device
    args.device = torch.device(device)
    torch.cuda.set_device(args.device)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

    model = load_model(args, args.model)
    model.to(device)

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

    # loss = smp.losses.DiceLoss('binary')
    # loss = DiceBCELoss(weight=0.5)
    # loss = loss.to(device)
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-5)
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    loss = nn.BCEWithLogitsLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=2e-4)

    best_test_iou = -float('inf')  # Initialize the best IoU with a very low number
    timestamp = "/" + datetime.now().strftime("%H_%M_%S")  + "/"
    save_dir="output/" + str(datetime.now().date()) + timestamp
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for epoch in range(1, args.epoch + 1):
        train_loss, train_iou = train(model, args.device, train_loader, optimizer, loss, epoch)
        logs.log({"Architecture test mIoU": train_iou})

        test_iou, latency = test(model, args.device, test_loader)
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