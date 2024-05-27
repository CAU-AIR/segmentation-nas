import os
import pdb
import warnings
import numpy as np

import torch
import torch.nn as nn
import torch.utils.data
import torch.backends.cudnn
import torch.optim as optim

import dataloaders
from utils.metrics import Evaluator
from utils.utils import AverageMeter
from utils.loss import build_criterion
from utils.step_lr_scheduler import Iter_LR_Scheduler
from retrain_model.build_autodeeplab import Retrain_Autodeeplab
from config_utils.re_train_autodeeplab import obtain_retrain_autodeeplab_args

# logging
import wandb
from utils.saver import Saver


def main():
    warnings.filterwarnings('ignore')
    assert torch.cuda.is_available()
    torch.backends.cudnn.benchmark = True

    args = obtain_retrain_autodeeplab_args()
    args.gpu = int(args.gpu)
    torch.cuda.set_device(args.gpu)

    if args.checkname is None:
        args.checkname = 'deeplab-'+str(args.backbone)

    # Define Saver
    saver = Saver(args)
    # saver.save_experiment_config()

    logs = wandb
    login_key = '1623b52d57b487ee9678660beb03f2f698fcbeb0'
    logs.login(key=login_key)
    logs.init(config=args, project='Segmentation NAS', name="AutoDeepLab_"+str(args.layer)+"layer")

    if args.dataset == 'pascal':
        raise NotImplementedError
    elif args.dataset == 'cityscapes':
        kwargs = {'num_workers': args.workers, 'pin_memory': True, 'drop_last': True}
        dataset_loader, num_classes = dataloaders.make_data_loader(args, **kwargs)
        args.num_classes = num_classes
    elif args.dataset == 'sealer':
        kwargs = {'num_workers': args.workers, 'pin_memory': True}
        train_loader, test_loader, num_classes = dataloaders.make_data_loader(args, **kwargs)
        args.num_classes = num_classes
    else:
        raise ValueError('Unknown dataset: {}'.format(args.dataset))

    if args.backbone == 'resnet':
        model = Retrain_Autodeeplab(args)
    else:
        raise ValueError('Unknown backbone: {}'.format(args.backbone))

    if args.criterion == 'Ohem':
        args.thresh = 0.7
        args.crop_size = [args.crop_size, args.crop_size] if isinstance(args.crop_size, int) else args.crop_size
        args.n_min = int((args.batch_size / len([args.gpu]) * args.crop_size[0] * args.crop_size[1]) // 16)
    criterion = build_criterion(args)

    # model = nn.DataParallel(model).cuda()
    model = model.cuda(args.gpu)
    
    if args.freeze_bn:
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                m.weight.requires_grad = False
                m.bias.requires_grad = False
    
    # optimizer = optim.SGD(model.module.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=0.0001)
    optimizer = optim.SGD(model.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=0.0001)

    max_iteration = len(train_loader) * args.epochs
    scheduler = Iter_LR_Scheduler(args, max_iteration, len(train_loader))
    start_epoch = 0

    if args.resume:
        if os.path.isfile(args.resume):
            print('=> loading checkpoint {0}'.format(args.resume))
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print('=> loaded checkpoint {0} (epoch {1})'.format(args.resume, checkpoint['epoch']))
        else:
            raise ValueError('=> no checkpoint found at {0}'.format(args.resume))

    best_pred = 0.0
    evaluator = Evaluator(args.num_classes)

    for epoch in range(start_epoch, args.epochs):
        losses = AverageMeter()

        # Train
        model.train()
        evaluator.reset()
        for i, sample in enumerate(train_loader):
            cur_iter = epoch * len(train_loader) + i
            scheduler(optimizer, cur_iter)

            # inputs = sample['image'].cuda()
            # target = sample['label'].cuda()
            inputs, target = sample[0].cuda(), sample[1].cuda()

            outputs = model(inputs)

            loss = criterion(outputs, target)
            if np.isnan(loss.item()) or np.isinf(loss.item()):
                pdb.set_trace()
            
            losses.update(loss.item(), args.batch_size)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            pred = outputs.data.cpu().numpy()
            target = target.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            pred = np.expand_dims(pred, axis=1)
            evaluator.add_batch(target, pred)

            mIoU = evaluator.Mean_Intersection_over_Union()

            print('epoch: {0}\t''iter: {1}/{2}\t''lr: {3:.6f}\t''loss: {loss.val:.4f} ({loss.ema:.4f})\t''mIoU: {iou:.4f}'.format(
                epoch + 1, i + 1, len(train_loader), scheduler.get_lr(optimizer), loss=losses, iou=mIoU))


        # Test
        model.eval()
        evaluator.reset()
        for i, sample in enumerate(test_loader):
            inputs, target = sample[0].cuda(), sample[1].cuda()

            outputs = model(inputs)

            pred = outputs.data.cpu().numpy()
            target = target.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            pred = np.expand_dims(pred, axis=1)
            evaluator.add_batch(target, pred)

            mIoU = evaluator.Mean_Intersection_over_Union()

            print('epoch: {0}\t''iter: {1}/{2}\t''mIoU: {3:.6f}\t'.format(epoch + 1, i + 1, len(test_loader), mIoU))

        Acc = evaluator.Pixel_Accuracy()
        Acc_class = evaluator.Pixel_Accuracy_Class()
        mIoU = evaluator.Mean_Intersection_over_Union()
        FWIoU = evaluator.Frequency_Weighted_Intersection_over_Union()

        print('Test:')
        print("[Epoch: {}] Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(epoch, Acc, Acc_class, mIoU, FWIoU))

        logs.log({"Test mIoU": mIoU})

        new_pred = mIoU
        if new_pred > best_pred:
            is_best = True
            best_pred = new_pred
            state_dict = model.state_dict()
            saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': state_dict,
                'optimizer': optimizer.state_dict(),
                'best_pred': best_pred,
            }, is_best, filename='best.pth.tar')

if __name__ == "__main__":
    main()
