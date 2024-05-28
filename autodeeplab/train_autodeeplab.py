import os
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from collections import OrderedDict
from dataloaders import make_data_loader
from modeling.sync_batchnorm.replicate import patch_replication_callback
from modeling.deeplab import *
from utils.loss import SegmentationLosses
from utils.lr_scheduler import LR_Scheduler
from utils.saver import Saver
from utils.metrics import Evaluator, AverageMeter
from auto_deeplab import AutoDeeplab
from config_utils.search_args import obtain_search_args
from utils.copy_state_dict import copy_state_dict

import segmentation_models_pytorch as smp

# logging
import wandb

torch.backends.cudnn.benchmark = True

class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.args.data_count = 0

        kwargs = {'num_workers': args.workers, 'pin_memory': True}
        self.train_loaderA, self.train_loaderB, self.val_loader, self.test_loader, self.nclass = make_data_loader(args, **kwargs)

        # Define Saver
        self.saver = Saver(args)
        self.saver.save_experiment_config()

        self.logs = wandb
        login_key = '1623b52d57b487ee9678660beb03f2f698fcbeb0'
        self.logs.login(key=login_key)
        self.logs.init(config=self.args, project='Segmentation NAS', name="AutoDeepLab_"+str(args.layer)+"layer_F" + str(self.args.data_count))

        # weight = None
        # self.criterion = SegmentationLosses(weight=weight, cuda=args.cuda).build_loss(mode=args.loss_type)
        # self.criterion = smp.losses.DiceLoss('binary')
        self.criterion = nn.BCEWithLogitsLoss()

        # Define network
        model = AutoDeeplab(self.nclass, args.layer, self.criterion, self.args.filter_multiplier,
                            self.args.block_multiplier, self.args.step)
        optimizer = torch.optim.SGD(
                model.weight_parameters(),
                args.lr,
                momentum=args.momentum,
                weight_decay=args.weight_decay
            )

        self.model, self.optimizer = model, optimizer

        self.architect_optimizer = torch.optim.Adam(self.model.arch_parameters(),
                                                    lr=args.arch_lr, betas=(0.9, 0.999),
                                                    weight_decay=args.arch_weight_decay)

        # Define Evaluator
        self.evaluator = Evaluator(self.nclass)
        # Define lr scheduler
        self.scheduler = LR_Scheduler(args.lr_scheduler, args.lr,
                                      args.epochs, len(self.train_loaderA), min_lr=args.min_lr)
        # TODO: Figure out if len(self.train_loader) should be devided by two ? in other module as well
        # Using cuda
        if args.cuda:
            self.model = self.model.cuda(args.gpu_ids[0])

        # Using data parallel
        if args.cuda and len(self.args.gpu_ids) > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpu_ids)
            patch_replication_callback(self.model)
            print('training on multiple-GPUs')

        #checkpoint = torch.load(args.resume)
        #print('about to load state_dict')
        #self.model.load_state_dict(checkpoint['state_dict'])
        #print('model loaded')
        #sys.exit()

        # Resuming checkpoint
        self.best_pred = 0.0
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'" .format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']

            # if the weights are wrapped in module object we have to clean it
            if args.clean_module:   # clean_module=0
                self.model.load_state_dict(checkpoint['state_dict'])
                state_dict = checkpoint['state_dict']
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k[7:]  # remove 'module.' of dataparallel
                    new_state_dict[name] = v
                # self.model.load_state_dict(new_state_dict)
                copy_state_dict(self.model.state_dict(), new_state_dict)

            else:
                if torch.cuda.device_count() > 1 or args.load_parallel: # args.load_parallel=0
                    # self.model.module.load_state_dict(checkpoint['state_dict'])
                    copy_state_dict(self.model.module.state_dict(), checkpoint['state_dict'])
                else:
                    # self.model.load_state_dict(checkpoint['state_dict'])
                    copy_state_dict(self.model.state_dict(), checkpoint['state_dict'])


            if not args.ft: # args.ft=False
                # self.optimizer.load_state_dict(checkpoint['optimizer'])
                copy_state_dict(self.optimizer.state_dict(), checkpoint['optimizer'])
            self.best_pred = checkpoint['best_pred']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))

        # Clear start epoch if fine-tuning
        if args.ft:
            args.start_epoch = 0

    def training(self, epoch):
        train_loss = AverageMeter()
        tbar = tqdm(self.train_loaderA)
        num_img_tr = len(self.train_loaderA)

        self.model.train()
        for i, sample in enumerate(tbar):
            # image, target = sample['image'], sample['label']
            image, target = sample[0], sample[1]
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()

            self.scheduler(self.optimizer, i, epoch, self.best_pred)
            self.optimizer.zero_grad()
            output = self.model(image)
            loss = self.criterion(output, target)

            loss.backward()
            self.optimizer.step()

            if epoch >= self.args.alpha_epoch:  # args.alpha_epoch=20
                search = next(iter(self.train_loaderB))
                # image_search, target_search = search['image'], search['label']
                image_search, target_search = search[0], search[1]
                if self.args.cuda:
                    image_search, target_search = image_search.cuda (), target_search.cuda ()

                self.architect_optimizer.zero_grad()
                output_search = self.model(image_search)
                arch_loss = self.criterion(output_search, target_search)

                arch_loss.backward()
                self.architect_optimizer.step()

            img_size = image.data.shape[0]
            train_loss.update(loss.item(), img_size)
            tbar.set_description('Train loss: %.3f' % (train_loss.avg / (i + 1)))
            #self.writer.add_scalar('train/total_loss_iter', loss.item(), i + num_img_tr * epoch)

        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        print('Loss: %.3f' % train_loss.avg)

        if self.args.no_val:
            # save checkpoint every epoch
            is_best = False
            if torch.cuda.device_count() > 1:
                state_dict = self.model.module.state_dict()
            else:
                state_dict = self.model.state_dict()
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': state_dict,
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, is_best)


    def validation(self, epoch):
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.val_loader, desc='\r')
        
        val_loss = AverageMeter()
        val_iou = AverageMeter()

        for i, sample in enumerate(tbar):
            # image, target = sample['image'], sample['label']
            image, target = sample[0], sample[1]
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            with torch.no_grad():
                output = self.model(image)
            loss = self.criterion(output, target)
            
            val_loss.update(loss.item())
            tbar.set_description('Test loss: %.3f' % (val_loss.avg / (i + 1)))

            # pred = output.data.cpu().numpy()
            # target = target.cpu().numpy()
            # pred = np.argmax(pred, axis=1)
            # pred = np.expand_dims(pred, axis=1)
            # # Add batch sample into evaluator
            # self.evaluator.add_batch(target, pred)

            output = torch.sigmoid(output)
            target = target.long()
            tp, fp, fn, tn = smp.metrics.get_stats(output, target, "binary", threshold=0.5)
            iou_score = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
            val_iou.update(iou_score, image.data.shape[0])

        # Fast test during the training
        # Acc = self.evaluator.Pixel_Accuracy()
        # Acc_class = self.evaluator.Pixel_Accuracy_Class()
        # mIoU = self.evaluator.Mean_Intersection_over_Union()
        # FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
        mIoU = val_iou.avg

        print('Validation:')
        print('[Epoch: %d] mIoU: %.3f Loss: %.3f]' % (epoch, mIoU, val_loss.avg))
        # print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))
        
        new_pred = mIoU
        if new_pred > self.best_pred:
            is_best = True
            self.best_pred = new_pred
            # if torch.cuda.device_count() > 1:
            #     state_dict = self.model.module.state_dict()
            # else:
            #     state_dict = self.model.state_dict()
            state_dict = self.model.state_dict()
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': state_dict,
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, is_best)

        self.logs.log({"Architecture test mIoU": mIoU})

def main():
    args = obtain_search_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.cuda.set_device(int(args.gpu_ids[0]))

    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

    if args.sync_bn is None:
        if args.cuda and len(args.gpu_ids) > 1:
            args.sync_bn = True
        else:
            args.sync_bn = False

    # default settings for epochs, batch_size and lr
    if args.epochs is None:
        epoches = {
            'coco': 30,
            'cityscapes': 40,
            'pascal': 50,
            'kd':10,
            'sealer': 50,
        }
        args.epochs = epoches[args.dataset.lower()]

    if args.batch_size is None:
        args.batch_size = 4 * len(args.gpu_ids)

    if args.test_batch_size is None:
        args.test_batch_size = args.batch_size

    #args.lr = args.lr / (4 * len(args.gpu_ids)) * args.batch_size

    if args.checkname is None:
        args.checkname = 'deeplab-'+str(args.backbone)
    print(args)
    torch.manual_seed(args.seed)
    trainer = Trainer(args)
    print('Starting Epoch:', trainer.args.start_epoch)
    print('Total Epoches:', trainer.args.epochs)
    for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
        trainer.training(epoch)
        if not trainer.args.no_val and epoch % args.eval_interval == (args.eval_interval - 1):
            trainer.validation(epoch)


if __name__ == "__main__":
   main()
