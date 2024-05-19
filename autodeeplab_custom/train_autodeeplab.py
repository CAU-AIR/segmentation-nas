import argparse
import os
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from mypath import Path
from dataloaders import make_data_loader
from modeling.sync_batchnorm.replicate import patch_replication_callback
from modeling.deeplab import *
from utils.loss import SegmentationLosses
from utils.calculate_weights import calculate_weigths_labels
from utils.lr_scheduler import LR_Scheduler
from utils.saver import Saver
from utils.summaries import TensorboardSummary
from utils.metrics import Evaluator, AverageMeter
from auto_deeplab import AutoDeeplab
from architect import Architect
import segmentation_models_pytorch as smp
import torchvision.transforms.functional as TF

# logging
import wandb

# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

class Trainer(object):
    def __init__(self, args):
        self.args = args

        # Define Saver
        self.saver = Saver(args)
        self.saver.save_experiment_config()
        # Define Tensorboard Summary
        # self.summary = TensorboardSummary(self.saver.experiment_dir)
        # self.writer = self.summary.create_summary()

        self.logs = wandb
        login_key = '1623b52d57b487ee9678660beb03f2f698fcbeb0'
        self.logs.login(key=login_key)
        self.logs.init(config=args, project='Segmentation NAS', name="AutoDeepLab_"+str(args.layer)+"layer")

        # Define Dataloader
        kwargs = {'num_workers': args.workers, 'pin_memory': True}
        self.train_loader1, self.train_loader2, self.val_loader, _, self.nclass = make_data_loader(args, **kwargs)
        
        # Define Criterion
        # whether to use class balanced weights
        if args.use_balanced_weights:
            classes_weights_path = os.path.join(Path.db_root_dir(args.dataset), args.dataset+'_classes_weights.npy')
            if os.path.isfile(classes_weights_path):
                weight = np.load(classes_weights_path)
            else:
                weight = calculate_weigths_labels(args.dataset, self.train_loader, self.nclass)
            weight = torch.from_numpy(weight.astype(np.float32))
        else:
            weight = None
        # self.criterion = SegmentationLosses(weight=weight, cuda=args.cuda).build_loss(mode=args.loss_type)
        self.criterion = smp.losses.DiceLoss('binary')

        # Define network
        # model = AutoDeeplab(self.nclass, 12, self.criterion, crop_size=self.args.crop_size)
        model = AutoDeeplab(self.nclass, args.layer, self.criterion, crop_size=self.args.crop_size)
        optimizer = torch.optim.SGD(
                model.weight_parameters(),
                args.lr,
                momentum=args.momentum,
                weight_decay=args.weight_decay
            )

        # Using cuda
        if args.cuda:
            torch.cuda.set_device(args.gpu_ids[0])  # Set the primary GPU
            model.cuda(args.gpu_ids[0])

            if len(args.gpu_ids) >= 2:
                self.model = torch.nn.DataParallel(self.model, device_ids=args.gpu_ids)
                patch_replication_callback(self.model)
                print("Using mult-gpu")
            else:
                print("Using single-gpu")

        # Define Optimizer
        self.model, self.optimizer = model, optimizer

        self.architect_optimizer = torch.optim.Adam(self.model.arch_parameters(),
                                            lr=args.arch_lr, betas=(0.9, 0.999),
                                            weight_decay=args.arch_weight_decay)

        # Define Evaluator
        self.evaluator = Evaluator(self.nclass)
        # Define lr scheduler
        self.scheduler = LR_Scheduler(args.lr_scheduler, args.lr,
                                            args.epochs, len(self.train_loader1))

        # self.architect = Architect(self.model, args)
        
        # Resuming checkpoint
        self.best_pred = 0.0
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'" .format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            if args.cuda:
                self.model.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint['state_dict'])
            if not args.ft:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.best_pred = checkpoint['best_pred']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))

        # Clear start epoch if fine-tuning
        if args.ft:
            args.start_epoch = 0

    def training(self, epoch):
        train_loss = 0.0
        self.model.train()
        tbar = tqdm(self.train_loader1)

        train_loss = AverageMeter()

        for i, sample in enumerate(tbar):
            image, target = sample[0], sample[1]
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            
            self.scheduler(self.optimizer, i, epoch, self.best_pred)
            self.optimizer.zero_grad()
            output = self.model(image)
            loss = self.criterion(output, target)
            
            loss.backward()
            self.optimizer.step()

            if epoch >= self.args.alpha_epoch:
                search = next(iter(self.train_loader2))
                image_search, target_search = search[0], search[1]
                if self.args.cuda:
                    image_search, target_search = image_search.cuda (), target_search.cuda () 

                self.architect_optimizer.zero_grad()
                output_search = self.model(image_search)
                arch_loss = self.criterion(output_search, target_search)

                arch_loss.backward()
                self.architect_optimizer.step()

            train_loss.update(loss.item())
            tbar.set_description('Train loss: %.3f' % (train_loss.avg / (i + 1)))

        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        print('Loss: %.3f' % train_loss.avg)

        if self.args.no_val:
            # save checkpoint every epoch
            print("save train model")
            is_best = False
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict(),
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

            iou_score = self.evaluator.get_iou_score(output, target)
            val_iou.update(iou_score)

        mIoU = val_iou.avg
        
        self.logs.log({"Architecture test mIoU": mIoU})
        
        print('Validation:')
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        print("Validation : mIoU:{}".format(mIoU))
        print('Loss: %.3f' % val_loss.avg)

        new_pred = float(mIoU)
        if new_pred > self.best_pred:
            print("save best val model")
            is_best = True
            self.best_pred = new_pred
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, is_best)

    ''' Test
    # def test(self, epoch):
    #     test_iou = AverageMeter()
    #     latency = AverageMeter()

    #     self.model.eval()
    #     with torch.no_grad():
    #         # warm_up for latency calculation
    #         rand_img = torch.rand(1, 3, 128, 128).cuda()
    #         for _ in range(10):
    #             _ = self.model(rand_img)

    #         starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    #         for batch_idx, (data, target) in enumerate(self.test_loader):
    #             torch.cuda.synchronize()
    #             torch.cuda.synchronize()
    #             starter.record()

    #             data, target = data.cuda(), target.cuda()
    #             output = self.model(data)

    #             # ori_image = data * 255.0
    #             # output_image = output * 255.0

    #             # ori_image[:, 0] = output_image[:, 0]

    #             # # 두 배열을 RGB 이미지로 결합
    #             # for n, (ori_img, out_img) in enumerate(zip(ori_image, output_image)):
    #             #     index = batch_idx * (len(ori_image))
    #             #     ori_img = TF.to_pil_image(ori_img.squeeze().byte(), mode='RGB')
    #             #     ori_img.save("overlap/output_image" + str(index + n) + ".jpg")

    #             #     out_img = TF.to_pil_image(out_img.squeeze().byte(), mode='L')
    #             #     out_img.save("results/output_image" + str(index + n) + ".jpg")

    #             iou_score = self.evaluator.get_iou_score(output, target)
    #             # iou.update(iou_score, self.args.batch_size)
    #             test_iou.update(iou_score)

    #             ender.record()
    #             torch.cuda.synchronize()
    #             torch.cuda.synchronize()
    #             latency_time = starter.elapsed_time(ender) / data.size(0)    # μs ()
    #             torch.cuda.empty_cache()

    #             latency.update(latency_time, self.args.batch_size)

    #     mIoU = float(test_iou.avg)

    #     latency_avg = latency.avg
    #     fps = 1000./latency_avg
    #     sec = latency_avg/1000.

    #     print("Test : mIoU:{}, FPS; {}, Sec; {}".format(mIoU, fps, sec))
    #     self.logs.log({"Test mIoU": mIoU})
    #     self.logs.log({"GPU time": latency_avg})

    #     new_pred = float(mIoU)
    #     if new_pred > self.best_pred:
    #         print("save best test model")
    #         is_best = True
    #         self.best_pred = new_pred
    #         self.saver.save_checkpoint({
    #             'epoch': epoch + 1,
    #             'state_dict': self.model.state_dict(),
    #             'optimizer': self.optimizer.state_dict(),
    #             'best_pred': self.best_pred,
    #         }, is_best)
    '''

def main():
    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Training")
    parser.add_argument('--backbone', type=str, default='resnet',
                        choices=['resnet', 'xception', 'drn', 'mobilenet'],
                        help='backbone name (default: resnet)')
    parser.add_argument('--out_stride', type=int, default=16,
                        help='network output stride (default: 8)')
    parser.add_argument('--dataset', type=str, default='sealer',
                        choices=['pascal', 'coco', 'cityscapes', 'sealer'],
                        help='dataset name (default: sealer)')
    parser.add_argument('--use_sbd', action='store_true', default=False,
                        help='whether to use SBD dataset (default: True)')
    parser.add_argument('--workers', type=int, default=0,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--base_size', type=int, default=128,
                        help='base image size')
    parser.add_argument('--crop_size', type=int, default=128,
                        help='crop image size')
    parser.add_argument('--resize', type=int, default=128,
                        help='resize image size')
    parser.add_argument('--sync_bn', type=bool, default=None,
                        help='whether to use sync bn (default: auto)')
    parser.add_argument('--freeze_bn', type=bool, default=False,
                        help='whether to freeze bn parameters (default: False)')
    parser.add_argument('--loss_type', type=str, default='ce',
                        choices=['ce', 'focal'],
                        help='loss func type (default: ce)')
    # training hyper params
    parser.add_argument('--epochs', type=int, default=None, metavar='N',
                        help='number of epochs to train (default: auto)')
    parser.add_argument('--alpha_epoch', type=int, default=20,
                        metavar='N', help='epoch to start training alphas')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--batch_size', type=int, default=None,
                        metavar='N', help='input batch size for \
                                training (default: auto)')
    parser.add_argument('--test_batch_size', type=int, default=None,
                        metavar='N', help='input batch size for \
                                testing (default: auto)')
    parser.add_argument('--use_balanced_weights', action='store_true', default=False,
                        help='whether to use balanced weights (default: False)')
    parser.add_argument('--layer', type=int, default=6,
                        help='set autodeeplab layer')
    # optimizer params
    parser.add_argument('--lr', type=float, default=0.025, metavar='LR',
                        help='learning rate (default: auto)')
    parser.add_argument('--arch_lr', type=float, default=3e-3, 
                       help='learning rate for alpha and beta in architect searching process')

    parser.add_argument('--lr_scheduler', type=str, default='cos',
                        choices=['poly', 'step', 'cos'],
                        help='lr scheduler mode: (default: cos)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=3e-4,
                        metavar='M', help='w-decay (default: 5e-4)')
    parser.add_argument('--arch_weight_decay', type=float, default=1e-3,
                        metavar='M', help='w-decay (default: 5e-4)')

    parser.add_argument('--nesterov', action='store_true', default=False,
                        help='whether use nesterov (default: False)')
    # cuda, seed and logging
    parser.add_argument('--no_cuda', action='store_true', default=
                        False, help='disables CUDA training')
    parser.add_argument('--gpu_ids', nargs='*', type=int, default=[0],
                        help='which GPU to train on (default: 0) / input = 0 1')
    parser.add_argument('--seed', type=int, default=0, metavar='S',
                        help='random seed (default: 1)')
    # checking point
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--checkname', type=str, default=None,
                        help='set the checkpoint name')
    # finetuning pre-trained models
    parser.add_argument('--ft', action='store_true', default=False,
                        help='finetuning on a different dataset')
    # evaluation option
    parser.add_argument('--eval_interval', type=int, default=1,
                        help='evaluuation interval (default: 1)')
    parser.add_argument('--no_val', action='store_true', default=False,
                        help='skip validation during training')
    

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    if args.sync_bn is None:
        if args.cuda and len(args.gpu_ids) > 1:
            args.sync_bn = True
        else:
            args.sync_bn = False

    # default settings for epochs, batch_size and lr
    if args.epochs is None:
        epoches = {
            'coco': 30,
            'cityscapes': 200,
            'pascal': 50,
        }
        args.epochs = epoches[args.dataset.lower()]

    if args.batch_size is None:
        args.batch_size = 4 * len(args.gpu_ids)

    if args.test_batch_size is None:
        args.test_batch_size = args.batch_size

    if args.lr is None:
        lrs = {
            'coco': 0.1,
            'cityscapes': 0.025,
            'pascal': 0.007,
        }
        #args.lr = lrs[args.dataset.lower()] / (4 * len(args.gpu_ids)) * args.batch_size


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
            print("Validation")
            trainer.validation(epoch)
        

if __name__ == "__main__":
   main()
