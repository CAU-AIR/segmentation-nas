import cv2
import torch
import numpy as np
import segmentation_models_pytorch as smp

class AverageMeter (object):
    def __init__(self):
        self.reset ()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,), test=False):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        if test:
            pred = torch.argmin(output, dim=1)[None].permute(1,0)
        else:
            _, pred = output.topk(maxk, 1, True, True)

        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
    
class ToTensor(object):
    '''
    This class converts the data to tensor so that it can be processed by PyTorch
    '''
    def __init__(self, scale=1):
        '''
        :param scale: ESPNet-C's output is 1/8th of original image size, so set this parameter accordingly
        '''
        self.scale = scale # original images are 2048 x 1024

    def __call__(self, image, label):

        if self.scale != 1:
            h, w = label.shape[:2]
            image = cv2.resize(image, (int(w), int(h)))
            label = cv2.resize(label, (int(w/self.scale), int(h/self.scale)), interpolation=cv2.INTER_NEAREST)

        image = image.transpose((2,0,1))

        image_tensor = torch.from_numpy(image).div(255)
        label_tensor =  torch.LongTensor(np.array(label, dtype=np.int)) #torch.from_numpy(label)

        return [image_tensor, label_tensor]
    
def get_iou_score(outputs, labels):
    outputs = torch.sigmoid(outputs)
    labels = labels.long()
    tp, fp, fn, tn = smp.metrics.get_stats(outputs, labels, "binary", threshold=0.5)
    iou_score = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
    miou = torch.mean(iou_score).item()
    return miou