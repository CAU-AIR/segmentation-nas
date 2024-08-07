# data and label with masking and save image
# dir -> ./output/"Date"/...tif
import os
import cv2
import numpy as np
import torch
from datetime import datetime
import segmentation_models_pytorch as smp
import torch.nn.functional as F

def get_iou_score(outputs, labels):
    labels = labels.long()

    outputs = F.softmax(outputs, dim=1)
    _, max_indices = torch.max(outputs, dim=1, keepdim=True)
    outputs = torch.zeros_like(outputs, dtype=torch.int64).scatter_(1, max_indices, 1)

    # tp, fp, fn, tn = smp.metrics.get_stats(outputs, labels, "binary", threshold=0.5)
    tp, fp, fn, tn = smp.metrics.get_stats(outputs, labels, "multiclass", num_classes=2)
    # tp, fp, fn, tn = smp.metrics.get_stats(outputs, labels, "multiclass", 0, num_classes=2)
    # iou_score = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
    iou_score = smp.metrics.iou_score(tp, fp, fn, tn)
    miou = torch.mean(iou_score[1]).item()

    return miou


def save_image(model, test_loader, test_dataset):
    output_dir = "./output/" + str(datetime.now().date()) + "/image/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    model.eval()
    outputs = []
    for batch_idx, (data, label) in enumerate(test_loader):
        data = data.cuda()
        output = model(data)
        outputs.append(output)

    outputs = torch.cat(outputs, dim=0)

    for i in range(len(outputs)):
        data, data_name = test_dataset.get_original_image(i)
        output = outputs[i]
        output = torch.sigmoid(output)
        output = output.cpu().detach().numpy()
        output = output.squeeze()
        # resize output to original size
        output = cv2.resize(output, (data.shape[1], data.shape[0]))
        output[output > 0.5] = 1
        output[output <= 0.5] = 0
        save_image_with_mask(data, output, data_name, output_dir)


def check_tensor_in_list(atensor, alist):
    if any([(atensor == t_).all() for t_ in alist if atensor.shape == t_.shape]):
        return True
    return False


def save_image_with_mask(data, label, data_name, output_dir="./output/"):
    # data, label shape: (height, width, channel), (height, width) numpy array
    # data -> original image, label -> predicted mask (0 or 1)

    # create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Create mask
    # Assuming 'data' is in BGR format since we're using cv2 for image saving
    mask = np.zeros_like(data, dtype=np.uint8)
    # red for the mask
    mask[label == 1] = [0, 0, 255]
    blended_image = cv2.addWeighted(data, 1, mask, 0.5, 0)

    file_path = os.path.join(output_dir, data_name)
    cv2.imwrite(file_path, blended_image)


class AverageMeter:
    def __init__(self):
        self.reset()

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
