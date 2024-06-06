import os
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from loss import DiceBCELoss

from dataloaders import load_data, train_val_test_split, ImageDataset, set_transforms
from supernet_dense import SuperNet, SampledNetwork
from param import CONFIG
from train_supernet import train_architecture
from utils import check_tensor_in_list, save_image
from train_samplenet import train_samplenet, check_gpu_latency, check_cpu_latency
import numpy as np
import random

# logging
import wandb

# set random seed
random.seed(CONFIG["SEED"])
np.random.seed(CONFIG["SEED"])
torch.manual_seed(CONFIG["SEED"])
torch.cuda.manual_seed(CONFIG["SEED"])
torch.cuda.manual_seed_all(CONFIG["SEED"])

gpu_index = ','.join(map(str, CONFIG["GPU"]))
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# os.environ["CUDA_VISIBLE_DEVICES"] = gpu_index


data_dir = CONFIG["DATA"]["data_dir"]
label_dir = CONFIG["DATA"]["label_dir"]
batch_size = CONFIG["DATA"]["batch_size"]
resize = CONFIG["DATA"]["shape"]
clip_grad = CONFIG["TRAIN"]["clip_grad"]
num_epochs = CONFIG["TRAIN"]["num_epochs"]
warmup_epochs = CONFIG["TRAIN"]["warmup_epochs"]
alpha_lr = CONFIG["TRAIN"]["alpha_lr"]
weight_lr = CONFIG["TRAIN"]["weight_lr"]
weight_decay = CONFIG["TRAIN"]["weight_decay"]
sample_weight_lr = CONFIG["TRAIN"]["sample_weight_lr"]

data = load_data(data_dir)
transform = set_transforms(*resize)
train_data, val_data, test_data = train_val_test_split(data)

train_dataset = ImageDataset(train_data, transform)
val_dataset = ImageDataset(val_data, transform)
test_dataset = ImageDataset(test_data, transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


def main():
    logs = wandb
    login_key = '1623b52d57b487ee9678660beb03f2f698fcbeb0'
    logs.login(key=login_key)
    logs.init(config=CONFIG, project='Segmentation NAS', name="DARTS_F"+str(len(data)))

    model = SuperNet(n_class=1)
    # loss = DiceBCELoss(weight=CONFIG["TRAIN"]["loss_weight"])
    loss = nn.BCEWithLogitsLoss()
    # loss = nn.BCEWithLogitsLoss(weight=CONFIG["TRAIN"]["loss_weight"])

    primary_gpu = CONFIG["GPU"][0]  # This will be 1 in your case
    torch.cuda.set_device(primary_gpu)  # Set the primary GPU
    model.cuda(primary_gpu)  # Move the model to the primary GPU before wrapping with DataParallel
    loss = loss.cuda()

    if torch.cuda.is_available():
        if len(CONFIG["GPU"]) >= 2:
            model = torch.nn.DataParallel(model, device_ids=CONFIG["GPU"])
            print("Using mult-gpu")
        else:
            print("Using single-gpu")

    alphas_params = [
        param for name, param in model.named_parameters() if "alphas" in name
    ]
    optimizer_alpha = torch.optim.Adam(alphas_params, lr=alpha_lr)

    params_except_alphas = [
        param
        for param in model.parameters()
        if not check_tensor_in_list(param, alphas_params)
    ]
    optimizer_weight = torch.optim.Adam(
       params_except_alphas, lr=weight_lr, weight_decay=weight_decay
    )

    timestamp = "/" + datetime.now().strftime("%H_%M_%S")  + "/"
    save_dir="output/" + str(datetime.now().date()) + timestamp
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    train_architecture(
        model,
        train_loader,
        val_loader,
        test_loader,
        loss,
        optimizer_weight,
        optimizer_alpha,
        num_epochs,
        warmup_epochs,
        clip_grad,
        logs,
        save_dir=save_dir
    )

    sampled_model = SampledNetwork(model)
    if len(CONFIG["GPU"]) >= 2:
        sampled_model = torch.nn.DataParallel(sampled_model, device_ids=CONFIG["GPU"])
        print("Using mult-gpu")
    else:
        print("Using single-gpu")

    optimizer = torch.optim.Adam(sampled_model.parameters(), lr=sample_weight_lr)

    train_samplenet(
        sampled_model, train_loader, test_loader, loss, optimizer, num_epochs, logs, save_dir
    )

    gpu_time = check_gpu_latency(sampled_model, resize[0], resize[1])
    logs.log({"GPU time": gpu_time})
    
    print(f"GPU Latency: {gpu_time:.4f} ms")
    # cpu_time = check_cpu_latency(sampled_model, resize[0], resize[1])
    # print(f"CPU Latency: {cpu_time:.4f} ms")
    # sampled_model = sampled_model.cuda()
    # save_image(sampled_model, test_loader, test_dataset)


if __name__ == "__main__":
    main()
