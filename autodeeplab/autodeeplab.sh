# Architecture Search
CUDA_VISIBLE_DEVICES=0 python train_autodeeplab.py --dataset cityscapes

# Decode
CUDA_VISIBLE_DEVICES=0 python decode_autodeeplab.py --dataset cityscapes --resume /AutoDeeplabpath/checkpoint.pth.tar

# Re-train
python train.py