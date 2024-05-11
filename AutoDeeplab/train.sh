DATA=sealer
BATCH=8
LAYER=6

# Architecture Search
python train_autodeeplab.py --backbone resnet --layer $LAYER --lr 0.007 --epochs 100 --batch_size $BATCH --eval_interval 1 --dataset $DATA --gpu_ids 2

# Decode
CUDA_VISIBLE_DEVICES=0 python decode_autodeeplab.py --dataset cityscapes --resume /AutoDeeplabpath/checkpoint.pth.tar

# Re-train
python train.py

