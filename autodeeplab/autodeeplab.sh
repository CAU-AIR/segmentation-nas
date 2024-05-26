# Architecture Search
python train_autodeeplab.py --layer 12 --dataset sealer --epochs 1 --batch-size 16 --gpu-ids 3

# # Decode
python decode_autodeeplab.py --dataset sealer --resume /run/sealer/deeplab-resnet/checkpoint.pth.tar

# # Re-train
# python train.py[]