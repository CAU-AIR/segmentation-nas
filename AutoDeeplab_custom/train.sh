DATA=sealer
BATCH=8
LAYER=6
EPOCHS=1
A_EPOCH=1

# Architecture Search
python train_autodeeplab.py --backbone resnet --layer $LAYER --lr 0.1 --epochs $EPOCHS --alpha_epoch $A_EPOCH --batch_size $BATCH --dataset $DATA --gpu_ids 3

# # Decode
# python decode_autodeeplab.py --dataset $DATA --batch_size $BATCH --resume run/sealer/deeplab-resnet/experiment_0/checkpoint.pth.tar --gpu_ids 3

# # Re-train
# python train.py --batch_size $BATCH --epochs $EPOCHS

