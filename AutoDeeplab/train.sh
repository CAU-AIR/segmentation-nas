DATA=sealer
BATCH=8
LAYER=6

# training
python train_autodeeplab.py --backbone resnet --layer $LAYER --lr 0.007 --epochs 100 --batch_size $BATCH --eval_interval 1 --dataset $DATA --gpu_ids 0

# #test
# CUDA_VISIBLE_DEVICES=2,3 python test_autodeeplab.py --dataset $DATA --batch_size $BATCH --gpu_ids 2 3 --layer $LAYERq