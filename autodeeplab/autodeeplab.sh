GPU=3
LAYER=12
DATASET=sealer
SAVE=./run/deeplab-resnet/experiment/checkpoint.pth.tar

EPOCHS=100
BATCH=16

NETWORK='./run/deeplab-resnet/experiment/network_path_space.npy'
BACKBONE='./run/deeplab-resnet/experiment/network_path.npy'
CELL='./run/deeplab-resnet/experiment/genotype.npy'

# # Architecture Search
echo "Starting Architecture Search..."
python train_autodeeplab.py --layer $LAYER --dataset $DATASET --epochs $EPOCHS --batch-size $BATCH --gpu-ids $GPU

# Decode
echo "Starting Decode..."
python decode_autodeeplab.py --dataset $DATASET --resume $SAVE

# Re-train
echo "Starting Re-train..."
python train.py --layer $LAYER --dataset $DATASET --epochs $EPOCHS --batch-size $BATCH --gpu $GPU --net_path $NETWORK --net_arch $BACKBONE --cell_arch $CELL