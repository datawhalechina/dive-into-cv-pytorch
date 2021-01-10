#!/bin/bash


# use which cuda device
gpu_devices='1'

# load pretrain model to continue training
load=False
# path of pretrained model. Effective when load is set to True.
pretrain_path="checkpoint/pretrain/test.pth"
# directory to save checkpoint
model_save_dir='checkpoint'

# directory of train image
image='/new_train_data/ansheng/dataset/segmentation/landmark_building_identification/train'
# directory of train mask
anno='/new_train_data/ansheng/dataset/segmentation/landmark_building_identification/train_mask.csv'

# number of epochs
epochs=30
# batch size
bs=6
# learning rate
lr=0.001

# path to save training log
log_file='logs/train.log'

nohup python train.py \
    --gpu_devices=${gpu_devices} \
    --load=${load} \
    --pretrain_path=${pretrain_path} \
    --model_save_dir=${model_save_dir} \
    --image=${image} \
    --anno=${anno} \
    --epochs=${epochs} \
    --bs=${bs} \
    --lr=${lr} \
    > ${log_file} 2>&1 &

echo "training start..."
