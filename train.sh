#!/usr/bin/env bash

# Deeplabv3plus_res101  PSPNet_res101  DualSeg_res101  BiSeNet  BiSeNetV2  DDRNet
# FCN_ResNet  SegTrans  

python train.py --model DDRNet \
                --max_epochs 200 --val_epochs 20 --batch_size 8 --lr 0.01 --optim sgd --loss ProbOhemCrossEntropy2d \
                --base_size 1024 --crop_size 1024  --tile_hw_size 1024,1024 \
                --root '/data/open_data' --dataset cityscapes --gpus_id 2,3