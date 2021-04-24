#!/bin/bash

# Deeplabv3plus_res101  PSPNet_res101  DualSeg_res101  BiSeNet  BiSeNetV2  DDRNet
# FCN_ResNet  SegTrans

python -m torch.distributed.launch --nproc_per_node=2 \
                train.py --model PSPNet_res50 --out_stride 8 \
                --max_epochs 200 --val_epochs 20 --batch_size 4 --lr 0.01 --optim sgd --loss ProbOhemCrossEntropy2d \
                --base_size 768 --crop_size 768  --tile_hw_size 768,768 \
                --root '/data/open_data' --dataset cityscapes --gpus_id 1,2
