#!/usr/bin/env bash

python predict_sliding.py --dataset paris \
                          --model UNet \
                          --checkpoint /media/ding/Study/graduate/Segmentation_Torch/checkpoint/paris/UNetbs8gpu1_train/model_250.pth \
                          --tile_size 512
