#!/bin/bash
python predict.py --model DDRNet \
                  --checkpoint /data/dingcheng/segment/checkpoint/cityscapes/DDRNet-75.24/best_model.pth \
                  --out_stride 8 \
                  --root '/data/open_data' \
                  --dataset cityscapes \
                  --predict_type validation \
                  --predict_mode whole \
                  --crop_size 1024 \
                  --tile_hw_size '1024,1024' \
                  --batch_size 2 \
                  --gpus 3 \
                  --overlap 0.3 \
                  --scales 0.5 0.75 1.0 1.25