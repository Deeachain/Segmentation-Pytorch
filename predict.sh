#!/bin/bash
python predict.py --model PSPNet_res50 \
                  --checkpoint /data/dingcheng/segment/checkpoint/cityscapes/PSPNet_res50/best_model.pth \
                  --out_stride 8 \
                  --root '/data/open_data' \
                  --dataset cityscapes \
                  --predict_type validation \
                  --predict_mode whole \
                  --crop_size 768 \
                  --tile_hw_size '768,768' \
                  --batch_size 2 \
                  --gpus 3 \
                  --overlap 0.3 \
                  --scales 0.5 0.75 1.0 1.25