# # !/usr/bin/env bash
# python predict.py --model FCN_ResNet \
#                   --backbone resnet50 \
#                   --pretrained \
#                   --out_stride 16 \
#                   --mult_grid \
#                   --checkpoint /data/dingcheng/segment/checkpoint/postdam/DDRNet-68.75/best_model.pth \
#                   --root '/data/dingcheng/ISPRS' \
#                   --dataset postdam \
#                   --predict_type validation \
#                   --predict_mode sliding \
#                   --tile_hw_size '1024,1024' \
#                   --batch_size 1 \
#                   --gpus 2 \
#                   --overlap 0.9 \
#                   --scales 0.75 1.0 1.25 \
#                   # --flip_merge \
#                   # --mult_grid \

# !/usr/bin/env bash
python predict.py --model DDRNet \
                  --checkpoint /data/dingcheng/segment/checkpoint/cityscapes/DDRNet/best_model.pth \
                  --root '/data/open_data' \
                  --dataset cityscapes \
                  --predict_type validation \
                  --predict_mode sliding \
                  --tile_hw_size '1024,1024' \
                  --batch_size 1 \
                  --gpus 2 \
                  --overlap 0.3 \
                  --scales 0.75 1.0 1.25 1.5 \
                #   --flip_merge \
                #   --mult_grid \