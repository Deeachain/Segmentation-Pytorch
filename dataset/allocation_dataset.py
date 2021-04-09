# _*_ coding: utf-8 _*_
"""
Time:     2020/11/21 下午5:27
Author:   Cheng Ding(Deeachain)
Version:  V 0.1
File:     allocation_dataset.py
Describe: Write during my study in Nanjing University of Information and Secience Technology
Github:   https://github.com/Deeachain
"""

import os
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm

"""
input:单通道图片
"""
root_dir = '/media/ding/Data/datasets/paris/512_image_625/crop_files/512_label'  # 存储图片的root目录
pic_ext = '*.png'  # 需要处理的图片后缀
pic_dir = os.path.join(root_dir, pic_ext)

pic_list = glob(pic_dir)  # 获取所有图片的路径，存储到列表中

test_gather = ['82', '610', '144', '553', '158', '201', '118', '109', '351', '544', '253', '134', '202', '428']
class_counter = [0, 0, 0]
label_pic_number = 0
label_pic_name = []
test_label_pic_name = []
sliding_test_label_pic_name = []
for seg in tqdm(pic_list, desc='Processing'):
    img = cv2.imread(seg, cv2.IMREAD_GRAYSCALE)
    origin_pic_number = seg.split('/')[-1].split('_')[0].split('s')[-1]
    if origin_pic_number not in test_gather:
        if np.sum(img == 0) < 125000 and np.sum(img == 2) < 125000:  # 4307
            class_counter[0] += np.sum(img == 0)
            class_counter[1] += np.sum(img == 1)
            class_counter[2] += np.sum(img == 2)

            label_pic_number += 1
            label_pic_name.append('512_image_625/crop_files/512_label/' + seg.split('/')[-1])
    else:
        test_label_pic_name.append('512_image_625/crop_files/512_label/' + seg.split('/')[-1])
        sliding_test_label_pic_name.append('paris_origin/paris' + origin_pic_number + '_labels_gray.png')
        continue

print(label_pic_number)
percent = class_counter / sum(class_counter)
print(percent)
print((1 / percent) / np.mean(1 / percent))

# """write picture path to txt, contain image/label path"""
# with open('paris/paris_train_list.txt', 'w') as f:
#     for index, name in enumerate(label_pic_name):
#         f.write('{}\t{}\n'.format(name.replace('512_label', '512_image'), name))
#
# with open('paris/paris_test_list.txt', 'w') as f:
#     for index, name in enumerate(test_label_pic_name):
#         f.write('{}\t{}\n'.format(name.replace('512_label', '512_image'), name))
#
# with open('paris/paris_sliding_test_list.txt', 'w') as f:
#     for index, name in enumerate(list(set(sliding_test_label_pic_name))):
#         f.write('{}\t{}\n'.format(name.replace('_labels_gray', '_image'), name))
