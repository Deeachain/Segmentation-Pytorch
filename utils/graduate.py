# _*_ coding: utf-8 _*_
"""
Time:     2021/2/24 下午8:38
Author:   Ding Cheng(Deeachain)
File:     graduate.py
Github:   https://github.com/Deeachain
"""
import cv2
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

img_path = '/media/ding/Data/datasets/paris/512image_unoverlap/train/512_image/paris1_0_0.png'
image = cv2.imread(img_path, cv2.IMREAD_COLOR)
image = torch.from_numpy(image)
image = image.permute(2,1,0).unsqueeze(0).float()

# 步长1卷积
s1_out = nn.Conv2d(3,3,3)(image)
s1 = np.squeeze(s1_out.detach().numpy()).transpose(2,1,0)
s1 = (((s1 - np.min(s1)) / (np.max(s1) - np.min(s1))) * 255).astype(np.uint8)
s1 = cv2.applyColorMap(s1, cv2.COLORMAP_JET)


# 步长2卷积
s2 = nn.Conv2d(3,3,3,2)(image)
s2 = np.squeeze(s2.detach().numpy()).transpose(2,1,0)
s2 = (((s2 - np.min(s2)) / (np.max(s2) - np.min(s2))) * 255).astype(np.uint8)
s2 = cv2.applyColorMap(s2, cv2.COLORMAP_JET)

# 最大池化
m = nn.MaxPool2d(2,2)(s1_out)
m = np.squeeze(m.detach().numpy()).transpose(2,1,0)
m = (((s2 - np.min(m)) / (np.max(m) - np.min(m))) * 255).astype(np.uint8)
m = cv2.applyColorMap(m, cv2.COLORMAP_JET)


# plt.figure(figsize=(8,2.5))
plt.figure(figsize=(4.5,4.5))
font = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 12,}
# ax1 = plt.subplot(1, 3, 1)
plt.imshow(s1, cmap='gray')
plt.xlabel('',font)
plt.savefig('s1.png')
plt.show()

plt.figure(figsize=(4.5,4.5))
# ax2 = plt.subplot(1, 3, 2)
plt.imshow(s2, cmap='gray')
plt.xlabel('',font)
plt.savefig('s2.png')
plt.show()

plt.figure(figsize=(4.5,4.5))
# ax3 = plt.subplot(1, 3, 3)
plt.imshow(m, cmap='gray')
plt.xlabel('',font)
plt.savefig('m.png')
plt.show()

# plt.savefig('s1_s2_m_compare.png')
# plt.show()