# _*_ coding: utf-8 _*_
"""
Time:     2021/1/21 下午3:05
Author:   Ding Cheng(Deeachain)
File:     data_argument.py
Github:   https://github.com/Deeachain
"""
from glob import glob
import cv2
import random
from tqdm import tqdm


image_list = glob('/media/ding/Storage/competition/aili/fenge/20210114/tianchi/train/*.tif')
print(len(image_list))
for img in tqdm(image_list):
    lab = img.replace('.tif', '.png')
    image = cv2.imread(img, cv2.IMREAD_COLOR)
    # image1 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    label = cv2.imread(lab, cv2.IMREAD_GRAYSCALE)

    # 水平垂直翻转
    # img1 = cv2.flip(img, 1)  # 水平翻转
    # img2 = cv2.flip(img, 0)  # 垂直翻转
    image1 = cv2.flip(image, -1)  # 水平垂直翻转
    label1 = cv2.flip(label, -1)  # 水平垂直翻转

    # 对比度亮度
    contrast = 1  # 对比度
    brightness = 100  # 亮度
    image2 = cv2.addWeighted(image, contrast, image, 0, brightness)
    image3 = cv2.addWeighted(image, 1.5, image, 0, 50)

    # 高斯模糊、均值滤波
    image4 = cv2.GaussianBlur(image, (9, 9), 1.5)  # 高斯模糊
    image5 = cv2.blur(image, (11, 11), (-1, -1))  # 均值滤波

    # 随机噪点
    for i in range(1500):
        image[random.randint(0, image.shape[0] - 1)][random.randint(0, image.shape[1] - 1)][:] = 255
        image6 = image

    cv2.imwrite('/media/ding/Storage/competition/aili/fenge/20210114/tianchi/image_argument/' + img.split('/')[-1].split('.')[0] \
                + '_1.tif', image1)
    cv2.imwrite('/media/ding/Storage/competition/aili/fenge/20210114/tianchi/image_argument/' + lab.split('/')[-1].split('.')[0] \
                + '_1.png', label1)
    cv2.imwrite('/media/ding/Storage/competition/aili/fenge/20210114/tianchi/image_argument/' + img.split('/')[-1].split('.')[0] \
                + '_2.tif', image2)
    cv2.imwrite('/media/ding/Storage/competition/aili/fenge/20210114/tianchi/image_argument/' + lab.split('/')[-1].split('.')[0] \
                + '_2.png', label1)
    cv2.imwrite('/media/ding/Storage/competition/aili/fenge/20210114/tianchi/image_argument/' + img.split('/')[-1].split('.')[0] \
                + '_3.tif', image3)
    cv2.imwrite('/media/ding/Storage/competition/aili/fenge/20210114/tianchi/image_argument/' + lab.split('/')[-1].split('.')[0] \
                + '_3.png', label1)
    cv2.imwrite('/media/ding/Storage/competition/aili/fenge/20210114/tianchi/image_argument/' + img.split('/')[-1].split('.')[0] \
                + '_4.tif', image4)
    cv2.imwrite('/media/ding/Storage/competition/aili/fenge/20210114/tianchi/image_argument/' + lab.split('/')[-1].split('.')[0] \
                + '_4.png', label1)
    cv2.imwrite('/media/ding/Storage/competition/aili/fenge/20210114/tianchi/image_argument/' + img.split('/')[-1].split('.')[0] \
                + '_5.tif', image5)
    cv2.imwrite('/media/ding/Storage/competition/aili/fenge/20210114/tianchi/image_argument/' + lab.split('/')[-1].split('.')[0] \
                + '_5.png', label1)
    cv2.imwrite('/media/ding/Storage/competition/aili/fenge/20210114/tianchi/image_argument/' + img.split('/')[-1].split('.')[0] \
                + '_6.tif', image6)
    cv2.imwrite('/media/ding/Storage/competition/aili/fenge/20210114/tianchi/image_argument/' + lab.split('/')[-1].split('.')[0] \
                + '_6.png', label1)