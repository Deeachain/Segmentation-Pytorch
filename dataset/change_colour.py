from PIL import Image
from tqdm import tqdm
import glob
import cv2
import numpy as np

i = 1
j = 1

# 获取指定目录下的所有图片
# ##paris
# image_dir = "/media/ding/Data/datasets/pairs/paris/*labels.png"
# image_list = glob.glob(image_dir)
# print(len(image_list))
# print(image_list[0].split('/')[-1].split('.')[0] + '_gray.png')

# ##austin
# image_dir = "/media/ding/Data/datasets/遥感数据集/遥感房屋分割数据集/AerialImageDataset/train/gt/*.tif"
# image_list = glob.glob(image_dir)

# ##road
# image_dir = "/media/ding/Data/datasets/massachusetts-roads-dataset/road_segmentation_ideal/training/output/*.png"  # train
# image_dir = "/media/ding/Data/datasets/massachusetts-roads-dataset/road_segmentation_ideal/testing/output/*.png"

# # ##lake
# image_dir = "/media/ding/Data/datasets/lake/label/val/*.png"
# image_list = glob.glob(image_dir)

# # camvid
# image_dir = "/media/ding/Data/datasets/camvid/camvid/labels/*.png"
# image_list = glob.glob(image_dir)

# # GID
# image_dir = '/media/ding/Data/datasets/GID/GID_label masks/*.tif'
# image_list = glob.glob(image_dir)

# LSPRS
image_dir = '/media/ding/Storage/competition/aili/fenge/20210114/suichang_round1_train_210120/*.png'
image_list = glob.glob(image_dir)

for cnt, pic in tqdm(enumerate(image_list)):

    # img = Image.open(pic).convert("L")
    # # img = Image.open(pic).convert('P')
    # width = img.size[0]  # 长度
    # height = img.size[1]  # 宽度
    # for i in range(0, width):  # 遍历所有长度的点
    #     for j in range(0, height):  # 遍历所有宽度的点
    #         data = (img.getpixel((i, j)))  # 打印该图片的所有点
    #         print(data)  # 打印每个像素点的颜色RGBA的值(r,g,b,alpha)
    #
    #         # paris分割数据集
    #         # if(data == 255):
    #         #     img.putpixel((i, j), 0)
    #         # if (data == 29):
    #         #     img.putpixel((i, j), 1)
    #         # if (data == 76):
    #         #     img.putpixel((i, j), 2)
    #
    #         # austin房屋分割数据集
    #         # if(data == 0):
    #         #     img.putpixel((i, j), 0)
    #         # if(data == 255):
    #         #     img.putpixel((i, j), 1)
    #
    #         # road分割数据集
    #         # if (data == 255):
    #         #     img.putpixel((i, j), 1)
    #
    #         # camvid数据集6类
    #         # if (data == 4):
    #         #     img.putpixel((i, j), 0)
    #         # elif (data == 17):
    #         #     img.putpixel((i, j), 1)
    #         # elif (data == 19):
    #         #     img.putpixel((i, j), 2)
    #         # elif (data == 21):
    #         #     img.putpixel((i, j), 3)
    #         # elif (data == 26):
    #         #     img.putpixel((i, j), 4)
    #         # else:
    #         #     img.putpixel((i, j), 5)
    #
    #
    #         # # GID数据集17类
    #         # # 0  15  16  22  28  34  40  52  88  94 100 136 184 190 214 220 225
    #         # if (data == 15):
    #         #     img.putpixel((i, j), 1)
    #         # elif (data == 16):
    #         #     img.putpixel((i, j), 2)
    #         # elif (data == 22):
    #         #     img.putpixel((i, j), 3)
    #         # elif (data == 28):
    #         #     img.putpixel((i, j), 4)
    #         # elif (data == 34):
    #         #     img.putpixel((i, j), 5)
    #         # elif (data == 40):
    #         #     img.putpixel((i, j), 6)
    #         # elif (data == 52):
    #         #     img.putpixel((i, j), 7)
    #         # elif (data == 88):
    #         #     img.putpixel((i, j), 8)
    #         # elif (data == 94):
    #         #     img.putpixel((i, j), 9)
    #         # elif (data == 100):
    #         #     img.putpixel((i, j), 10)
    #         # elif (data == 136):
    #         #     img.putpixel((i, j), 11)
    #         # elif (data == 184):
    #         #     img.putpixel((i, j), 12)
    #         # elif (data == 190):
    #         #     img.putpixel((i, j), 13)
    #         # elif (data == 214):
    #         #     img.putpixel((i, j), 14)
    #         # elif (data == 220):
    #         #     img.putpixel((i, j), 15)
    #         # elif (data == 225):
    #         #     img.putpixel((i, j), 16)
    #
    #
    #         # lsprs数据集6类Potsdam
    #         # 29  76 150 179 226 255
    #         if (data == 29):
    #             img.putpixel((i, j), 0)
    #         elif (data == 76):
    #             img.putpixel((i, j), 1)
    #         elif (data == 150):
    #             img.putpixel((i, j), 2)
    #         elif (data == 179):
    #             img.putpixel((i, j), 3)
    #         elif (data == 226):
    #             img.putpixel((i, j), 4)
    #         elif (data == 255):
    #             img.putpixel((i, j), 5)
    #
    #
    #     # img.save("/media/ding/Data/datasets/pairs/paris/" + image_list[cnt].split('/')[-1].split('.')[0] + '_gray.png')#保存修改像素点后的图片
    #     # img.save("/media/ding/Data/datasets/遥感数据集/遥感房屋分割数据集/AerialImageDataset/train/gt/" + image_list[cnt].split('/')[-1].split('.')[0] + '_gray.png')#保存修改像素点后的图片
    #
    #     # img.save("/media/ding/Data/datasets/massachusetts-roads-dataset/road_segmentation_ideal/training/output/" + image_list[cnt].split('/')[-1].split('.')[0] + '_gray.png') #保存修改像素点后的图片 training
    #     # img.save("/media/ding/Data/datasets/massachusetts-roads-dataset/road_segmentation_ideal/testing/output/" + image_list[cnt].split('/')[-1].split('.')[0] + '_gray.png')
    #     # img.save("/media/ding/Data/datasets/camvid/camvid/labels-6/" + image_list[cnt].split('/')[-1].replace('_P', ''))
    #     img.save(
    #         '/media/ding/Data/datasets/ISPRS_BENCHMARK_DATASETS/Potsdam/5_Labels_all/' + image_list[cnt].split('/')[
    #             -1].replace('_label', '_label_L'))


    # label = cv2.imread(pic, cv2.IMREAD_GRAYSCALE)
    #
    # # 29  76 150 179 226 255
    # label[np.where(label == 255)] = 0
    # label[np.where(label == 76)] = 1#76 29
    # label[np.where(label == 226)] = 2#226 179
    # label[np.where(label == 150)] = 3#
    # label[np.where(label == 179)] = 4#179 226
    # label[np.where(label == 29)] = 5#29 76
    # label = np.asarray(label, np.uint8)
    # """
    # 0,Impervious_surfaces,255 255 255,255
    # 1,Building,0 0 255,29
    # 2,Low_vegetation,0 255 255,179
    # 3,Tree,0 255 0,150
    # 4,Car,255 255 0,226
    # 5,Clutter_background,255 0 0,76

    label = cv2.imread(pic)
    label[np.where(label == 1)] = 0
    label[np.where(label == 2)] = 1
    label[np.where(label == 3)] = 2
    label[np.where(label == 4)] = 3
    label[np.where(label == 5)] = 4
    label[np.where(label == 6)] = 5
    label[np.where(label == 7)] = 6
    label[np.where(label == 8)] = 7
    label[np.where(label == 9)] = 8
    label[np.where(label == 10)] = 9
    label = np.asarray(label, np.uint8)
    """
    0,Impervious_surfaces,255 255 255,255
    1,Building,0 0 255,29
    2,Low_vegetation,0 255 255,179
    3,Tree,0 255 0,150
    4,Car,255 255 0,226
    5,Clutter_background,255 0 0,76
    
    """
    print(np.unique(label))
    # cv2.imwrite(
    #     '/media/ding/Data/datasets/ISPRS_BENCHMARK_DATASETS/Potsdam/5_Labels_all/' + image_list[cnt].split('/')[
    #         -1].replace('_label', '_label_L'), label)


# import os
# import cv2
# import numpy as np
# from glob import glob
# from tqdm import tqdm
#
# """
# input:单通道图片，每个像素点是用一个三位数表示
# output:将三位数的像素表达改变成一位数
# """
# root_dir = 'train/labels'  # 存储图片的root目录
# pic_ext = '*.png'  # 需要处理的图片后缀
# pic_dir = os.path.join(root_dir, pic_ext)
#
# pic_list = glob(pic_dir)  # 获取所有图片的路径，存储到列表中
#
# for seg in tqdm(pic_list, desc='Processing'):
#     img = cv2.imread(seg, cv2.IMREAD_UNCHANGED)
#     print(img)
#
#     img[np.where(img == 800)] = 7  # 将7赋值给800
#     img[np.where(img == 700)] = 6  # 将6赋值给700
#     img[np.where(img == 600)] = 5  # 将5赋值给600
#     img[np.where(img == 500)] = 4  # 将4赋值给500
#     img[np.where(img == 400)] = 3  # 将3赋值给400
#     img[np.where(img == 300)] = 2  # 将2赋值给300
#     img[np.where(img == 200)] = 1  # 将1赋值给200
#     img[np.where(img == 100)] = 0  # 将0赋值给100
#
#     print(img)
#     cv2.imwrite(seg.split('.')[0] + 'changed.png', img)
