from PIL import Image
from tqdm import tqdm
import glob

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

# camvid
image_dir = "/media/ding/Data/datasets/camvid/camvid/labels/*.png"
image_list = glob.glob(image_dir)

for cnt, pic in tqdm(enumerate(image_list)):

    img = Image.open(pic).convert("L")
    width = img.size[0]  # 长度
    height = img.size[1]  # 宽度
    for i in range(0, width):  # 遍历所有长度的点
        for j in range(0, height):  # 遍历所有宽度的点
            data = (img.getpixel((i, j)))  # 打印该图片的所有点
            # print(data)  # 打印每个像素点的颜色RGBA的值(r,g,b,alpha)

            # paris分割数据集
            # if(data == 255):
            #     img.putpixel((i, j), 0)
            # if (data == 29):
            #     img.putpixel((i, j), 1)
            # if (data == 76):
            #     img.putpixel((i, j), 2)

            # austin房屋分割数据集
            # if(data == 0):
            #     img.putpixel((i, j), 0)
            # if(data == 255):
            #     img.putpixel((i, j), 1)

            # roaD分割数据集
            # if (data == 255):
            #     img.putpixel((i, j), 1)

            # camvid数据集6类
            if (data == 4):
                img.putpixel((i, j), 0)
            elif (data == 17):
                img.putpixel((i, j), 1)
            elif (data == 19):
                img.putpixel((i, j), 2)
            elif (data == 21):
                img.putpixel((i, j), 3)
            elif (data == 26):
                img.putpixel((i, j), 4)
            else:
                img.putpixel((i, j), 5)

    # img.save("/media/ding/Data/datasets/pairs/paris/" + image_list[cnt].split('/')[-1].split('.')[0] + '_gray.png')#保存修改像素点后的图片
    # img.save("/media/ding/Data/datasets/遥感数据集/遥感房屋分割数据集/AerialImageDataset/train/gt/" + image_list[cnt].split('/')[-1].split('.')[0] + '_gray.png')#保存修改像素点后的图片

    # img.save("/media/ding/Data/datasets/massachusetts-roads-dataset/road_segmentation_ideal/training/output/" + image_list[cnt].split('/')[-1].split('.')[0] + '_gray.png') #保存修改像素点后的图片 training
    # img.save("/media/ding/Data/datasets/massachusetts-roads-dataset/road_segmentation_ideal/testing/output/" + image_list[cnt].split('/')[-1].split('.')[0] + '_gray.png')
    img.save("/media/ding/Data/datasets/camvid/camvid/labels-6/" + image_list[cnt].split('/')[-1].replace('_P', ''))
