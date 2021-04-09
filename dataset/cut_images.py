from PIL import Image
import cv2
import glob
from tqdm import tqdm
import re

'''执行之前确认以下两个dir的路径是否正确'''
input_dir = "/data/dingcheng/ISPRS/postdam/5_Labels_all/"  # 需要裁剪的大图目录
output_dir = "/data/dingcheng/ISPRS/postdam/val"  # 裁剪好的小图目录

image_dir = input_dir + "/*label_L.tif"
image_list = glob.glob(image_dir)
image_list.sort()
num_data = len(image_list)
# 所有图片放在一个文件夹，切分数据集8:2:2，验证集和测试机相同
# image_list_train, image_list_val, image_list_test = image_list[:int(0.8*num_data)], image_list[int(0.8*num_data):], image_list[int(0.8*num_data):]
# print(image_list[0].split('/')[-1].split('.')[0] + '_gray.png')
# print(image_list[:3])
# train_id = ['2_10', '3_10', '3_11', '3_12', '4_11', '5_10', '5_12', '6_8', '6_9', '6_10', '6_11', '6_12',
#             '7_7', '7_9', '7_11', '7_12']
# test_id = ['2_11', '2_12', '4_10', '5_11', '7_8', '7_10']
train_id = ['2_10', '2_11', '2_12', '3_10', '3_11', '3_12', '4_10', '4_11', '5_10', '5_11', '5_12', '6_8',
            '6_9', '6_10', '6_11', '6_12', '7_7', '7_8', '7_9', '7_10', '7_11', '7_12']
test_id = ['2_13', '2_14', '3_13', '3_14', '4_13', '4_14', '4_15', '5_13', '5_14', '5_15', '6_13', '6_14', '6_15',
            '7_13']


def cut_image(image_list=None):
    for image in tqdm(image_list):
        for id in test_id:
            new_str_pat = re.compile(r'{}'.format(id))
            if new_str_pat.findall(image) == []:
                continue
            else:
                image_path = image.replace('label_L', 'RGB').replace('5_Labels_all', '2_Ortho_RGB')
                label_path = image
                # 打开一张图
                img = Image.open(image_path)
                label = Image.open(label_path)
                # 图片尺寸
                size = img.size
                cropsize = 1024
                stride = 1024
                num_h, num_w = 0, 0
                if (size[0] % stride > 0):
                    num_w = int(size[0] / stride) + 1
                    if ((stride * (num_w - 2) + cropsize) > size[0]):
                        num_w = int(size[0] / stride)  # 防止最后一个滑窗溢出，重复计算
                elif (size[0] % stride == 0):
                    num_w = int(size[0] / stride)

                if (size[1] % stride > 0):
                    num_h = int(size[1] / stride) + 1
                    if stride * (num_h - 2) + cropsize > size[1]:
                        num_h = int(size[1] / stride)  # 防止最后一个滑窗溢出，重复计算
                elif (size[1] % stride == 0):
                    num_h = int(size[1] / stride)

                for j in range(num_h):
                    for i in range(num_w):
                        x1 = int(i * stride)  # 起始位置x1 = 0 * 513 = 0   0*400
                        y1 = int(j * stride)  # y1 = 0 * 513 = 0   0*400
                        x2 = min(x1 + cropsize, size[0])  # 末位置x2 = min(0+512, 3328)
                        y2 = min(y1 + cropsize, size[1])  # y2 = min(0+512, 3072)
                        x1 = max(int(x2 - cropsize), 0)  # 重新校准起始位置x1 = max(512-512, 0)
                        y1 = max(int(y2 - cropsize), 0)  # y1 = max(512-512, 0)
                        box = (x1, y1, x2, y2)
                        # # print(box)
                        crop_image = img.crop(box)
                        crop_label = label.crop(box)
                        image_name = image_path.split('/')[-1].split('_RGB')[0]
                        crop_image.save(output_dir + '/{}_image/{}_{}_{}.tif'.format(cropsize, image_name, j, i))
                        crop_label.save(output_dir + '/{}_label/{}_{}_{}.tif'.format(cropsize, image_name, j, i))


if __name__ == '__main__':
    cut_image(image_list)
