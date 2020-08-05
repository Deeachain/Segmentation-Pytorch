from PIL import Image
import glob
from tqdm import tqdm

'''执行之前确认以下两个dir的路径是否正确'''
input_dir = "/media/ding/Data/datasets/paris" # 需要裁剪的label目录
output_dir = "/media/ding/Data/datasets/paris"  # 裁剪好的label目录

cropsize = 324


def crop_overlap(mode):
    image_dir = input_dir + '/' + mode + "/512_label/paris*.png"
    image_list = glob.glob(image_dir)
    image_list.sort()
    num_data = len(image_list)
    # print(image_list[:5])
    # print(num_data)
    pbar = tqdm(iterable=image_list, total=len(image_list))
    for i in pbar:
        img = Image.open(i)
        left = (img.size[1]-cropsize)/2
        up = (img.size[0]-cropsize)/2
        right = img.size[1]-(img.size[1]-cropsize)/2
        down = img.size[0]-(img.size[0]-cropsize)/2
        box = (int(left), int(up), int(right), int(down))
        crop_img = img.crop(box)
        crop_img.save(output_dir + '/' + mode + '/324_label/'+ i.split('/')[-1])

if __name__ == "__main__":
    # crop_overlap('train')
    crop_overlap('val')
    crop_overlap('test')