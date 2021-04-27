# _*_ coding: utf-8 _*_
"""
Time:     2020/11/30 17:02
Author:   Ding Cheng(Deeachain)
File:     generate_txt.py
Describe: Write during my study in Nanjing University of Information and Secience Technology
Github:   https://github.com/Deeachain
"""
import glob
import random
import os


def generate_txt(mode='train'):
    root_dir = '/data/open_data/cityscapes/leftImg8bit/'
    dir_list = os.listdir(os.path.join(root_dir, mode))
    filename_list = []
    for dir in dir_list:
        filename = glob.glob(os.path.join(root_dir, mode, dir) + '/'+'*.png')
        filename.sort()

        random.shuffle(filename)
        filename_list.extend(filename)
    with open('cityscapes_{}_list.txt'.format(mode), 'w+') as f:
        for filename in filename_list[:]:
            filename_gt = filename.replace('leftImg8bit', 'gtFine').replace('.png', '_labelTrainIds.png')
            print(filename, filename_gt)
            f.write('{}/{}/{}/{}\t{}/{}/{}/{}\n'.format(filename.split('/')[-4], filename.split('/')[-3],
                                                        filename.split('/')[-2], filename.split('/')[-1],
                                                        filename_gt.split('/')[-4], filename_gt.split('/')[-3],
                                                        filename_gt.split('/')[-2], filename_gt.split('/')[-1]))


if __name__ == '__main__':
    generate_txt('train')
    print('Finsh!')
