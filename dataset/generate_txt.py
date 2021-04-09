import glob
import random

# isprs 6类数据集
def generate_txt(mode = 'train'):
    filename_list = glob.glob('/data/dingcheng/ISPRS/postdam/val/1024_image/*.tif')
    filename_list.sort()
    # print(filename_list)

    random.shuffle(filename_list)

    with open('isprs_{}_list.txt'.format(mode), 'w+') as f:
        for filename in filename_list[:]:
            filename_gt = filename.replace('1024_image', '1024_label')
            # print(filename, filename_gt)
            f.write('{}/{}/{}\t{}/{}/{}\n'.format(mode, filename.split('/')[-2], filename.split('/')[-1],
                                                  mode,filename_gt.split('/')[-2], filename_gt.split('/')[-1]))


if __name__ == '__main__':
    # generate_txt('train')
    generate_txt('test')
    print('Finsh!')