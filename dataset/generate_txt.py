import glob
# def generate_txt(mode = 'train'):
#     filename_list = glob.glob('/media/ding/Data/datasets/paris/512_image_107/overlap/{}/512_image/*.png'.format(mode))
#     filename_list.sort()
#     # print(filename_list)
#     if mode == 'train':
#         with open('./paris/paris_{}_list.txt'.format(mode), 'w+') as f:
#             for filename in filename_list[:]:
#                 filename_gt = filename.replace('512_image', '512_label')
#                 # print(filename.split('/')[-2]+'/'+train.split('/')[-1])
#                 # print(filename_gt)
#                 f.write('{}/{}\t{}/{}\n'.format(mode, filename.split('/')[-2]+'/'+filename.split('/')[-1], mode, filename_gt.split('/')[-2]+'/'+filename_gt.split('/')[-1]))
#
#     elif mode == 'val':
#         with open('./paris/paris_{}_list.txt'.format(mode), 'w+') as f:
#             for filename in filename_list[:]:
#                 filename_gt = filename.replace('512_image', '512_label')
#                 # print(filename.split('/')[-2]+'/'+train.split('/')[-1])
#                 # print(filename_gt)
#                 f.write('{}/{}\t{}/{}\n'.format(mode, filename.split('/')[-2]+'/'+filename.split('/')[-1], mode, filename_gt.split('/')[-2]+'/'+filename_gt.split('/')[-1]))
#
#     else:
#         with open('./paris/paris_{}_list.txt'.format(mode), 'w+') as f:
#             for filename in filename_list:
#                 filename_gt = filename.replace('512_image', '512_label')
#                 # print(filename.split('/')[-2]+'/'+train.split('/')[-1])
#                 # print(filename_gt)
#                 f.write('{}/{}\t{}/{}\n'.format(mode, filename.split('/')[-2]+'/'+filename.split('/')[-1], mode, filename_gt.split('/')[-2]+'/'+filename_gt.split('/')[-1]))

##austin分割数据集
# def generate_txt(mode = 'train'):
#     filename_list = glob.glob('/media/ding/Data/datasets/austin/{}/512_image/*.png'.format(mode))
#     filename_list.sort()
#     # print(filename_list)
#     if mode == 'train':
#         with open('./austin/austin_{}_list.txt'.format(mode), 'w+') as f:
#             for filename in filename_list[:10000]:
#                 filename_gt = filename.replace('512_image', '512_label')
#                 # print('{}/{}\t{}/{}\n'.format(mode, filename.split('/')[-2]+'/'+filename.split('/')[-1], mode, filename_gt.split('/')[-2]+'/'+filename_gt.split('/')[-1]))
#                 # print(filename.split('/')[-2]+'/'+train.split('/')[-1])
#                 # print(filename_gt)
#                 # print(filename_gt.split('/')[-2]+'/'+filename_gt.split('/')[-1])
#                 f.write('{}/{}\t{}/{}\n'.format(mode, filename.split('/')[-2]+'/'+filename.split('/')[-1], mode, filename_gt.split('/')[-2]+'/'+filename_gt.split('/')[-1]))
#
#     elif mode == 'val':
#         with open('./austin/austin_{}_list.txt'.format(mode), 'w+') as f:
#             for filename in filename_list[:]:
#                 filename_gt = filename.replace('512_image', '512_label')
#                 # print(filename.split('/')[-2]+'/'+train.split('/')[-1])
#                 # print(filename_gt)
#                 f.write('{}/{}\t{}/{}\n'.format(mode, filename.split('/')[-2]+'/'+filename.split('/')[-1], mode, filename_gt.split('/')[-2]+'/'+filename_gt.split('/')[-1]))
#
#     else:
#         with open('./austin/austin_{}_list.txt'.format(mode), 'w+') as f:
#             for filename in filename_list:
#                 filename_gt = filename.replace('512_image', '512_label')
#                 # print(filename.split('/')[-2]+'/'+train.split('/')[-1])
#                 # print(filename_gt)
#                 f.write('{}/{}\t{}/{}\n'.format(mode, filename.split('/')[-2]+'/'+filename.split('/')[-1], mode, filename_gt.split('/')[-2]+'/'+filename_gt.split('/')[-1]))


# road数据集
# def generate_txt(mode = 'train'):
#     filename_list = glob.glob('/media/ding/Data/datasets/road/{}/512_image/*.png'.format(mode))
#     filename_list.sort()
#     # print(filename_list)
#     if mode == 'train':
#         with open('./road/road_{}_list.txt'.format(mode), 'w+') as f:
#             for filename in filename_list[:8000]:
#                 filename_gt = filename.replace('512_image', '512_label')
#                 # print(filename.split('/')[-2]+'/'+train.split('/')[-1])
#                 # print(filename_gt)
#                 f.write('{}/{}\t{}/{}\n'.format(mode, filename.split('/')[-2]+'/'+filename.split('/')[-1], mode, filename_gt.split('/')[-2]+'/'+filename_gt.split('/')[-1]))
#
#     elif mode == 'val':
#         with open('./road/road_{}_list.txt'.format(mode), 'w+') as f:
#             for filename in filename_list[:]:
#                 filename_gt = filename.replace('512_image', '512_label')
#                 # print(filename.split('/')[-2]+'/'+train.split('/')[-1])
#                 # print(filename_gt)
#                 f.write('{}/{}\t{}/{}\n'.format(mode, filename.split('/')[-2]+'/'+filename.split('/')[-1], mode, filename_gt.split('/')[-2]+'/'+filename_gt.split('/')[-1]))
#
#     else:
#         with open('./road/road_{}_list.txt'.format(mode), 'w+') as f:
#             for filename in filename_list:
#                 filename_gt = filename.replace('512_image', '512_label')
#                 # print(filename.split('/')[-2]+'/'+train.split('/')[-1])
#                 # print(filename_gt)
#                 f.write('{}/{}\t{}/{}\n'.format(mode, filename.split('/')[-2]+'/'+filename.split('/')[-1], mode, filename_gt.split('/')[-2]+'/'+filename_gt.split('/')[-1]))


# def generate_txt(mode = 'train'):
#     filename_list = glob.glob('/media/ding/Data/datasets/lake/origin/{}/*.png'.format(mode))
#     filename_list.sort()
#     # print(filename_list)
#     if mode == 'train':
#         with open('./lake/lake_{}_list.txt'.format(mode), 'w+') as f:
#             for filename in filename_list[:]:
#                 filename_gt = filename.replace('origin', 'label')
#                 # print(filename.split('/')[-2]+'/'+train.split('/')[-1])
#                 # print(filename_gt)
#                 f.write('{}/{}/{}\t{}/{}/{}\n'.format(filename.split('/')[-3], filename.split('/')[-2], filename.split('/')[-1], filename_gt.split('/')[-3], filename_gt.split('/')[-2], filename_gt.split('/')[-1].split('.')[0] + '_gray.png'))
#
#     elif mode == 'val':
#         with open('./lake/lake_{}_list.txt'.format(mode), 'w+') as f:
#             for filename in filename_list[:]:
#                 filename_gt = filename.replace('origin', 'label')
#                 # print(filename.split('/')[-2]+'/'+train.split('/')[-1])
#                 # print(filename_gt)
#                 f.write('{}/{}/{}\t{}/{}/{}\n'.format(filename.split('/')[-3], filename.split('/')[-2], filename.split('/')[-1], filename_gt.split('/')[-3], filename_gt.split('/')[-2], filename_gt.split('/')[-1].split('.')[0] + '_gray.png'))
#     else:
#         with open('./lake/lake_{}_list.txt'.format(mode), 'w+') as f:
#             for filename in filename_list[:]:
#                 filename_gt = filename.replace('origin', 'label')
#                 # print(filename.split('/')[-2]+'/'+train.split('/')[-1])
#                 # print(filename_gt)
#                 f.write('{}/{}/{}\t{}/{}/{}\n'.format(filename.split('/')[-3], filename.split('/')[-2], filename.split('/')[-1], filename_gt.split('/')[-3], filename_gt.split('/')[-2], filename_gt.split('/')[-1].split('.')[0] + '_gray.png'))

# camvid 6类数据集
def generate_txt(mode = 'train'):
    filename_list = glob.glob('/media/ding/Data/datasets/camvid/{}/*.png'.format(mode))
    filename_list.sort()
    # print(filename_list)
    if mode == 'train':
        with open('../camvid/camvid_{}_list.txt'.format(mode), 'w+') as f:
            for filename in filename_list[:]:
                filename_gt = filename.replace('train', 'trainannot')
                f.write('{}/{}\t{}/{}\n'.format(filename.split('/')[-2], filename.split('/')[-1], filename_gt.split('/')[-2], filename_gt.split('/')[-1]))

    elif mode == 'val':
        with open('../camvid/camvid_{}_list.txt'.format(mode), 'w+') as f:
            for filename in filename_list[:]:
                filename_gt = filename.replace('val', 'valannot')

                f.write('{}/{}\t{}/{}\n'.format(filename.split('/')[-2], filename.split('/')[-1],
                                                   filename_gt.split('/')[-2], filename_gt.split('/')[-1]))
    else:
        with open('../camvid/camvid_{}_list.txt'.format(mode), 'w+') as f:
            for filename in filename_list[:]:
                filename_gt = filename.replace('test', 'testannot')

                f.write('{}/{}\t{}/{}\n'.format(filename.split('/')[-2], filename.split('/')[-1],
                                                   filename_gt.split('/')[-2], filename_gt.split('/')[-1]))
if __name__ == '__main__':
    generate_txt('train')
    generate_txt('val')
    generate_txt('test')
    print('Finsh!')
