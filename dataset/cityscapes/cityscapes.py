# _*_ coding: utf-8 _*_
"""
Time:     2021/3/12 10:28
Author:   Ding Cheng(Deeachain)
File:     cityscapes.py
Github:   https://github.com/Deeachain
"""
import os.path as osp
import numpy as np
import cv2
from torch.utils import data
import pickle
from PIL import Image
from torchvision import transforms
from utils import image_transform as tr


class CityscapesTrainDataSet(data.Dataset):
    """
       CityscapesTrainDataSet is employed to load train set
       Args:
        root: the Cityscapes dataset path,
         cityscapes
          ├── gtFine
          ├── leftImg8bit
        list_path: cityscapes_train_list.txt, include partial path
        mean: bgr_mean (73.15835921, 82.90891754, 72.39239876)

    """

    def __init__(self, root='', list_path='', max_iters=None, base_size=513, crop_size=513, mean=(128, 128, 128),
                 std=(128, 128, 128), ignore_label=255):
        self.root = root
        self.list_path = list_path
        self.base_size = base_size
        self.crop_size = crop_size
        self.mean = mean
        self.std = std
        self.ignore_label = ignore_label
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        if not max_iters == None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []

        for name in self.img_ids:
            img_file = osp.join(self.root, name.split()[0])
            label_file = osp.join(self.root, name.split()[1])
            name = name.strip().split()[0].strip().split('/', 3)[3].split('.')[0]
            self.files.append({"img": img_file, "label": label_file, "name": name})

        print("length of train dataset: ", len(self.files))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        image = Image.open(datafiles["img"]).convert('RGB')
        label = Image.open(datafiles["label"])
        size = np.asarray(image).shape
        name = datafiles["name"]

        composed_transforms = transforms.Compose([
            tr.RandomHorizontalFlip(),
            # tr.RandomRotate(180),
            tr.RandomScaleCrop(base_size=self.base_size, crop_size=self.crop_size, fill=255),
            tr.RandomGaussianBlur(),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])
        sample = {'image': image, 'label': label}
        sampled = composed_transforms(sample)
        image, label = sampled['image'], sampled['label']
        return image, label, np.array(size), name


class CityscapesValDataSet(data.Dataset):
    """
       CityscapesDataSet is employed to load val set
       Args:
        root: the Cityscapes dataset path,
         cityscapes
          ├── gtFine
          ├── leftImg8bit
        list_path: cityscapes_val_list.txt, include partial path

    """

    def __init__(self, root='', list_path='', crop_size=513, mean=(128, 128, 128), std=(128, 128, 128),
                 ignore_label=255):
        self.root = root
        self.list_path = list_path
        self.crop_size = crop_size
        self.mean = mean
        self.std = std
        self.ignore_label = ignore_label
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        self.files = []
        for name in self.img_ids:
            img_file = osp.join(self.root, name.split()[0])
            label_file = osp.join(self.root, name.split()[1])
            name = name.strip().split()[0].strip().split('/', 3)[3].split('.')[0]
            self.files.append({"img": img_file, "label": label_file, "name": name})

        print("length of validation dataset: ", len(self.files))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        image = Image.open(datafiles["img"]).convert('RGB')
        label = Image.open(datafiles["label"])
        size = np.asarray(image).shape
        name = datafiles["name"]
        composed_transforms = transforms.Compose([
            tr.FixScaleCrop(crop_size=self.crop_size),
            # tr.FixedResize(size=(1024,512)),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])
        sample = {'image': image, 'label': label}
        sampled = composed_transforms(sample)
        image, label = sampled['image'], sampled['label']

        return image, label, np.array(size), name


class CityscapesTestDataSet(data.Dataset):
    """
       CityscapesDataSet is employed to load test set
       Args:
        root: the Cityscapes dataset path,
        list_path: cityscapes_test_list.txt, include partial path

    """

    def __init__(self, root='', list_path='', crop_size=513, mean=(128, 128, 128), std=(128, 128, 128),
                 ignore_label=255):
        self.root = root
        self.list_path = list_path
        self.crop_size = crop_size
        self.mean = mean
        self.std = std
        self.ignore_label = ignore_label
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        self.files = []
        for name in self.img_ids:
            img_file = osp.join(self.root, name.split()[0])
            name = name.strip().split()[0].strip().split('/', 3)[3].split('.')[0]
            self.files.append({"img": img_file, "name": name})

        print("length of validation dataset: ", len(self.files))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        image = Image.open(datafiles["img"]).convert('RGB')
        size = np.asarray(image).shape
        name = datafiles["name"]
        composed_transforms = transforms.Compose([
            # tr.FixedResize(size=(1024,512)),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])
        sample = {'image': image}
        sampled = composed_transforms(sample)
        image = sampled['image']

        return image, np.array(size), name


class CityscapesTrainInform:
    """ To get statistical information about the train set, such as mean, std, class distribution.
        The class is employed for tackle class imbalance.
    """

    def __init__(self, data_dir='', classes=19,
                 train_set_file="", inform_data_file="", normVal=1.10):
        """
        Args:
           data_dir: directory where the dataset is kept
           classes: number of classes in the dataset
           inform_data_file: location where cached file has to be stored
           normVal: normalization value, as defined in ERFNet paper
        """
        self.data_dir = data_dir
        self.classes = classes
        self.classWeights = np.ones(self.classes, dtype=np.float32)
        self.normVal = normVal
        self.mean = np.zeros(3, dtype=np.float32)
        self.std = np.zeros(3, dtype=np.float32)
        self.train_set_file = train_set_file
        self.inform_data_file = inform_data_file

    def compute_class_weights(self, histogram):
        """to compute the class weights
        Args:
            histogram: distribution of class samples
        """
        normHist = histogram / np.sum(histogram)
        for i in range(self.classes):
            self.classWeights[i] = 1 / (np.log(self.normVal + normHist[i]))

    def readWholeTrainSet(self, fileName, train_flag=True):
        """to read the whole train set of current dataset.
        Args:
        fileName: train set file that stores the image locations
        trainStg: if processing training or validation data

        return: 0 if successful
        """
        global_hist = np.zeros(self.classes, dtype=np.float32)

        no_files = 0
        min_val_al = 0
        max_val_al = 0
        with open(self.data_dir + '/' + fileName, 'r') as textFile:
            for line in textFile:
                # we expect the text file to contain the data in following format
                # <RGB Image> <Label Image>
                line_arr = line.split()
                img_file = ((self.data_dir).strip() + '/' + line_arr[0].strip()).strip()
                label_file = ((self.data_dir).strip() + '/' + line_arr[1].strip()).strip()

                label_img = cv2.imread(label_file, 0)
                unique_values = np.unique(label_img)
                max_val = max(unique_values)
                min_val = min(unique_values)

                max_val_al = max(max_val, max_val_al)
                min_val_al = min(min_val, min_val_al)

                if train_flag == True:
                    hist = np.histogram(label_img, self.classes, range=(0, 18))
                    global_hist += hist[0]

                    rgb_img = cv2.imread(img_file)
                    self.mean[0] += np.mean(rgb_img[:, :, 0])
                    self.mean[1] += np.mean(rgb_img[:, :, 1])
                    self.mean[2] += np.mean(rgb_img[:, :, 2])

                    self.std[0] += np.std(rgb_img[:, :, 0])
                    self.std[1] += np.std(rgb_img[:, :, 1])
                    self.std[2] += np.std(rgb_img[:, :, 2])

                else:
                    print("we can only collect statistical information of train set, please check")

                if max_val > (self.classes - 1) or min_val < 0:
                    print('Labels can take value between 0 and number of classes.')
                    print('Some problem with labels. Please check. label_set:', unique_values)
                    print('Label Image ID: ' + label_file)
                no_files += 1

        # divide the mean and std values by the sample space size
        self.mean /= no_files * 255
        self.std /= no_files

        # compute the class imbalance information
        self.compute_class_weights(global_hist)
        return 0

    def collectDataAndSave(self):
        """ To collect statistical information of train set and then save it.
        The file train.txt should be inside the data directory.
        """
        print('Processing training data')
        return_val = self.readWholeTrainSet(fileName=self.train_set_file)

        print('Pickling data')
        if return_val == 0:
            data_dict = dict()
            data_dict['mean'] = self.mean
            data_dict['std'] = self.std
            data_dict['classWeights'] = self.classWeights
            pickle.dump(data_dict, open(self.inform_data_file, "wb"))
            return data_dict
        return None
