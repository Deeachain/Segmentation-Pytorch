# _*_ coding: utf-8 _*_
"""
Time:     2020/11/30 17:02
Author:   Ding Cheng(Deeachain)
File:     dataset_builder.py
Describe: Write during my study in Nanjing University of Information and Secience Technology
Github:   https://github.com/Deeachain
"""
import os
import pickle
import pandas as pd
from dataset.cityscapes.cityscapes import CityscapesTrainDataSet, CityscapesTrainInform, CityscapesValDataSet, \
    CityscapesTestDataSet

def build_dataset_train(root, dataset, base_size, crop_size):
    data_dir = os.path.join(root, dataset)
    train_data_list = os.path.join(data_dir, dataset + '_' + 'train_list.txt')
    inform_data_file = os.path.join('./dataset/inform/', dataset + '_inform.pkl')

    # inform_data_file collect the information of mean, std and weigth_class
    if not os.path.isfile(inform_data_file):
        print("%s is not found" % (inform_data_file))
        if dataset == "cityscapes":
            dataCollect = CityscapesTrainInform(data_dir, 19, train_set_file=train_data_list,
                                                inform_data_file=inform_data_file)
        else:
            raise NotImplementedError(
                "This repository now supports two datasets: cityscapes and camvid, %s is not included" % dataset)

        datas = dataCollect.collectDataAndSave()
        if datas is None:
            print("error while pickling data. Please check.")
            exit(-1)
    else:
        datas = pickle.load(open(inform_data_file, "rb"))

    if dataset == "cityscapes":
        TrainDataSet = CityscapesTrainDataSet(data_dir, train_data_list, base_size=base_size, crop_size=crop_size,
                                        mean=datas['mean'], std=datas['std'], ignore_label=255)
        return datas, TrainDataSet


def build_dataset_test(root, dataset, crop_size, mode='whole', gt=False):
    data_dir = os.path.join(root, dataset)
    inform_data_file = os.path.join('./dataset/inform/', dataset + '_inform.pkl')
    train_data_list = os.path.join(data_dir, dataset + '_train_list.txt')
    if mode == 'whole':
        test_data_list = os.path.join(data_dir, dataset + '_test' + '_list.txt')
    else:
        test_data_list = os.path.join(data_dir, dataset + '_test_sliding' + '_list.txt')

    # inform_data_file collect the information of mean, std and weigth_class
    if not os.path.isfile(inform_data_file):
        print("%s is not found" % (inform_data_file))
        if dataset == "cityscapes":
            dataCollect = CityscapesTrainInform(data_dir, 19, train_set_file=train_data_list,
                                                inform_data_file=inform_data_file)
        else:
            raise NotImplementedError(
                "This repository now supports two datasets: cityscapes and camvid, %s is not included" % dataset)

        datas = dataCollect.collectDataAndSave()
        if datas is None:
            print("error while pickling data. Please check.")
            exit(-1)
    else:
        datas = pickle.load(open(inform_data_file, "rb"))

    class_dict_df = pd.read_csv(os.path.join('./dataset', dataset, 'class_map.csv'))
    if dataset == "cityscapes":
        # for cityscapes, if test on validation set, set none_gt to False
        # if test on the test set, set none_gt to True
        if gt:
            test_data_list = os.path.join(data_dir, dataset + '_val' + '_list.txt')
            testdataset = CityscapesValDataSet(data_dir, test_data_list, crop_size=crop_size, mean=datas['mean'],
                                     std=datas['std'], ignore_label=255)
        else:
            test_data_list = os.path.join(data_dir, dataset + '_test' + '_list.txt')
            testdataset = CityscapesTestDataSet(data_dir, test_data_list, crop_size=crop_size, mean=datas['mean'],
                                      std=datas['std'], ignore_label=255)
        return testdataset, class_dict_df
