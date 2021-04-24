# _*_ coding: utf-8 _*_
"""
Time:     2021/4/22 14:47
Author:   Cheng Ding(Deeachain)
Version:  V 0.1
File:     distributed.py
"""
import os
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils import data


def Distribute(args, traindataset, model, criterion, device, gpus):
    # process_group = torch.distributed.new_group(list(range(gpus)))
    # model = nn.SyncBatchNorm.convert_sync_batchnorm(model, process_group)

    train_sampler = data.distributed.DistributedSampler(traindataset)
    DataLoader = data.DataLoader(traindataset, batch_size=args.batch_size//gpus, sampler=train_sampler,
                shuffle=False, num_workers=args.batch_size, pin_memory=True, drop_last=True)

    model.to(device)
    criterion = criterion.to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                      output_device=args.local_rank,
                                                      find_unused_parameters=True)
    return DataLoader, model, criterion