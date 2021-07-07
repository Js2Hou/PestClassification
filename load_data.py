# -*- coding: utf-8 -*-
"""
@Time ： 2021/6/30 11:15
@Auth ： JsHou
@Email : 137073284@qq.com
@File ：load_data.py

"""

import torch
from torch.utils.data import random_split

from data import Pest


def all_dataset(img_size, ratio=(0.8, 0.1, 0.1)):
    """
    return train_dataset, val_dataset, test_dataset according to ratio
    :param img_size: image size of model input
    :param ratio: ratio of train_data, val_data, test_data
    :return:
    """
    p = Pest(img_size=img_size)
    dataset_ = p.dataset()
    num_samples = len(dataset_)
    n_train = int(num_samples * ratio[0])
    n_val = int(num_samples * ratio[1])
    n_test = num_samples - n_train - n_val
    train_dataset, val_dataset, test_dataset = random_split(dataset_, [n_train, n_val, n_test],
                                                            generator=torch.Generator().manual_seed(42))
    return train_dataset, val_dataset, test_dataset
