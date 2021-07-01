# -*- coding: utf-8 -*-
"""
@Time ： 2021/7/1 6:22
@Auth ： JsHou
@Email : 137073284@qq.com
@File ：test.py

"""

import os
import numpy as np
import torch
import pandas as pd

from models import ResNet50
from data import Pest
from utils import get_root_path


def get_best_model_path():
    root_path = get_root_path()
    with open(os.path.join(root_path, 'checkpoints', 'resnet50', 'info.txt'), 'r') as f:
        ckpt_best_path = f.readline()
    return ckpt_best_path


def classify(x, net, transform=None):
    # 处理数据
    if len(x.shape) == 3:
        x = x[np.newaxis, :, :, :]
    x = torch.from_numpy(x)
    if transform:
        x = transform(x)

    # 加载模型
    y = net(x.float())
    return y.detach().numpy()


def write_to_csv(names, labels):
    predict_data = np.vstack((names, labels)).T
    pd_data = pd.DataFrame(predict_data)
    pd_data.columns = ['image', 'label']
    pd_data.to_csv(r'./results/test.csv', index=False)


if __name__ == '__main__':
    ckpt_best_path = get_best_model_path()
    model = ResNet50.load_from_checkpoint(ckpt_best_path)
    names, imgs = Pest.testdata()
    y = classify(imgs, model)
    labels = np.argmax(y, axis=1)
    write_to_csv(names, labels)
