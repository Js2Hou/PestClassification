# -*- coding: utf-8 -*-
"""
@Time ： 2021/7/1 6:22
@Auth ： JsHou
@Email : 137073284@qq.com
@File ：classify.py

"""

import os
import numpy as np
import torch
import pandas as pd

from models import ResNet50, EasyCNN
from data import Pest
from utils import get_root_path


def get_best_model_path(Model):
    root_path = get_root_path()
    with open(os.path.join(root_path, 'checkpoints', str(Model.__name__).lower(), 'info.txt'), 'r') as f:
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


def write_to_csv(names, labels, Model):
    predict_data = np.vstack((names, labels)).T
    pd_data = pd.DataFrame(predict_data)
    pd_data.columns = ['image', 'label']
    pd_data.to_csv(f'./results/{str(Model.__name__).lower()}.csv', index=False)


def classify_with_model(Model, img_size):
    ckpt_best_path = get_best_model_path(Model)
    _model = Model.load_from_checkpoint(ckpt_best_path)

    p = Pest(img_size=img_size)
    names, imgs = p.testdata()

    y = classify(imgs, _model)
    labels = np.argmax(y, axis=1)
    if np.min(labels) == 0:
        labels = labels + 1
    write_to_csv(names, labels, Model)
    print('save to .csv successfully!')


if __name__ == '__main__':
    classify_with_model(ResNet50, img_size=(224, 224))  # classify_with_model(EasyCNN, img_size=(100, 100))
