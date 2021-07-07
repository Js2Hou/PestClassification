# -*- coding: utf-8 -*-
"""
@Time ： 2021/7/6 14:11
@Auth ： JsHou
@Email : 137073284@qq.com
@File ：augment_data.py
Augment dataset.
Perform dataset augmentation.
Augmented data saved in path 'train/i/output/', where i represents category.
"""

import os
import Augmentor


def augImg(path):
    p = Augmentor.Pipeline(path)
    p.rotate90(probability=0.5)
    p.rotate270(probability=0.5)
    p.flip_left_right(probability=0.8)
    p.flip_top_bottom(probability=0.3)
    p.crop_random(probability=1, percentage_area=0.5)
    p.resize(probability=1.0, width=120, height=120)
    p.sample(2000)


if __name__ == '__main__':
    data_path = '/chen/dataset/PestsVegetables/train'
    # i represents category in training data
    for i in range(1, 4):
        path_ = os.path.join(data_path, str(i))
        augImg(path_)
