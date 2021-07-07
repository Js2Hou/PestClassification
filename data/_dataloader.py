# -*- coding: utf-8 -*-
"""
@Time ： 2021/6/29 21:41
@Auth ： JsHou
@Email : 137073284@qq.com
@File ：_dataloader.py

"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from torchvision import transforms
from torchvision.datasets import ImageFolder


class Pest:
    # root_path = os.path.abspath(os.path.dirname(__file__)).split('PestClassification')[0]
    root_path = r'/chen/dataset/Pestsvegetables'
    train_path = os.path.join(root_path, 'train')
    test_path = os.path.join(root_path, 'test')

    def __init__(self, img_size):
        self.img_size = img_size

    def dataset(self):
        """
        Returns all training data in form of torch.utils.data.data.Dataset
        """
        transform = transforms.Compose([transforms.Resize(self.img_size), transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        dataset_ = ImageFolder(self.train_path, transform=transform)
        return dataset_

    def testdata(self):
        """return testing data"""
        names = os.listdir(Pest.test_path)
        imgs = []
        for name_ in names:
            img = cv2.imread(os.path.join(Pest.test_path, name_))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, self.img_size)
            img = np.transpose(img, (2, 0, 1))
            imgs.append(img)
        return np.array(names), np.array(imgs)


if __name__ == '__main__':
    p = Pest(img_size=(100, 100))
    d = p.dataset()
