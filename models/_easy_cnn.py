# -*- coding: utf-8 -*-
"""
@Time ： 2021/7/6 15:42
@Auth ： JsHou
@Email : 137073284@qq.com
@File ：_easy_cnn.py

"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torchsummary import summary

from load_data import all_dataset


class EasyCNN(pl.LightningModule):
    def __init__(self, n_classes=3, learning_rate: float = 0.0002, batch_size: int = 256, **kwargs):
        super().__init__(**kwargs)
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.train_dataset, self.val_dataset, self.test_dataset = all_dataset(img_size=(100, 100))

        # build classification model
        self.net = nn.Sequential(nn.Conv2d(3, 64, kernel_size=5, padding=2), nn.MaxPool2d(2), nn.ReLU(),
                                 nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.MaxPool2d(2), nn.ReLU(), nn.Flatten(),
                                 nn.Linear(128 * 25 * 25, 120), nn.ReLU(), nn.Linear(120, n_classes), nn.Softmax())

    def forward(self, x):
        return self.net(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)

        labels_hat = torch.argmax(y_hat, dim=1)
        acc = torch.sum(y == labels_hat).item() / (len(y) * 1.0)

        self.log_dict({'train_loss': loss, 'train_acc': acc})
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        labels_hat = torch.argmax(y_hat, dim=1)
        acc = torch.sum(y == labels_hat).item() / (len(y) * 1.0)
        self.log_dict({'val_loss': loss, 'val_acc': acc})

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        labels_hat = torch.argmax(y_hat, dim=1)
        acc = torch.sum(y == labels_hat).item() / (len(y) * 1.0)
        self.log_dict({'test_loss': loss, 'test_acc': acc})
        return loss

    def train_dataloader(self):
        batch_size = self.batch_size
        train_data = self.train_dataset
        train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
        return train_dataloader

    def val_dataloader(self):
        batch_size = self.batch_size
        val_dataset = self.val_dataset
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        return val_dataloader

    def test_dataloader(self):
        batch_size = self.batch_size
        test_dataset = self.test_dataset
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        return test_dataloader


if __name__ == '__main__':
    model = EasyCNN(n_classes=3)
    print(summary(model, (3, 100, 100)))
