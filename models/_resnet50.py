# -*- coding: utf-8 -*-
"""
@Time ： 2021/6/30 20:05
@Auth ： JsHou
@Email : 137073284@qq.com
@File ：_resnet50.py

"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision.models import resnet50
from torch.nn import functional as F
from torchsummary import summary

from load_data import all_dataset


class ResNet50(pl.LightningModule):
    def __init__(self, n_classes=3, learning_rate: float = 0.0002, batch_size: int = 64, **kwargs):
        super().__init__(**kwargs)
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.train_dataset, self.val_dataset, self.test_dataset = all_dataset()

        # init a pretrained resnet
        backbone = resnet50(pretrained=True)
        num_filters = backbone.fc.in_features
        layers = list(backbone.children())[:-1]
        self.feature_extractor = nn.Sequential(*layers)

        # use the pretrained model to classify cifar-10 (10 image classes)
        num_target_classes = n_classes
        self.classifier = nn.Sequential(nn.Linear(num_filters, num_target_classes), nn.Softmax(dim=1))

    def forward(self, x):
        self.feature_extractor.eval()
        with torch.no_grad():
            representations = self.feature_extractor(x).flatten(1)
        x = self.classifier(representations)
        return x

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
        # calculate acc
        labels_hat = torch.argmax(y_hat, dim=1)
        acc = torch.sum(y == labels_hat).item() / (len(y) * 1.0)
        # log the outputs
        self.log_dict({'val_loss': loss, 'val_acc': acc})

    def testing_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('test_loss', loss)
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
    model = ResNet50(n_classes=3)
    print(summary(model, (3, 224, 224)))
