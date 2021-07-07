# -*- coding: utf-8 -*-
"""
@Time ： 2021/6/29 21:33
@Auth ： JsHou
@Email : 137073284@qq.com
@File ：train.py

"""
import os

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from models import ResNet50, EasyCNN
from utils import get_root_path


def train(Model):
    """
    train specified model
    :param Model: class name of model to be trained
    :return: None
    """
    # model instance name
    model_name = str(Model.__name__).lower()

    # project root path
    root_path = get_root_path()

    # define tensorboard logger and pytorch checkpoint callback
    logger = TensorBoardLogger(os.path.join(root_path, 'tb_logs'), name=model_name)
    checkpoint_callback = ModelCheckpoint(dirpath=os.path.join(root_path, 'checkpoints', model_name),
                                          monitor='val_loss')

    # delete "gpus=？" if you don't have gpu devices
    trainer = pl.Trainer(gpus=1, logger=logger, callbacks=[checkpoint_callback], val_check_interval=0.5, max_epochs=20)
    _model = Model(n_classes=3)
    trainer.fit(_model)

    # save best checkpoint's path
    with open(os.path.join(checkpoint_callback.dirpath, "info.txt"), "w", encoding='utf-8') as f:
        f.write(checkpoint_callback.best_model_path)
        print('save best paras sucessfully!')


def test(Model):
    """
    test specified model on testing data
    :param Model: class name of model to be tested
    :return: None
    """
    # model instance name
    model_name = str(Model.__name__).lower()

    # project root path
    root_path = get_root_path()
    with open(os.path.join(root_path, 'checkpoints', model_name, 'info.txt'), 'r') as f:
        best_model_path = f.readline()

    model = Model.load_from_checkpoint(best_model_path)
    trainer = pl.Trainer(gpus=1, precision=16)
    trainer.test(model)


if __name__ == '__main__':
    test(ResNet50)
