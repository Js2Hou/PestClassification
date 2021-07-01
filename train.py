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

from models import ResNet50
from utils import get_root_path


if __name__ == '__main__':
    # project root path
    root_path = get_root_path()

    # define tensorboard logger and pytorch checkpoint callback
    logger = TensorBoardLogger(os.path.join(root_path, 'tb_logs'), name="resnet50")
    checkpoint_callback = ModelCheckpoint(dirpath=os.path.join(root_path, 'checkpoints', 'resnet50'), monitor='val_loss')

    # delete "gpus=？" if you don't have gpu devices
    trainer = pl.Trainer(gpus=1, logger=logger, callbacks=[checkpoint_callback], val_check_interval=0.25,
                         max_epochs=100)
    model = ResNet50(n_classes=3)
    trainer.fit(model)

    # save best checkpoint's path
    with open(os.path.join(checkpoint_callback.dirpath, "info.txt"), "w", encoding='utf-8') as f:
        f.write(checkpoint_callback.best_model_path)
        print('save best paras sucessfully!')

    # test model on testing data
    model = ResNet50.load_from_checkpoint(checkpoint_callback.best_model_path)
    trainer = pl.Trainer(gpus=1, precision=16)
    trainer.test(model)
