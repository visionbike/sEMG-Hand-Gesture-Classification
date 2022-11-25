from typing import Union
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import EPOCH_OUTPUT
from pytorch_lightning.loggers.neptune import NeptuneLogger
from neptune.new.types import File
from network import *
from optimizer import *
from loss import *
from lr_scheduler import *
from metric import *

__all__ = ['Trainer']


class Trainer(pl.LightningModule):
    """
    The pytorch-lightning implementation of the trainer module.
    """

    def __init__(self,
                 network_kwargs: dict,
                 optimizer_kwargs: dict,
                 criterion_kwargs: dict,
                 scheduler_kwargs: dict,
                 metric_kwargs: dict):
        """

        :param network_kwargs: the network argument dict.
        :param optimizer_kwargs: the optimizer argument dict.
        :param criterion_kwargs: the criterion argument dict.
        :param scheduler_kwargs: the lr scheduler argument dict.
        :param metric_kwargs: the metric argument dict.
        """

        super(Trainer, self).__init__()
        self.network = get_network(**network_kwargs)
        self.criterion = get_loss(**criterion_kwargs)
        self.optimizer_kwargs = optimizer_kwargs
        self.scheduler_kwargs = scheduler_kwargs
        self.metric_train = get_metrics(prefix='train/', **metric_kwargs)
        self.metric_val = get_metrics(prefix='val/', **metric_kwargs)
        self.metric_test = get_metrics(prefix='test/', cfmat=True, **metric_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.network(x)
        return z

    def training_step(self, batch: tuple, batch_idx: int) -> dict:
        x, y = batch
        z = self(x)
        loss = self.criterion(z, y)
        self.metric_train.update(z, y)
        self.log('train/loss', loss, prog_bar=True, logger=True, on_epoch=True, on_step=False)
        return dict(loss=loss)

    def validation_step(self, batch: tuple, batch_idx: int) -> None:
        x, y = batch
        z = self(x)
        loss = self.criterion(z, y)
        self.metric_val.update(z, y)
        self.log('val/loss', loss, prog_bar=True, logger=True, on_epoch=True, on_step=False, batch_size=x.shape[0])

    def test_step(self, batch: tuple, batch_idx: int) -> None:
        x, y = batch
        z = self(x)
        self.metric_test.update(z, y)

    def training_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        self.log_dict(self.metric_train.compute(), logger=True, on_epoch=True)
        self.metric_train.reset()

    def validation_epoch_end(self, outputs: Union[EPOCH_OUTPUT, list[EPOCH_OUTPUT]]) -> None:
        self.log_dict(self.metric_val.compute(), logger=True, on_epoch=True)
        self.metric_val.reset()

    def test_epoch_end(self, outputs: Union[EPOCH_OUTPUT, list[EPOCH_OUTPUT]]) -> None:
        # turn confusion matrix into a figure (Tensor cannot be logged as a scalar)
        metric_test = self.metric_test.compute()
        # log figure
        cfmat = metric_test.pop('test/cfmat').cpu().detach().numpy()
        print(cfmat.shape)
        # fig = plt.figure(figsize=(10, 10), dpi=600)
        ax = sns.heatmap(cfmat, annot=False, fmt='.2f', square=True, vmin=0, vmax=1, center=0, cbar=False)
        ax.set_xlim([0, cfmat.shape[0]])
        ax.set_ylim([0, cfmat.shape[0]])
        # plt.show()
        if isinstance(self.logger, NeptuneLogger):
            self.logger.experiment['test/confusion_matrix'].upload(File.as_image(fig))
        self.log_dict(metric_test, logger=True, on_epoch=True)
        self.metric_test.reset()

    def configure_optimizers(self) -> dict:
        optimizer = get_optimizer(self.network.parameters(), **self.optimizer_kwargs)
        lr_scheduler = get_scheduler(optimizer, **self.scheduler_kwargs)
        return dict(optimizer=optimizer,
                    lr_scheduler=dict(scheduler=lr_scheduler, interval='epoch', frequency=1, monitor='loss/val'))
