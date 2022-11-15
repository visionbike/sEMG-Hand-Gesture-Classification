from typing import Union
import torch
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import EPOCH_OUTPUT
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
        self.metric_train = get_metrics(postfix='/train', **metric_kwargs).clone()
        self.metric_val = get_metrics(postfix='/val', **metric_kwargs).clone()
        self.metric_test = get_metrics(postfix='/test', **metric_kwargs).clone()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.network(x)
        return z

    def training_step(self, batch: tuple, batch_idx: int) -> dict:
        x, y = batch
        z = self.forward(x)
        loss = self.criterion(z, y)
        self.metric_train.update(z, y)
        self.log('loss/train', loss, prog_bar=True, logger=True, on_epoch=True, on_step=False)
        return dict(loss=loss)

    def validation_step(self, batch: tuple, batch_idx: int) -> None:
        x, y = batch
        z = self.forward(x)
        loss = self.criterion(z, y)
        self.metric_val.update(z, y)
        self.log('loss/val', loss, prog_bar=True, logger=True, on_epoch=True, on_step=False)

    def test_step(self, batch: tuple, batch_idx: int) -> None:
        x, y = batch
        z = self.forward(x)
        self.metric_test.update(z, y)

    def training_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        self.lr_scheduler.step(self.current_epoch)
        self.log_dict(self.metric_train.compute(), logger=True, on_epoch=True)
        self.metric_train.reset()

    def validation_epoch_end(self, outputs: Union[EPOCH_OUTPUT, list[EPOCH_OUTPUT]]) -> None:
        self.log_dict(self.metric_val.compute(), logger=True, on_epoch=True)
        self.metric_val.reset()

    def test_epoch_end(self, outputs: Union[EPOCH_OUTPUT, list[EPOCH_OUTPUT]]) -> None:
        self.log_dict(self.metric_test.compute(), logger=True, on_epoch=True)
        self.metric_test.reset()

    def configure_optimizers(self) -> dict:
        self.optimizer = get_optimizer(self.network.parameters(), **self.optimizer_kwargs)
        self.lr_scheduler = get_scheduler(self.optimizer, **self.scheduler_kwargs)
        return dict(optimizer=self.optimizer)
