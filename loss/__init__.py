from typing import Optional, Union
import torch.nn as nn
from .focal import *

__all__ = ['get_loss']


def get_loss(name: str = 'focal',
             weights: Optional[Union[list[float], float]] = None,
             gamma: Optional[float] = 2.,
             **kwargs) -> nn.Module:
    """
    Get criterion.

    :param weights: the class balanced weights. Default: None.
    :param name: the criterion name. Default: 'focal'.
    :param gamma: the controlling constant in FocalLoss. Default: 2.
    :param kwargs: the optional arguments.
    :return: the criterion.
    """

    if 'reduction' not in kwargs.keys():
        raise ValueError(f"Not found 'reduction' argument, but got {kwargs}.")

    if name == 'ce':
        loss = nn.CrossEntropyLoss(weight=weights, reduction=kwargs['reduction'])
    elif name == 'focal':
        if gamma is None or gamma < 0:
            raise ValueError(f"Expected positive float 'gamma', but got 'gamma' = {gamma}.")
        if 'num_classes' not in kwargs.keys():
            raise ValueError(f"Not found 'num_classes' argument, but got {kwargs}.")
        loss = FocalLoss(kwargs['num_classes'], alpha=weights, gamma=gamma, reduction=kwargs['reduction'])
    else:
        raise ValueError(f"Expected values: 'ce'|'focal', but got 'name' = {name}. ")
    return loss
