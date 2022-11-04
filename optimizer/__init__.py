from typing import Union, Iterable
import torch
import torch.optim as optim
from .ranger import *

__all__ = ['get_optimizer']


def get_optimizer(params: Union[Iterable[torch.Tensor], Iterable[dict]],
                  name: str = 'ranger',
                  lr: float = 1e-3,
                  **kwargs) -> optim.Optimizer:
    """
    Get the optimizer.

    :param params: the model optimizers.
    :param name: the optimizer name. Default: 'ranger'.
    :param lr: the learning rate. Default: 1e-3.
    :param kwargs: optional argument dict.
    :return: the optimizer.
    """

    if name == 'sgd':
        optimizer = optim.SGD(params, lr, **kwargs)
    elif name == 'adam':
        optimizer = optim.Adam(params, lr, **kwargs)
    elif name == 'adamw':
        optimizer = optim.AdamW(params, lr, **kwargs)
    elif name == 'radam':
        optimizer = optim.RAdam(params, lr, **kwargs)
    elif name == 'ranger':
        optimizer = Ranger(params, lr, **kwargs)
    else:
        raise ValueError(f"Expected values: 'adam'|'adamw'|'radam'|'ranger', but got 'name' = {name}.")
    return optimizer
