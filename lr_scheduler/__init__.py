from torch import nn
from torch.optim import Optimizer
from .cosine_start import *

__all__ = ['get_scheduler']


def get_scheduler(optimizer: Optimizer, name: str = 'cosine', **kwargs) -> nn.Module:
    """
    Get learning rate scheduler.

    :param optimizer: the optimizer.
    :param name: the scheduler name. Default: 'cosine'.
    :param kwargs: the optional arguments.
    :return: the learning rate scheduler.
    """

    if name == 'cosine_start':
        if not {'T_0', 'T_start', 'T_mult', 'eta_min', 'gamma'}.issubset(kwargs.keys()):
            raise ValueError(f"Not found required arguments = {{'T_0', 'T_start', 'T_mult', 'eta_min', 'gamma'}}, but got {kwargs.keys()}.")
        scheduler = CosineAnnealingStartLR(optimizer, **kwargs)
    else:
        raise ValueError(f"Expected values: 'cosine', but got 'name' = {name}.")
    return scheduler
