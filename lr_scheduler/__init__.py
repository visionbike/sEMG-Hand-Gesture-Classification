from torch import nn
from torch.optim import Optimizer
from .cosine_onecycle_start import *
from .cosine_warmup_restart import *

__all__ = ['get_scheduler']


def get_scheduler(optimizer: Optimizer, name: str = 'cosine', **kwargs) -> nn.Module:
    """
    Get learning rate scheduler.

    :param optimizer: the optimizer.
    :param name: the scheduler name. Default: 'cosine'.
    :param kwargs: the optional arguments.
    :return: the learning rate scheduler.
    """

    if name == 'cosine_onecycle':
        scheduler = CosineAnnealingOneCycleRL(optimizer, **kwargs)
    elif name == 'cosine_warmup_restart':
        scheduler = CosineAnnealingWarmupRestartRL(optimizer, **kwargs)
    else:
        raise ValueError(f"Expected values: 'cosine_onecycle'|'cosine_warmup_restart', but got 'name' = {name}.")
    return scheduler
