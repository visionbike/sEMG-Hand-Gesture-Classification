import torch.nn as nn
from network.layer import *

__all__ = ['init_weight']


def init_weight(m: nn.Module) -> None:
    """
    Reference:
    - https://github.com/digantamisra98/Mish/issues/37

    :param m: the input Pytorch module.
    :return:
    """
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, a=0.0003, mode='fan_out')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
        elif isinstance(m, (SameConv1d, SameConv2d)):
            nn.init.kaiming_normal_(m.weight, a=0.0003, mode='fan_out')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

