import torch.nn as nn

__all__ = ['init_weight']


def init_weight(m: nn.Module) -> None:
    """
    Reference:
    - https://github.com/digantamisra98/Mish/issues/37

    :param m: the input Pytorch module.
    :return:
    """
    if isinstance(m, (nn.Conv1d, nn.Linear)):
        nn.init.kaiming_uniform_(m.weight, a=0.0003)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
        nn.init.constant_(m.weight, 1,)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
