from typing import Optional
import torch.nn as nn
from .layer import *
from .model import *

__all__ = ['get_network']


def get_norm_layer(name: str) -> nn.Module:
    """
    The function to return normalization layer.

    :param name: the layer name.
    :return:
    """

    if name == 'layernorm':
        norm = LayerNorm
    elif name == 'batchnorm1d':
        norm = nn.BatchNorm1d
    elif name == 'batchnorm2d':
        norm = nn.BatchNorm2d
    elif name == 'none':
        norm = nn.Identity
    else:
        raise ValueError(f"Expected 'name' value: 'none'| 'layernorm' | 'batchnorm1d' | 'batchnorm2d', but got 'name' = {name}.")
    return norm


def get_act_layer(name: str) -> nn.Module:
    """
    The function to return activation layer.

    :param name: the layer name.
    :return:
    """

    if name == 'relu':
        act = nn.ReLU
    elif name == 'mish':
        act = nn.Mish
    elif name == 'none':
        act = nn.Identity
    else:
        raise ValueError(f"Expected 'name' value: 'none' | 'relu' | 'mish', but got 'name' = {name}.")
    return act


def get_att_layer(name: str) -> nn.Module:
    """
    The function to return attention layer.

    :param name: the layer name.
    :return:
    """

    if name == 'none':
        att = nn.Identity
    elif name == 'simple':
        att = SimpleAttention
    elif name == 'gc':
        att = GlobalContextAttention
    else:
        raise ValueError(f"Expected 'name' value: 'none' | 'simple' | 'gc', but got 'name' = {name}.")
    return att


def get_model(name: str) -> nn.Module:
    """
    The function to return the model.

    :param name: the model name
    :return:
    """

    if name == 'baseline':
        model = Baseline
    elif name == 'ffcnet':
        model = FFCNet
    else:
        raise ValueError(f"Expected 'name' value: 'baseline', but got 'name' = {name}.")
    return model


def get_network(name: str,
                in_channels: int,
                mid_channels: int,
                num_classes: int,
                norm: str = 'layernorm',
                act: str = 'mish',
                **kwargs):
    """
    Get network architecture.

    :param name: the model
    :param in_channels: the number of input channels.
    :param mid_channels: the number of intermediate channels.
    :param num_classes: the number of classes.
    :param norm: the normalization layer name. Default: 'layernorm'.
    :param act: the activation layer name. Default: 'mish'.
    :param kwargs:
    :return:
    """

    if 'attention' not in kwargs.keys():
        att_name = 'none'
        att_kwargs = {}
    else:
        att_kwargs = kwargs.pop('attention')
        att_name = att_kwargs.pop('name')

    norm_layer = get_norm_layer(norm)
    act_layer = get_act_layer(act)
    att_layer = get_att_layer(att_name)
    model = get_model(name)
    return model(in_channels=in_channels,
                 mid_channels=mid_channels,
                 num_classes=num_classes,
                 norm_layer=norm_layer,
                 act_layer=act_layer,
                 att_layer=att_layer,
                 att_kwargs=att_kwargs,
                 **kwargs)
