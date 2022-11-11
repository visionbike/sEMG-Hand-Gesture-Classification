from typing import Optional
from collections import OrderedDict
import torch
import torch.nn as nn
from network.layer import *
from network.model.init_weight import *

__all__ = ['Baseline']


class Baseline(nn.Module):
    """
    The implementation of the network from 10.48550/arXiv.2006.03645
    Given an input of (B, C, N), that will be used to predict a classification

    Reference:
    - https://doi.org/10.48550/arXiv.2006.03645
    """

    def __init__(self,
                 in_channels: int,
                 mid_channels: int,
                 in_dims: int,
                 num_classes: int,
                 norm_layer: nn.Module = nn.Identity,
                 act_layer: nn.Module = nn.Identity,
                 att_layer: nn.Module = nn.Identity,
                 drop_rate: Optional[float] = 0.3,
                 att_kwargs: Optional[dict] = None,
                 **kwargs: dict):
        """

        :param in_channels:the number of the input tensor channels.
        :param mid_channels: the number of the intermediate layer channels.
        :param in_dims: the number of feature dims.
        :param num_classes: the number of classes.
        :param norm_layer: the normalization layer. Default: `nn.Identity`.
        :param act_layer: the activation layer. Default: `nn.Identity`.
        :param att_layer: the attention layer. Default: `nn.Identity`.
        :param drop_rate: the dropout ratio. Default: 0.3.
        :param att_kwargs: the attention argument dict. Default: None.
        :param kwargs: argument dict.
        """

        if att_kwargs is None:
            self.att_kwargs = {}
        else:
            self.att_kwargs = att_kwargs

        if drop_rate is None:
            drop = nn.Identity
        else:
            drop = nn.Dropout

        super(Baseline, self).__init__()

        # expansion block
        self.expansion = nn.Sequential(OrderedDict(
            conv=conv1d_same_1x1(in_channels, mid_channels, bias=True),
            norm=norm_layer([mid_channels, in_dims]) if norm_layer.__name__ == 'LayerNorm' else norm_layer(mid_channels),
            act=act_layer(inplace=True)
        ))

        # attention layer
        self.att = att_layer(mid_channels, **att_kwargs)

        # classifier
        self.clf = nn.Sequential(OrderedDict(
            linear_1=nn.Linear(mid_channels, 500, bias=True),
            norm_1=nn.LayerNorm(500),
            act_1=nn.Mish(inplace=True),
            drop_1=drop(p=drop_rate),
            #
            linear_2=nn.Linear(500, 500, bias=True),
            norm_2=nn.LayerNorm(500),
            act_2=nn.Mish(inplace=True),
            drop_2=drop(p=drop_rate),
            #
            linear_3=nn.Linear(500, 2000, bias=True),
            norm_3=nn.LayerNorm(2000),
            act_3=nn.Mish(inplace=True),
            drop_3=drop(p=drop_rate),
            #
            linear_4=nn.Linear(2000, num_classes, bias=False)
        ))

        self.apply(init_weight)

    def forward(self, x: torch.Tensor):
        """

        :param x: the input tensor in shape of (B, C, N).
        :return:
        """

        z = self.expansion(x)       # expansion layer
        z = self.att(z)             # attention layer
        z = torch.sum(z, dim=-1)    # temporal reduction
        z = self.clf(z)             # classifier
        return z
