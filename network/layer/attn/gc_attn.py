from collections import OrderedDict
import torch
import torch.nn as nn
from network.layer.conv1d import *

__all__ = ['GlobalContextAttention']


class GlobalContextAttention(nn.Module):
    """
    The implementation of Non-local inspired Global Context Attention mechanism.

    Reference:
    - https://arxiv.org/pdf/2012.13375.pdf
    - https://github.com/xvjiarui/GCNet/blob/master/mmdet/ops/gcb/context_block.py
    """

    def __init__(self, in_channels: int, ratio: float = 1.):
        """

        :param in_channels: the number of channels in input tensor.
        :param ratio: the reduction ratio for bottleneck transform module.
        """

        super(GlobalContextAttention, self).__init__()

        mid_channels = int(in_channels // ratio)
        mid_channels = mid_channels if mid_channels > 1 else 1

        self.context_conv = conv1d_same_1x1(in_channels, 1, bias=True)
        self.transform = nn.Sequential(OrderedDict(
            conv1=conv1d_same_1x1(in_channels, mid_channels, bias=True),
            norm=nn.LayerNorm(mid_channels),
            act=nn.Mish(inplace=True),
            conv2=conv1d_same_1x1(mid_channels, in_channels, bias=True)
        ))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """

        :param x: the input tensor in shape of (B, C, N) or (B, C, F, T).
        :return: the output attention map in shape of (B, C, N) or (B, C, F, T).
        """

        if x.dim() not in [3, 4]:
            raise ValueError(f"The input tensor should be in 3D or 4D tensor in shape of (B, C, N) or (B, C, F, T). Got {x.dim()}D tensor.")

        b, c = x.size(0), x.size(1)

        z = None
        if x.dim() == 3:
            z = x.clone().detach()
        elif x.dim() == 4:
            n = x.size(2) * x.size(3)
            z = x.view(b, c, n)     # (B, C, F, T) -> (B, C, F*T)

        # compute context map
        # (B, C, N) -> (B, 1, N)
        context_mask = self.context_conv(z)
        # (B, 1, N) -> (B, 1, N, 1)
        context_mask = context_mask.unsqueeze(-1)
        # (B, 1, C, N) * (B, 1, N, 1) -> (B, 1, C, 1)
        context = torch.matmul(z, context_mask)
        print(context.shape)
        # (B, 1, C, 1) -> (B, C, 1)
        context = context.view(b, c, -1)

        print(context.shape)

        # recalibrate context map or obtain attention score map
        # (B, C, 1) -> (B, C // r, 1) -> (B, C, 1)
        att_score = self.transform(context)
        if x.dim() == 4:
            att_score = att_score.unsqueeze(-1)     # (B, C, 1, 1)

        # get output map
        z = x + att_score

        return z
