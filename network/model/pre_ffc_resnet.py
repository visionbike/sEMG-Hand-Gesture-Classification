from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as Fn
from network.layer import *
from network.block import *

__all__ = ['PreFFCResnet']


class PreBasicBlock(nn.Module):
    expansion = 1

    def __init__(self,
                 in_channels: int,
                 in_freq: int,
                 in_time: int,
                 out_channels: int,
                 ratio_gin: float,
                 ratio_gout: float,
                 ksize: int,
                 stride: int = 1,
                 padding: int = 0,
                 groups: int = 1,
                 enable_lfu: bool = True,
                 norm_layer: nn.Module = nn.LayerNorm,
                 act_layer: nn.Module = nn.Mish):

        super(PreBasicBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_cg = int(in_channels * ratio_gin)

        self.ffc_block1 = PreFFCBlock(in_channels, in_freq, in_time, in_channels * self.expansion,
                                      ratio_gin, ratio_gout,
                                      ksize, stride, padding, groups,
                                      enable_lfu, norm_layer, act_layer)
        self.ffc_block2 = PreFFCBlock(in_channels * self.expansion, in_freq, in_time, out_channels,
                                      ratio_gin, ratio_gout,
                                      ksize, stride, padding, groups,
                                      enable_lfu, norm_layer, act_layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        :param x: the input tensor in shape of (B, C, F, T) or the tuple of 2 global and local tensors in shape of (B, C, F, T).
        :return: local and global branch tensor.
        """

        z = self.ffc_block1(x)
        z = self.ffc_block2(z)
        # if expand channel, also pad zeros to identity
        if self.in_channels != self.out_channels:
            x = torch.permute(x, (0, 2, 3, 1)).contiguous()
            c1 = (self.out_channels - self.in_channels) // 2
            c2 = self.out_channels - self.in_channels - c1
            x = Fn.pad(x, (c1, c2), 'constant', 0)
            x = torch.permute(x, (0, 3, 1, 2)).contiguous()
        #
        z += x
        return z


class PreFFCResnet(nn.Module):
    def __init__(self,
                 in_channels: int,
                 in_freq: int,
                 in_time: int,
                 mid_channels: int,
                 num_classes: int,
                 num_blocks: int,
                 ratio_gin: float,
                 ratio_gout: float,
                 ksize: int,
                 stride: int = 1,
                 padding: int = 0,
                 groups: int = 1,
                 enable_lfu: bool = True,
                 norm_layer: nn.Module = nn.LayerNorm,
                 act_layer: nn.Module = nn.Mish,
                 att_layer: nn.Module = nn.Identity,
                 att_kwargs: Optional[dict] = None,
                 **kwargs: dict):

        super(PreFFCResnet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=7, stride=1, padding=3, bias=True)
        #
        ffc_blocks = []
        in_channels = mid_channels
        for i in range(num_blocks):
            out_channels = 2 * mid_channels
            ffc_blocks += [
                PreBasicBlock(in_channels, in_freq, in_time, out_channels,
                              ratio_gin, ratio_gout,
                              ksize=ksize, stride=stride, padding=padding, groups=groups,
                              enable_lfu=enable_lfu, norm_layer=norm_layer, act_layer=act_layer)
            ]
            in_channels = out_channels
        self.ffc_blocks = nn.ModuleList(ffc_blocks)
        #
        self.act2 = act_layer(inplace=True)
        self.norm2 = norm_layer(2 * mid_channels)
        self.conv2 = nn.Conv2d(2 * mid_channels, mid_channels, kernel_size=7, stride=1, padding=3, bias=True)
        #
        self.att = att_layer(**att_kwargs)
        #
        self.linear = nn.Linear(mid_channels * in_freq * in_time, num_classes, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        :param x:
        :return:
        """

        b, _, _, _ = x.size()
        z = self.conv1(x)
        #
        for block in self.ffc_blocks:
            z = block(z)
        #
        z = self.act2(z)
        if isinstance(self.norm2, LayerNorm):
            z = torch.permute(z, (0, 2, 3, 1)).contiguous()
            z = self.norm2(z)
            z = torch.permute(z, (0, 3, 1, 2)).contiguous()
        else:
            z = self.norm2(z)
        z = self.conv2(z)
        #
        z = self.att(z)
        #
        z = z.view(b, -1)
        z = self.linear(z)
        return z
