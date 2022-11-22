from typing import Union
import torch
import torch.nn as nn
import torch.nn.functional as Fn
from network.layer import *
from network.block import *
from network.model.init_weight import *

__all__ = ['FFCResnet']


class BasicBlock(nn.Module):
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
        
        super(BasicBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_cg = int(in_channels * ratio_gin)
        
        self.ffc_block1 = FFCBlock(in_channels, in_freq, in_time, in_channels * self.expansion,
                                   ratio_gin, ratio_gout,
                                   ksize, stride, padding, groups,
                                   enable_lfu, norm_layer, act_layer)
        self.ffc_block2 = FFCBlock(in_channels * self.expansion, in_freq, in_time, out_channels,
                                   ratio_gin, ratio_gout,
                                   ksize, stride, padding, groups,
                                   enable_lfu, norm_layer, act_layer)
        
    def forward(self, x: Union[tuple[torch.Tensor, torch.Tensor], torch.Tensor]) -> torch.Tensor:
        """
        
        :param x: the input tensor in shape of (B, C, F, T) or the tuple of 2 global and local tensors in shape of (B, C, F, T).
        :return: local and global branch tensor.
        """

        x_l, x_g = x[:, :-self.in_cg], x[:, -self.in_cg:] if type(x) is torch.Tensor else x
        #
        z_l, z_g = self.ffc_block1(x)
        z_l, z_g = self.ffc_block2((z_l, z_g))
        #
        # if expand channel, also pad zeros to identity
        if x_l.size(1) != z_l.size(1):
            x_l = torch.permute(x_l, (0, 2, 3, 1)).contiguous()
            c1 = (z_l.size(1) - x_l.size(-1)) // 2
            c2 = z_l.size(1) - x_l.size(-1) - c1
            x_l = Fn.pad(x_l, (c1, c2), 'constant', 0)
            x_l = torch.permute(x_l, (0, 3, 1, 2)).contiguous()
        #
        if x_g.size(1) != z_g.size(1):
            x_g = torch.permute(x_g, (0, 2, 3, 1)).contiguous()
            c1 = (z_g.size(1) - x_g.size(-1)) // 2
            c2 = z_g.size(1) - x_g.size(-1) - c1
            x_g = Fn.pad(x_g, (c1, c2), 'constant', 0)
            x_g = torch.permute(x_g, (0, 3, 1, 2)).contiguous()
        #
        z_l += x_l
        z_g += x_g
        #
        z = torch.cat((z_l, z_g), dim=1)
        return z


class FFCResnet(nn.Module):
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
                 act_layer: nn.Module = nn.Mish):

        super(FFCResnet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=7, stride=1, padding=3, bias=True)
        self.norm1 = norm_layer(mid_channels)
        self.act1 = act_layer(inplace=True)
        #
        self.ffc_blocks = []
        in_channels = mid_channels
        for i in range(num_blocks):
            out_channels = 2 * mid_channels
            self.ffc_blocks += [
                BasicBlock(in_channels, in_freq, in_time, out_channels,
                           ratio_gin, ratio_gout,
                           ksize=ksize, stride=stride, padding=padding, groups=groups,
                           enable_lfu=enable_lfu, norm_layer=norm_layer, act_layer=act_layer)
            ]
            in_channels = out_channels
        #
        self.conv2 = nn.Conv2d(2 * mid_channels, mid_channels, kernel_size=7, stride=1, padding=3, bias=True)
        self.linear = nn.Linear(mid_channels * in_freq * in_time, num_classes, bias=False)

        self.apply(init_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        :param x:
        :return:
        """

        b, c, _, _ = x.size()
        z = self.conv1(x)
        if isinstance(self.norm1, LayerNorm):
            z = torch.permute(z, (0, 2, 3, 1)).contiguous()
            z = self.norm1(z)
            z = torch.permute(z, (0, 3, 1, 2)).contiguous()
        else:
            z = self.norm1(z)
        z = self.act1(z)
        #
        for block in self.ffc_blocks:
            z = block(z)
        #
        z = self.conv2(z)
        #
        z = z.view(b, -1)
        z = self.linear(z)
        return z
