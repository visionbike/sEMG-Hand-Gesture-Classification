from typing import Optional, Union
import torch
import torch.nn as nn
from network.layer.fourier import *

__all__ = ['FFCNormAct']


class FFCNormAct(nn.Module):
    """
    The implementation of the block of FFC-Normalization-Activation layers.
    """

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
                 act_layer: nn.Module = nn.Mish,
                 drop_rate: Optional[float] = None):
        """

        :param in_channels: the number of input tensor channels.
        :param in_freq: the number of input frequency sample.
        :param in_time: the number of input time sample.
        :param out_channels: the number of output tensor channels.
        :param ratio_gin: the ratio of feature channels allocated to the global part.
        :param ratio_gout: the ratio of global part for output tensor.
        :param ksize: kernel size.
        :param stride: the convolutional stride. Default: 1.
        :param padding: the convolutional padding. Default: 0.
        :param groups: the groups. Default: 1.
        :param enable_lfu: whether to apply local fourier unit to capture semi-local information.
                Apply if in_freq % 2 = 0 and in_time % 2 = 0. Default: True.
        :param norm_layer: the normalization layer. Default: `nn.LayerNorm`.
        :param act_layer: the activation layer. Default: `nn.MIsh`.
        :param drop_rate: the drop ratio. Default: None.
        """

        if drop_rate is None:
            drop = nn.Identity
        else:
            drop = nn.Dropout

        super(FFCNormAct, self).__init__()

        self.ffc = FFConv(in_channels, in_freq, in_time,
                          out_channels,
                          ratio_gin, ratio_gout,
                          ksize, stride, padding, groups,
                          enable_lfu=enable_lfu, norm_layer=norm_layer, act_layer=act_layer)

        norm_l = nn.Identity if ratio_gout == 1 else norm_layer
        norm_g = nn.Identity if ratio_gout == 0 else norm_layer
        out_freq = int((in_freq + 2 * padding - ksize) / stride + 1)
        out_time = int((in_time + 2 * padding - ksize) / stride + 1)
        self.norm_l = norm_l([out_channels * int(1 - ratio_gout), out_freq, out_time]) if norm_l.__name__ == 'LayerNorm' else norm_l(out_channels * int(1 - ratio_gout))
        self.norm_g = norm_g([out_channels * int(out_channels * ratio_gout), out_freq, out_time]) if norm_g.__name__ == 'LayerNorm' else norm_g(out_channels * int(1 - ratio_gout))

        act_l = nn.Identity if ratio_gout == 1 else act_layer
        act_g = nn.Identity if ratio_gout == 0 else act_layer
        self.act_l = act_l(inplace=True)
        self.act_g = act_g(inplace=True)

        self.drop_l = drop(p=drop_rate)
        self.drop_g = drop(p=drop_rate)

    def  forward(self, x: Union[tuple[torch.Tensor, torch.Tensor], torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """

        :param x: the input tensor in shape of (B, C, F, T) or the tuple of 2 global and local tensors in shape of (B, C, F, T).
        :return: local and global branch tensor.
        """

        z_l, z_g = self.ffc(x)
        z_l = self.act_l(self.norm_l(z_l))
        z_g = self.act_g(self.norm_g(z_g))
        z_l = self.drop_l(z_l)
        z_g = self.drop_g(z_g)
        return z_l, z_g
