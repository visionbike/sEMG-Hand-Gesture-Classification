from typing import Optional
import torch
import torch.nn as nn
from network.layer import *
from network.model.ffc_resnet import *
from network.model.pre_ffc_resnet import *
from network.model.init_weight import *

__all__ = ['FFCNet']


class FFCNet(nn.Module):
    def __init__(self,
                 fft_len: int,
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
                 backbone: str = 'resnet',
                 norm_layer: nn.Module = nn.LayerNorm,
                 act_layer: nn.Module = nn.Mish,
                 att_layer: nn.Module = nn.Identity,
                 att_kwargs: Optional[dict] = None,
                 **kwargs: dict):

        super(FFCNet, self).__init__()

        self.stft_conv = StftConv(fft_len=fft_len, win_hop=(fft_len // 2), feature_type='complex', trainable=False)
        if backbone == 'resnet':
            self.ffc_resnet = FFCResnet(2 * in_channels, in_freq, in_time, mid_channels,
                                        num_classes, num_blocks,
                                        ratio_gin, ratio_gout,
                                        ksize, stride, padding, groups, enable_lfu,
                                        norm_layer, act_layer, att_layer, att_kwargs, **kwargs)
        elif backbone == 'preresnet':
            self.ffc_resnet = PreFFCResnet(2 * in_channels, in_freq, in_time, mid_channels,
                                           num_classes, num_blocks,
                                           ratio_gin, ratio_gout,
                                           ksize, stride, padding, groups, enable_lfu,
                                           norm_layer, act_layer, att_layer, att_kwargs, **kwargs)

        self.apply(init_weight)

    def forward(self, x: torch.Tensor):
        # (B, C, F, T) -> (B, C, F, T, 2)
        z = self.stft_conv(x)
        # concatenate real and imagine parts
        # (B, C, F, T, 2) -> (B, F, T, C, 2)
        z = torch.permute(z, (0, 2, 3, 1, 4)).contiguous()
        z = z.view(z.size(0), z.size(1), z.size(2), -1)
        # (B, F, T, 2 * C) -> (B, 2 * C, F, T)
        z = torch.permute(z, (0, 3, 1, 2)).contiguous()
        # (B, 2 * C, F, T) -> (B, 2 * C, F - 1, T)
        z = z[:, :, 1:, :].contiguous()
        #
        z = self.ffc_resnet(z)
        return z
