from typing import Union
import torch
import torch.nn as nn
from network.layer.norm import *

__all__ = ['FFConv', 'FourierUnit', 'SpectralTransform']


class FourierUnit(nn.Module):
    """
    The implementation of Fourier unit that transforms spatial features into spectral domain, conduct update on spectral domain,
    and convert back to spatial domain.

    Reference:
    - https://github.com/pkumivision/FFC/blob/main/model_zoo/ffc.py
    - https://github.com/saic-mdal/lama/blob/main/saicinpainting/training/modules/ffc.py
    - https://proceedings.neurips.cc/paper/2020/file/2fd5d41ec6cfab47e32164d5624269b1-Paper.pdf
    """

    def __init__(self,
                 in_channels: int,
                 in_height: int,
                 in_width: int,
                 out_channels: int,
                 groups: int = 1,
                 fft_norm: str = 'ortho',
                 norm_layer: nn.Module = nn.LayerNorm,
                 act_layer: nn.Module = nn.Mish):
        """

        :param in_channels: the number of channels in input tensor.
        :param in_height: the size of height dimension in the tensor.
        :param in_width: the size of width dimension in the tensor.
        :param out_channels: the number of channels in output tensor.
        :param groups: the number of channel groups. Default: 1.
        :param fft_norm: normalization mode, including:
                'forward': normalized by 1/n;
                'backward': no normalization;
                'ortho': normalized by 1/sqrt(n). Default: 'ortho'.
        :param norm_layer: the normalization layer. Default: `nn.LayerNorm`.
        :param act_layer: the activation layer. Default: `nn.Mish`.
        """

        self.groups = groups
        self.fft_norm = fft_norm
        super(FourierUnit, self).__init__()

        self.conv = nn.Conv2d(in_channels=(2 * in_channels), out_channels=(2 * out_channels),
                              kernel_size=1, stride=1, padding=0, groups=self.groups, bias=False)
        self.norm = norm_layer(2 * out_channels)
        self.act = act_layer(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        :param x: the input tensor in shape of (B, C, F, T).
        :return:
        """

        b, c, f, t = x.shape

        # apply real unidimensional fast Fourier transform across frequency dimension
        # fft1d
        ffted = torch.fft.rfftn(x, dim=2, norm=self.fft_norm)   # (B, C, F, T) -> (B, C, F/2+1, T)
        # concat
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)   # (B, C, F/2+1, T) -> (B, C, F/2+1, T, 2)
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()       # (B, C, F/2+1, T, 2) -> (B, C, 2, F/2+1, T)
        ffted = ffted.view((b, -1,) + ffted.size()[3:])         # (B, C, 2, F/2+1, T) -> (B, 2*C, F/2+1, T)

        # apply 1x1 convolutional block in the frequency domain
        ffted = self.conv(ffted)    # (B, 2*C, F/2+1, T)
        if isinstance(self.norm, LayerNorm):
            ffted = torch.permute(ffted, (0, 2, 3, 1)).contiguous()
            ffted = self.norm(ffted)
            ffted = torch.permute(ffted, (0, 3, 1, 2)).contiguous()
        else:
            ffted = self.norm(ffted)
        ffted = self.act(ffted)
        ffted = ffted.view((b, -1, 2,) + ffted.size()[2:])      # (B, 2*C, F/2+1, T) -> (B, C, 2, F/2+1, T)
        ffted = ffted.permute(0, 1, 3, 4, 2).contiguous()       # (B, C, 2, F/2+1, T) -> (B, C, F/2+1, T, 2)
        ffted = torch.complex(ffted[..., 0], ffted[..., 1])     # (B, C, F/2+1, T, 2) -> (B, C, F/2+1, T)

        # apply inverse Fourier transform
        # (B, C, F/2+1, T) -> (B, C, F/2+1, T)
        z = torch.fft.irfft(ffted, n=x.shape[2], dim=2, norm=self.fft_norm)
        return z


class SpectralTransform(nn.Module):
    """
    Implementation of spectral transform module to enlarge the receptive field of convolution to
    full resolution of input feature map in fourier domain.

    Reference:
    - https://github.com/pkumivision/FFC/blob/main/model_zoo/ffc.py
    - https://github.com/saic-mdal/lama/blob/main/saicinpainting/training/modules/ffc.py
    - https://proceedings.neurips.cc/paper/2020/file/2fd5d41ec6cfab47e32164d5624269b1-Paper.pdf
    """

    def __init__(self,
                 in_channels: int,
                 in_height: int,
                 in_width: int,
                 out_channels: int,
                 stride: int = 1,
                 groups: int = 1,
                 enable_lfu: bool = True,
                 fft_norm: str = 'ortho',
                 norm_layer: nn.Module = nn.LayerNorm,
                 act_layer: nn.Module = nn.Mish):
        """

        :param in_channels: the number of channels in input tensor.
        :param in_height: the size of height dimension in the tensor.
        :param in_width: the size of width dimension in the tensor.
        :param out_channels: the number of channels in output tensor.
        :param stride: the convolution stride. Default: 1.
        :param groups: the convolution groups. Default: 1.
        :param enable_lfu: to apply local fourier unit to capture semi-local information. Default: True.
        :param fft_norm: normalization mode:
                'forward': normalized by 1/n;
                'backward': no normalization;
                'ortho': normalized by 1/sqrt(n). Default: 'ortho'.
        :param norm_layer: the normalization layer. Default: `nn.LayerNorm`.
        :param act_layer: the activation layer. Default: `nn.Mish`.
        """

        self.enable_lfu = enable_lfu
        self.stride = stride
        self.fft_norm = fft_norm
        super(SpectralTransform, self).__init__()

        if stride == 2:
            self.downsample = nn.AvgPool2d(kernel_size=2, stride=2)
            in_height = in_height // 2
            in_width = in_width // 2
        else:
            self.downsample = nn.Identity()

        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=(out_channels // 2),
                               kernel_size=1, groups=groups, bias=False)
        self.norm1 = norm_layer(out_channels // 2)
        self.act1 = act_layer(inplace=True)

        self.conv2 = nn.Conv2d(in_channels=(out_channels // 2),
                               out_channels=out_channels,
                               kernel_size=1, groups=groups, bias=False)

        self.fu = FourierUnit(in_channels=(out_channels // 2),
                              in_height=in_height,
                              in_width=in_width,
                              out_channels=(out_channels // 2),
                              groups=groups, fft_norm=self.fft_norm,
                              norm_layer=norm_layer, act_layer=act_layer)
        if self.enable_lfu:
            self.lfu = FourierUnit(in_channels=(out_channels // 2),
                                   in_height=(in_height // 2),
                                   in_width=(in_width // 2),
                                   out_channels=(out_channels // 2),
                                   groups=groups, fft_norm=self.fft_norm,
                                   norm_layer=norm_layer, act_layer=act_layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        :param x: the input tensor in shape of (B, C, F, T).
        :return:
        """

        x = self.downsample(x)
        x = self.conv1(x)
        if isinstance(self.norm1, LayerNorm):
            x = torch.permute(x, (0, 2, 3, 1)).contiguous()
            x = self.norm1(x)
            x = torch.permute(x, (0, 3, 1, 2)).contiguous()
        else:
            x = self.norm1(x)
        x = self.act1(x)
        z = self.fu(x)

        if self.enable_lfu:
            b, c, f, t = x.shape
            num_splits = 2
            split_size_f = f // num_splits
            split_size_t = t // num_splits
            xs = torch.cat(torch.split(x[:, : (c // 4)], split_size_f, dim=-2), dim=1).contiguous()
            xs = torch.cat(torch.split(xs, split_size_t, dim=-1), dim=1).contiguous()
            xs = self.lfu(xs)
            xs = xs.repeat(1, 1, num_splits, num_splits).contiguous()
        else:
            xs = 0
        z = self.conv2(x + z + xs)
        return z


class FFConv(nn.Module):
    """
    The implementation of Fast Fourier Convolution module.

    Reference:
    - https://github.com/pkumivision/FFC/blob/main/model_zoo/ffc.py
    - https://github.com/saic-mdal/lama/blob/main/saicinpainting/training/modules/ffc.py
    - https://proceedings.neurips.cc/paper/2020/file/2fd5d41ec6cfab47e32164d5624269b1-Paper.pdf
    """

    def __init__(self,
                 in_channels: int,
                 in_height: int,
                 in_width: int,
                 out_channels: int,
                 ratio_gin: float,
                 ratio_gout: float,
                 ksize: int,
                 stride: int = 1,
                 padding: int = 0,
                 groups: int = 1,
                 padding_type: str = 'reflect',
                 enable_lfu: bool = True,
                 fft_norm: str = 'ortho',
                 norm_layer: nn.Module = nn.LayerNorm,
                 act_layer: nn.Module = nn.Mish):
        """

        :param in_channels: the number of channels in input tensor.
        :param in_height: the size of height dimension in the tensor.
        :param in_width: the size of width dimension in the tensor.
        :param out_channels: the number of channels in output tensor.
        :param ratio_gin: the ratio of feature channels allocated to the global part.
        :param ratio_gout: the ratio of global part for output tensor.
        :param ksize: the kernel size.
        :param stride: the convolution stride. Default: 1.
        :param padding: the convolution padding. Default: 0.
        :param groups: the groups. Default: 1.
        :param padding_type: the padding mode. Default: 'reflect'.
        :param enable_lfu: to apply local fourier unit to capture semi-local information. Default: True.
        :param fft_norm: normalization mode:
                'forward': normalized by 1/n;
                'backward': no normalization;
                'ortho': normalized by 1/sqrt(n).
                Default: 'ortho'.
        :param norm_layer: the normalization layer. Default: `nn.LayerNorm`.
        :param act_layer: the activation layer. Default: `nn.Mish`.
        """

        if stride not in [1, 2]:
            raise ValueError(f"Invalid argument 'stride' = {stride}. Valid values should be [1, 2].")

        self.stride = stride
        self.ratio_gin = ratio_gin
        self.ratio_gout = ratio_gout
        self.padding_type = padding_type
        self.fft_norm = fft_norm

        super(FFConv, self).__init__()

        self.in_cg = int(in_channels * self.ratio_gin)
        self.in_cl = in_channels - self.in_cg
        self.out_cg = int(out_channels * self.ratio_gout)
        self.out_cl = out_channels - self.out_cg

        module = nn.Identity if (self.in_cl == 0 or self.out_cl == 0) else nn.Conv2d
        self.conv_l2l = module(self.in_cl, self.out_cl, ksize, stride, padding, groups=groups, bias=True)
        module = nn.Identity if (self.in_cl == 0 or self.out_cg == 0) else nn.Conv2d
        self.conv_l2g = module(self.in_cl, self.out_cg, ksize, stride, padding, groups=groups, bias=True)
        module = nn.Identity if (self.in_cg == 0 or self.out_cl == 0) else nn.Conv2d
        self.conv_g2l = module(self.in_cg, self.out_cl, ksize, stride, padding, groups=groups, bias=True)
        module = nn.Identity if (self.in_cg == 0 or self.out_cg == 0) else SpectralTransform
        self.conv_g2g = module(self.in_cg, in_height, in_width, self.out_cg, stride, groups if groups == 1 else (groups // 2), enable_lfu, fft_norm, norm_layer, act_layer)

    def forward(self, x: Union[tuple[torch.Tensor, torch.Tensor], torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """

        :param x: the input tensor in shape of (B, C, F, T) or the tuple of 2 global and local tensors in shape of (B, C, F, T).
        :return: local and global branch tensor.
        """

        if isinstance(x, tuple):
            x_l, x_g = x
        else:
            x_l, x_g = x[:, :-self.in_cg], x[:, -self.in_cg:]
        z_l, z_g = None, None

        if self.ratio_gout != 1:
            z_l = self.conv_l2l(x_l) + self.conv_g2l(x_g)
        if self.ratio_gout != 0:
            z_g = self.conv_l2g(x_l) + self.conv_g2g(x_g)
        return z_l, z_g
