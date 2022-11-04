import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

__all__ = ['SameConv2d', 'conv2d_same_1x1', 'conv2d_same_3x3', 'conv2d_same_5x5', 'conv2d_same_7x7']


class SameConv2d(nn.Conv2d):
    """
    Representation the `same` padding functionality from tensorflow for 2D convolution.
    Note that the padding argument in the initializer doesn't do anything now.

    Reference:
    - https://gist.github.com/ujjwal-9/e13f163d8d59baa40e741fc387da67df
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        :param x: the input tensor in shape of (B, C, F, T) or (C, F, T).
        :return:
        """

        if x.dim() not in [3, 4]:
            raise ValueError(f"The input tensor should be 3D or 4D tensor, but got dim = {x.dim()}.")

        kernel_size = (self.weight.size(2), self.weight.size(3))
        in_height, in_width = x.size(-2), x.size(-1)
        out_height = int(np.ceil(float(in_height) / float(kernel_size[0])))
        out_width  = int(np.ceil(float(in_width) / float(kernel_size[1])))

        pad_along_height = max((out_height - 1) * self.stride[0] + kernel_size[0] - in_height, 0)
        pad_along_width  = max((out_width - 1) * self.stride[1] + kernel_size[1] - in_width, 0)

        pad_top    = pad_along_height // 2
        pad_bottom = pad_along_height - pad_top
        pad_left   = pad_along_width // 2
        pad_right  = pad_along_width - pad_left

        x = Fn.pad(x, (pad_left, pad_right, pad_top, pad_bottom))
        return Fn.conv1d(x, weight=self.weight, bias=self.bias, stride=self.stride, padding=kernel_size, dilation=self.dilation, groups=self.groups)


def conv2d_same_1x1(in_channels: int, out_channels: int, stride: int = 1, dilation: int  = 1, bias: bool = True) -> SameConv2d:
    """
    The 2D convolution with 1x1 kernel and 'same' padding.

    :param in_channels: the number of features in input tensor.
    :param out_channels: the number of feature in output tensor.
    :param stride: the convolution stride. Default: 1.
    :param dilation: the convolution dilation. Default: 1.
    :param bias: to add a learnable bias to the output or not. Default: True.
    :return: the 1x1 'same' padding convolution layer.
    """

    return SameConv2d(in_channels, out_channels, kernel_size=1, stride=stride, dilation=dilation, bias=bias)


def conv2d_same_3x3(in_channels: int, out_channels: int, stride: int = 1, dilation: int = 1, bias: bool = True) -> SameConv2d:
    """
    The 2D convolution with 3x3 kernel and 'same' padding.

    :param in_channels: the number of features in input tensor.
    :param out_channels: the number of feature in output tensor.
    :param stride: the convolution stride. Default: 1.
    :param dilation: the convolution dilation. Default: 1.
    :param bias: to add a learnable bias to the output or not. Default: True.
    :return: the 3x3 'same' padding convolution layer
    """
    return SameConv2d(in_channels, out_channels, kernel_size=3, stride=stride, dilation=dilation, bias=bias)


def conv2d_same_5x5(in_channels: int, out_channels: int, stride: int = 1, dilation: int = 1, bias: bool = True) -> SameConv2d:
    """
    The 2D convolution with 5x5 kernel and 'same' padding.

    :param in_channels: the number of features in input tensor.
    :param out_channels: the number of feature in output tensor.
    :param stride: the convolution stride. Default: 1.
    :param dilation: the convolution dilation. Default: 1.
    :param bias: to add a learnable bias to the output or not. Default: True.
    :return: the 5x5 'same' padding convolution layer.
    """

    return SameConv2d(in_channels, out_channels, kernel_size=5, stride=stride, dilation=dilation, bias=bias)


def conv2d_same_7x7(in_channels: int, out_channels: int, stride: int = 1, dilation: int = 1, bias: bool = True) -> SameConv2d:
    """
    The 2D convolution with 7x7 kernel and 'same' padding.

    :param in_channels: the number of features in input tensor.
    :param out_channels: the number of feature in output tensor.
    :param stride: the convolution stride. Default: 1.
    :param dilation: the convolution dilation. Default: 1.
    :param bias: to add a learnable bias to the output or not. Default: True.
    :return: the 7x7 'same' padding convolution layer.
    """

    return SameConv2d(in_channels, out_channels, kernel_size=7, stride=stride, dilation=dilation, bias=bias)
