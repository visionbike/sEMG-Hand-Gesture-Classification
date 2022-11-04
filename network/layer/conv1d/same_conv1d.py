import torch
import torch.nn as nn
import torch.nn.functional as Fn

__all__ = ['SameConv1d', 'conv1d_same_1x1', 'conv1d_same_3x3', 'conv1d_same_5x5', 'conv1d_same_7x7']


class SameConv1d(nn.Conv1d):
    """
    Representation the `same` padding functionality from tensorflow for 1D convolution.
    Note that the padding argument in the initializer doesn't do anything now.

    Reference:
    - https://github.com/pytorch/pytorch/issues/3867
    - https://github.com/okrasolar/pytorch-timeseries/blob/master/src/models/utils.py
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        :param x: the input 1D tensor in shape of (B, C, N) or (C, N).
        :return:
        """

        if x.dim() not in [2, 3]:
            raise ValueError(f"The input tensor should be 2D or 3D tensor, but got dim = {x.dim()}.")

        kernel_size, stride, dilation = self.weight.size(2), self.stride[0], self.dilation[0]
        l_in = l_out = x.size(-1)
        padding = (((l_out - 1) * stride) - l_in + (dilation * (kernel_size - 1)) + 1)
        if padding % 2 != 0:
            x = Fn.pad(x, [0, 1])
        return Fn.conv1d(x, weight=self.weight, bias=self.bias, stride=self.stride, padding=(padding // 2), dilation=dilation, groups=self.groups)


def conv1d_same_1x1(in_channels: int, out_channels: int, stride: int = 1, dilation: int  = 1, bias: bool = True) -> SameConv1d:
    """
    The 1D convolution with 1x1 kernel and 'same' padding.

    :param in_channels: the number of channels in input tensor.
    :param out_channels: the number of channels in output tensor.
    :param stride: the convolution stride. Default: 1.
    :param dilation: the convolution dilation. Default: 1.
    :param bias: to add a learnable bias to the output or not. Default: True.
    :return: the 1x1 'same' padding convolution layer.
    """

    return SameConv1d(in_channels, out_channels, kernel_size=1, stride=stride, dilation=dilation, bias=bias)


def conv1d_same_3x3(in_channels: int, out_channels: int, stride: int = 1, dilation: int = 1, bias: bool = True) -> SameConv1d:
    """
    The 1D convolution with 3x3 kernel and 'same' padding.

    :param in_channels: the number of channels in input tensor.
    :param out_channels: the number of channels in output tensor.
    :param stride: the convolution stride. Default: 1.
    :param dilation: the convolution dilation. Default: 1.
    :param bias: to add a learnable bias to the output or not. Default: True.
    :return: the 3x3 'same' padding convolution layer
    """
    return SameConv1d(in_channels, out_channels, kernel_size=3, stride=stride, dilation=dilation, bias=bias)


def conv1d_same_5x5(in_channels: int, out_channels: int, stride: int = 1, dilation: int = 1, bias: bool = True) -> SameConv1d:
    """
    The 1D convolution with 5x5 kernel and 'same' padding.

    :param in_channels: the number of channels in input tensor.
    :param out_channels: the number of channels in output tensor.
    :param stride: the convolution stride. Default: 1.
    :param dilation: the convolution dilation. Default: 1.
    :param bias: to add a learnable bias to the output or not. Default: True.
    :return: the 5x5 'same' padding convolution layer.
    """

    return SameConv1d(in_channels, out_channels, kernel_size=5, stride=stride, dilation=dilation, bias=bias)


def conv1d_same_7x7(in_channels: int, out_channels: int, stride: int = 1, dilation: int = 1, bias: bool = True) -> SameConv1d:
    """
    The 1D convolution with 7x7 kernel and 'same' padding.

    :param in_channels: the number of channels in input tensor.
    :param out_channels: the number of channels in output tensor.
    :param stride: the convolution stride. Default: 1.
    :param dilation: the convolution dilation. Default: 1.
    :param bias: to add a learnable bias to the output or not. Default: True.
    :return: the 7x7 'same' padding convolution layer.
    """

    return SameConv1d(in_channels, out_channels, kernel_size=7, stride=stride, dilation=dilation, bias=bias)
