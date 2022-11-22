from collections.abc import Iterable
from itertools import repeat
import torch
import torch.nn as nn
import torch.nn.functional as Fn

__all__ = ['SameConv2d', 'conv2d_same_1x1', 'conv2d_same_3x3', 'conv2d_same_5x5', 'conv2d_same_7x7']


# class SameConv2d(nn.Conv2d):
#     """
#     Representation the `same` padding functionality from tensorflow for 2D convolution.
#     Note that the padding argument in the initializer doesn't do anything now.
#
#     Reference:
#     - https://gist.github.com/ujjwal-9/e13f163d8d59baa40e741fc387da67df
#     """
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#
#         :param x: the input tensor in shape of (B, C, F, T) or (C, F, T).
#         :return:
#         """
#
#         if x.dim() not in [3, 4]:
#             raise ValueError(f"The input tensor should be 3D or 4D tensor, but got dim = {x.dim()}.")
#
#         kernel_size = (self.weight.size(2), self.weight.size(3))
#         in_height, in_width = x.size(-2), x.size(-1)
#         out_height = int(np.ceil(float(in_height) / float(kernel_size[0])))
#         out_width  = int(np.ceil(float(in_width) / float(kernel_size[1])))
#
#         pad_along_height = max((out_height - 1) * self.stride[0] + kernel_size[0] - in_height, 0)
#         pad_along_width  = max((out_width - 1) * self.stride[1] + kernel_size[1] - in_width, 0)
#
#         pad_top    = pad_along_height // 2
#         pad_bottom = pad_along_height - pad_top
#         pad_left   = pad_along_width // 2
#         pad_right  = pad_along_width - pad_left
#
#         x = Fn.pad(x, (pad_left, pad_right, pad_top, pad_bottom))
#         return Fn.conv2d(x, weight=self.weight, bias=self.bias, stride=self.stride, padding=kernel_size, dilation=self.dilation, groups=self.groups)

def _ntuple(n):
    """Copied from PyTorch since it's not importable as an internal function

    https://github.com/pytorch/pytorch/blob/v1.10.0/torch/nn/modules/utils.py#L6
    """
    def parse(x):
        if isinstance(x, Iterable):
            return tuple(x)
        return tuple(repeat(x, n))

    return parse


_pair = _ntuple(2)


class SameConv2d(nn.Module):
    """Manual convolution with same padding

    Although PyTorch >= 1.10.0 supports ``padding='same'`` as a keyword argument,
    this does not export to CoreML as of coremltools 5.1.0, so we need to
    implement the internal torch logic manually. Currently the ``RuntimeError`` is

    "PyTorch convert function for op '_convolution_mode' not implemented"

    Also same padding is not supported for strided convolutions at the moment
    https://github.com/pytorch/pytorch/blob/v1.10.0/torch/nn/modules/conv.py#L93
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, **kwargs):
        """Wrap base convolution layer

        See official PyTorch documentation for parameter details
        https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        """
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            **kwargs)

        # Setup internal representations
        kernel_size_ = _pair(kernel_size)
        dilation_ = _pair(dilation)
        self._reversed_padding_repeated_twice = [0, 0] * len(kernel_size_)

        # Follow the logic from ``nn._ConvNd``
        # https://github.com/pytorch/pytorch/blob/v1.10.0/torch/nn/modules/conv.py#L116
        for d, k, i in zip(dilation_, kernel_size_, range(len(kernel_size_) - 1, -1, -1)):
            total_padding = d * (k - 1)
            left_pad = total_padding // 2
            self._reversed_padding_repeated_twice[2 * i] = left_pad
            self._reversed_padding_repeated_twice[2 * i + 1] = (
                    total_padding - left_pad)

    def forward(self, imgs):
        """Setup padding so same spatial dimensions are returned

        All shapes (input/output) are ``(N, C, W, H)`` convention

        :param torch.Tensor imgs:
        :return torch.Tensor:
        """
        padded = Fn.pad(imgs, self._reversed_padding_repeated_twice)
        return self.conv(padded)


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
