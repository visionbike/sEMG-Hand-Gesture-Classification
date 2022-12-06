from typing import Union
import math
from scipy.linalg import hadamard
import torch
import torch.nn as nn

__all__ = ['init_zero_linear', 'init_zero_conv1d', 'init_zero_conv2d']


def init_zero_linear(w: Union[nn.Parameter, torch.Tensor]):
    """
    ZerO initialization for fully connected layer (algorithm I in the paper).

    Reference:
    - https://arxiv.org/pdf/2110.12661.pdf

    :param w: the input linear layer weight parameter.
    :return:
    """

    l_out, l_in = w.data.size(0), w.data.size(1)

    if l_out <= l_in:
        w.data = nn.init.eye_(torch.empty(l_out, l_in))
    else:
        l_clog = math.ceil(math.log2(l_out))
        l_pow = 2 ** l_clog
        w.data = nn.init.eye_(torch.empty(l_out, l_pow)) @ (torch.tensor(hadamard(l_pow)).float() / (2 ** (l_clog / 2))) @ nn.init.eye_(torch.empty(l_pow, l_in))


def init_zero_conv1d(w: Union[nn.Parameter, torch.Tensor]):
    """
    ZerO initialization for 1D convolutional layer (Algorithm II in the paper).

    Reference:
    - https://arxiv.org/pdf/2110.12661.pdf

    :param w: the input 1D convolutional layer weight parameter.
    :return:
    """

    c_out, c_in, k = w.data.size(0), w.data.size(1), w.data.size(2)
    if (k % 2) == 0:
        raise ValueError(f"Expected odd value of kernel, but got {k}.")
    n = k // 2

    w.data = nn.init.zeros_(torch.empty(c_out, c_in, k))
    if c_out <= c_in:
        w.data[:, :, n] = nn.init.eye_(torch.empty(c_out, c_in))
    else:
        c_clog = math.ceil(math.log2(c_out))
        c_pow = 2 ** c_clog
        w.data[:, :, n] = nn.init.eye_(torch.empty(c_out, c_pow)) @ (torch.tensor(hadamard(c_pow)).float() / (2 ** (c_clog / 2))) @ nn.init.eye_(torch.empty(c_pow, c_in))


def init_zero_conv2d(w: Union[nn.Parameter, torch.Tensor]):
    """
    ZerO initialization for 2D convolutional layer (Algorithm II in the paper).

    Reference:
    - https://arxiv.org/pdf/2110.12661.pdf

    :param w: the input convolutional layer weight parameter.
    :return:
    """

    c_out, c_in, k_h, k_w = w.data.size(0), w.data.size(1), w.data.size(2), w.data.size(3)
    if (k_h % 2) == 0 or (k_w % 2) == 0:
        raise ValueError(f"Expected odd value of kernel, but got ({k_h}, {k_w}).")
    n_h, n_w = k_h // 2, k_w // 2

    w.data = nn.init.zeros_(torch.empty(c_out, c_in, k_h, k_w))
    if c_out <= c_in:
        w.data[:, :, n_h, n_w] = nn.init.eye_(torch.empty(c_out, c_in))
    else:
        c_clog = math.ceil(math.log2(c_out))
        c_pow = 2 ** c_clog
        w.data[:, :, n_h, n_w] = nn.init.eye_(torch.empty(c_out, c_pow)) @ (torch.tensor(hadamard(c_pow)).float() / (2 ** (c_clog / 2))) @ nn.init.eye_(torch.empty(c_pow, c_in))
