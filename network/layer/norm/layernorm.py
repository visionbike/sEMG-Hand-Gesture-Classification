from typing import Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as Fn

__all__ = ['LayerNorm']


class LayerNorm(nn.Module):
    """
    LayerNorm for timeseries that supports two data formats: `channels_last` (default) or `channels_first`.
    The ordering of the dimensions in the inputs.
    `channel_last` corresponds to tensor of shape (B, *, C).
    `channel_first` corresponds to tensor of shape (B, C, *).

    Reference:
    -   https://github.com/pytorch/pytorch/issues/76012
    -   https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py
    """

    def __init__(self,
                 normalized_shape: Optional[Union[int, list[int], torch.Size]],
                 eps: float = 1e-6,
                 order: str = 'last'):
        """

        :param normalized_shape: the normalization input shape.
        :param eps: the epsilon value. Default: 1e-6.
        :param order: order of the input dimensions, includes
            'last' corresponds to tensor in shape (B, *, C).
            'first' corresponds to tensor in shape (B, C, *).
            Default: 'last'.
        """

        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.order = order
        if self.order not in ['first', 'last']:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        :param x: the input tensor in shape of (B, *, C) or (B, C, *).
        :return:
        """

        if self.order == 'last':
            return Fn.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.order == 'first':
            mean = x.mean(1, keepdim=True)
            var = (x - mean).pow(2).mean(1, keepdim=True)
            x = (x - mean) / torch.sqrt(var + self.eps)
            x = self.weight[:, None] * x + self.bias[:, None]
            return x
