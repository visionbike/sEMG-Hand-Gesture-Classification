import torch
import torch.nn as nn

__all__ = ['SimpleAttention']


class SimpleAttention(nn.Module):
    """
    The Simple Attention implementation from 10.48550/arXiv.2006.03645

    Reference:
    - https://doi.org/10.48550/arXiv.2006.03645
    - https://github.com/josephsdavid/semg_repro/blob/master/repro.py
    """

    def __init__(self, in_channels: int):
        """

        :param in_channels: the number of channels in input tensor.
        """

        super(SimpleAttention, self).__init__()

        self.linear = nn.Linear(in_channels, in_channels, bias=True)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """

        :param x: the input tensor in shape of (B, C, N).
        :return: the output attention map in shape of (B, C, N).
        """

        if x.dim() != 3:
            raise ValueError(f"The input tensor should be in shape of (B, C, N), but got dim = {x.dim()}.")

        # (B, C, N) -> (B, N, C)
        z = x.permute(0, 2, 1).contiguous()
        # get attention probability along sample-axis
        att_weights = self.linear(z)
        att_score = torch.softmax(att_weights, dim=-1)
        # (B, N, C) -> (B, C, N)
        att_score = att_score.permute(0, 2, 1).contiguous()
        # get attention map (B, C, N)
        z = x * att_score
        return z
