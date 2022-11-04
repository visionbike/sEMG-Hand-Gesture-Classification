from typing import Optional
import numpy as np
import torch
import torch.nn as nn

__all__ = ['FocalLoss']


class FocalLoss(nn.Module):
    """
    The implementation of Focal Loss, which is proposed in Focal Loss for Dense Object Detection.
    Loss = - \alpha (1 - softmax(x)[class])^gamma \log(softmax(x)[class])
    The losses are averaged across observations for each mini-batch.

    Reference:
    - https://arxiv.org/abs/1708.02002.
    - https://github.com/AdeelH/pytorch-multi-class-focal-loss/blob/master/focal_loss.py
    """

    def __init__(self,
                 num_classes: int,
                 alpha: Optional[list | torch.Tensor],
                 gamma: float = 2.,
                 reduction: str = 'mean'):
        """

        :param num_classes: the number of classes.
        :param alpha: the class balanced weight. Default: None.
        :param gamma: the controlling constant. Default: 2.
        :param reduction: reduction option. Default: 'mean'.
        """

        # parameter checks
        if gamma < 0.:
            raise ValueError(f"Expected 'gamma' >= 0., but got 'gamma' = {gamma}.")
        if reduction not in ['mean', 'sum', 'none']:
            raise ValueError(f"Expected values: 'none'|'mean'|'sum', but got 'reduction' = {reduction}.")

        super(FocalLoss, self).__init__()

        self.num_classes = num_classes
        if isinstance(alpha, list):
            alpha = torch.from_numpy(np.array(alpha)).float()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.nll_loss = nn.NLLLoss(weight=alpha, reduction='none')

    def forward(self, pred: torch.Tensor, targ: torch.Tensor) -> torch.Tensor:
        """

        :param pred: the predicted tensor.
        :param targ: the target tensor.
        :return:
        """
        N, C = pred.shape[:2]
        if self.alpha.shape[0] != C:
            raise RuntimeError("The length not equal to the number of classes.")
        if pred.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.to(pred.device)

        # compute weighted cross entropy term: -alpha * log(pt)
        # (alpha is already part of self.nll_loss)
        log_p = torch.log_softmax(pred, dim=-1)
        ce = self.nll_loss(log_p, targ)

        # get true class column from each row
        all_rows = torch.arange(len(pred))
        log_pt = log_p[all_rows, targ]

        # compute focal term: (1 - pt)^gamma
        pt = torch.exp(log_pt)
        ft = (1 - pt) ** self.gamma

        # compute loss: -alpha * ((1. - pt) ^ gamma) * log(pt)
        loss = ft * ce

        # do reduction
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss
