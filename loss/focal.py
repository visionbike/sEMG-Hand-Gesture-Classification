from typing import Optional, Union
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
    - https://github.com/pytorch/vision/issues/3250
    """

    def __init__(self,
                 num_classes: int,
                 alpha: Optional[list[float]] = None,
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
            raise ValueError(f"Expected float 'gamma' >= 0., but got 'gamma' = {gamma}.")
        if reduction not in ['mean', 'sum', 'none']:
            raise ValueError(f"Expected values: 'none'|'mean'|'sum', but got 'reduction' = {reduction}.")

        super(FocalLoss, self).__init__()

        self.num_classes = num_classes
        self.alpha = torch.tensor(alpha, requires_grad=False).float() if (alpha is not None) else alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ce = nn.CrossEntropyLoss(reduction='none')
        self.eps = 1e-9

    def forward(self, predicts: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """

        :param predicts: the predicted logit tensor.
        :param targets: the target tensor.
        :return:
        """
        if self.alpha is not None:
            self.alpha = self.alpha.to(predicts.device)
        # compute ce loss
        ce_loss = self.ce(predicts + self.eps, targets)
        # compute pt term
        pt = torch.exp(-ce_loss)
        # compute focal losses
        if self.alpha is None:
            focal_loss = ce_loss * ((1 - pt) ** self.gamma)
        else:
            focal_loss = self.alpha[targets] * ce_loss * ((1 - pt) ** self.gamma)
            focal_loss / self.alpha.sum()
        # do reduction
        if self.reduction == 'mean':
            focal_loss = focal_loss.mean()
        elif self.reduction == 'sum':
            focal_loss = focal_loss.sum()
        return focal_loss
