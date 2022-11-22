from typing import Optional
import math
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

__all__ = ['CosineAnnealingOneCycleRL']


class CosineAnnealingOneCycleRL(_LRScheduler):
    """
    Cosine Annealing LR Scheduler, applying after T_start epochs.
    It has been proposed in `SGDR: Stochastic Gradient Descent with Warm Restarts`.

    Reference:
    - https://arxiv.org/abs/1608.03983.
    - https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup/blob/master/cosine_annealing_warmup/scheduler.py
    """
    def __init__(self,
                 optimizer: Optimizer,
                 T_max: int,
                 T_start: int = 0,
                 eta_max: float = 1e-3,
                 eta_min: float = 1e-5,
                 last_epoch: float = -1):
        """

        :param optimizer: the wrapped optimizer.
        :param T_max: the number of iterations for cosine annealing.
        :param T_start: the number of iterations before cosine annealing. Default: 0.
        :param eta_max: the maximum learning rate. Default: 1e-3.
        :param eta_min: the minimum learning rate. Default: 1e-5.
        :param last_epoch:
        """

        self.T_max = T_max
        self.eta_max = eta_max
        self.eta_min = eta_min
        self.T_start = T_start
        self.init_lrs = []

        super(CosineAnnealingOneCycleRL, self).__init__(optimizer, last_epoch)
        self.init_lr()

    def init_lr(self):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.eta_max
            self.init_lrs.append(self.eta_max)

    def get_lr(self):
        if self.last_epoch < self.T_start:
            return self.init_lrs
        elif self.T_start <= self.last_epoch < (self.T_start + self.T_max):
            return [
                (self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * (self.last_epoch - self.T_start) / self.T_max)) / 2) for base_lr in self.base_lrs
            ]
        return [self.eta_min if (param_group['lr'] < self.eta_min) else param_group['lr'] for param_group in self.optimizer.param_groups]


# class CosineAnnealingStartLR(_LRScheduler):
#     """
#     Cosine Annealing LR Scheduler, applying after T_start epochs.
#     It has been proposed in `SGDR: Stochastic Gradient Descent with Warm Restarts`.
#
#     Reference:
#     - https://arxiv.org/abs/1608.03983.
#     - https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup/blob/master/cosine_annealing_warmup/scheduler.py
#     """
#
#     def __init__(self,
#                  optimizer: Optimizer,
#                  T_0: int,
#                  T_start: int = 0,
#                  T_mult: float = 1.,
#                  eta_min: float = 1e-8,
#                  gamma: float = 1.,
#                  last_epoch=-1,
#                  verbose=False):
#         """
#
#         :param optimizer: the wrapped optimizer.
#         :param T_0: number of iterations for the first cycle.
#         :param T_start: number of iteration before starting the scheduler. Default: 0.
#         :param T_mult: a factor to increase cycle step. Default: 1.
#         :param eta_min: minimum learning rate. Default: 0.
#         :param gamma: decreasing rate of base learning rate. Default: 0.
#         :param last_epoch: the index of last epoch. Default: -1.
#         :param verbose: If True, prints a message to stdout for each update. Default: False.
#         """
#
#         # parameter checks
#         self.last_epoch = None
#         if T_0 <= 0 or not isinstance(T_0, int):
#             raise ValueError(f"Expected 'T_0' positive integer type, but got 'T_0' = {T_0}.")
#         if T_start < 0 or not isinstance(T_start, int):
#             raise ValueError(f"Expected 'T_start' positive integer type, but got 'T_start' = {T_start}.")
#         if T_start >= T_0:
#             raise ValueError(f"Expected 'T_start' < 'T_0', but got 'T_start' >= 'T_0'.")
#         if T_mult < 1. or not isinstance(T_mult, float):
#             raise ValueError(f"Expected float 'T_mult' >= 1., but got 'T_mult' = {T_mult}.")
#         if not (0. < gamma <= 1.) or not isinstance(gamma, float):
#             raise ValueError(f"Expected float 0. < 'gamma' <= 1., but got 'gamma' = {gamma}.")
#
#         self.T_0 = T_0              # the first cycle step size
#         self.T_start = T_start
#         self.T_mult = T_mult
#         self.eta_min = eta_min
#         self.gamma = gamma
#         self.T_idx = 0              # the cycle index
#         self.T_cur = T_0            # the current cycle step size
#         self.i_cur = last_epoch     # the current step
#         super(CosineAnnealingStartLR, self).__init__(optimizer, last_epoch, verbose)
#
#         # get base_lrs
#         self.base_lrs = []
#         for param_group in self.optimizer.param_groups:
#             self.base_lrs.append(param_group['lr'])
#         self.cur_base_lrs = self.base_lrs.copy()
#
#     def get_lr(self) -> list:
#         if (self.i_cur == -1) or (self.i_cur < self.T_start - 1):
#             return self.cur_base_lrs
#         else:
#             return [
#                 self.eta_min +
#                 (base_lr - self.eta_min) *
#                 (1 + math.cos(math.pi * (self.i_cur - self.T_start) / (self.T_cur - self.T_start))) / 2
#                 for base_lr in self.cur_base_lrs
#             ]
#
#     def step(self, epoch: Optional[int] = None) -> None:
#         if epoch is None:
#             epoch = self.last_epoch + 1
#             self.i_cur += 1
#             if self.i_cur >= self.T_cur:
#                 self.T_idx += 1
#                 self.i_cur = self.i_cur - self.T_cur
#                 self.T_cur = int((self.T_cur - self.T_start) * self.T_mult) + self.T_start
#         else:
#             if epoch >= (self.T_0 + self.T_start):
#                 if self.T_mult == 1.:
#                     self.i_cur = epoch % self.T_0
#                     self.T_idx = epoch // self.T_0
#                 else:
#                     n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
#                     self.T_idx = n
#                     self.i_cur = epoch - int(self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1))
#                     self.T_cur = self.T_0 * self.T_mult ** n
#             else:
#                 self.T_cur = self.T_0
#                 self.i_cur = epoch
#         # update base lr and current lr
#         self.cur_base_lrs = [base_lr * (self.gamma ** self.T_idx) for base_lr in self.base_lrs]
#         self.last_epoch = math.floor(epoch)
#         for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
#             param_group['lr'] = lr
