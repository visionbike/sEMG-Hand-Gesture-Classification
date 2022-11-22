import math
import torch
from torch.optim.lr_scheduler import _LRScheduler

__all__ = ['CosineAnnealingWarmupRestartRL']


class CosineAnnealingWarmupRestartRL(_LRScheduler):
    """
    Reference:
    - https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup
    """

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 T_0: int,
                 T_mult: float = 1.,
                 eta_max: float = 0.1,
                 eta_min: float = 0.001,
                 T_warmup: int = 0,
                 gamma: float = 1.,
                 last_epoch: int = -1):
        """

        :param optimizer: the wrapped optimizer.
        :param T_0: the first cycle step size.
        :param T_mult: the cycle step magnification. Default: 1.
        :param eta_max: the maximum learning rate. Default: 0.1.
        :param eta_min: the minimum learning rate. Default: 0.001.
        :param T_warmup: the linear warmup step. Default: 0.
        :param gamma: the decrease rate of the maximum learning by cycle. Default: 1.
        :param last_epoch: the index of the last epoch. Default: Default: -1.
        """

        if T_warmup >= T_0:
            raise ValueError(f"Expected 'T_warmup' < 'T_0', but got 'T_0' >= 'T_warmup'.")

        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_max_base = eta_max
        self.eta_max = eta_max
        self.eta_min = eta_min
        self.T_warmup = T_warmup
        self.gamma = gamma

        self.T_cur = T_0
        self.T_i = 0  # cycle count
        self.step_in_cycle = last_epoch  # step size of the current cycle

        super(CosineAnnealingWarmupRestartRL, self).__init__(optimizer, last_epoch)

        # set learning rate min_lr
        self.init_lr()

    def init_lr(self):
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.eta_min
            self.base_lrs.append(self.eta_min)

    def get_lr(self):
        if self.step_in_cycle == -1:
            return self.base_lrs
        elif self.step_in_cycle < self.T_warmup:
            return [
                (self.eta_max - base_lr) * self.step_in_cycle / self.T_warmup + base_lr for base_lr in self.base_lrs
            ]
        else:
            return [
                base_lr + (self.eta_max - base_lr) * (1 + math.cos(math.pi * (self.step_in_cycle - self.T_warmup) / (self.T_cur - self.T_warmup))) / 2 for base_lr in self.base_lrs
            ]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            if self.step_in_cycle >= self.T_cur:
                self.T_i += 1
                self.step_in_cycle = self.step_in_cycle - self.T_cur
                self.T_cur = int((self.T_cur - self.T_warmup) * self.T_mult) + self.T_warmup
        else:
            if epoch >= self.T_0:
                if self.T_mult == 1.:
                    self.step_in_cycle = epoch % self.T_0
                    self.T_i = epoch // self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.T_i = n
                    self.step_in_cycle = epoch - int(self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1))
                    self.T_cur = self.T_0 * (self.T_mult ** n)
            else:
                self.T_cur = self.T_0
                self.step_in_cycle = epoch

        self.eta_max = self.eta_max_base * (self.gamma ** self.T_i)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr
