from typing import Optional, Callable
import math
import torch
from torch.optim.optimizer import Optimizer

__all__ = ['AdamW']


class AdamW(Optimizer):
    """
    AdamW optimizer.

    Reference:
    - https://github.com/LiyuanLucasLiu/RAdam/blob/master/radam.py
    - https://github.com/mgrankin/over9000/blob/master/radam.py
    """

    def __init__(self,
                 params,
                 lr: float = 1e-3,
                 betas: tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8,
                 weight_decay: float = 0.0,
                 warmup: int = 0):
        """

        :param params:
        :param lr:
        :param betas:
        :param eps:
        :param weight_decay:
        :param warmup:
        """

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, warmup=warmup)
        super(AdamW, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AdamW, self).__setstate__(state)

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError(f"AdamW does not support sparse gradients, please consider SparseAdam instead!")

                p_data_fp32 = p.data.float()
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                state['step'] += 1
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                denom = exp_avg_sq.sqrt().add_(group['eps'])
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                if group['warmup'] > state['step']:
                    lr_scheduled = state['step'] * group['lr'] / group['warmup'] + 1e-8
                else:
                    lr_scheduled = group['lr']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1
                if group['weight_decay'] != 0:
                    p_data_fp32.add_(-group['weight_decay'] * lr_scheduled, p_data_fp32)
                p_data_fp32.addcdiv_(-step_size, exp_avg, denom)
                p.data.copy_(p_data_fp32)
        return loss
