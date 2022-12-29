from typing import Optional, Callable
from torch.optim.optimizer import Optimizer
from .radam import RAdam
from .lookahead import Lookahead


class Ranger(Optimizer):
    """
    Ranger (RAdam + Lookahead) optimizer.

    Reference:
    - https://github.com/mgrankin/over9000/blob/master/ranger.py
    -
    """

    def __init__(self,
                 params,
                 lr: float = 1e-3,
                 betas: tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8,
                 weight_decay: float = 0.0,
                 n_sma_threshold: int  = 5,
                 alpha: float = 0.5,
                 k: int = 6):
        """

        :param params:
        :param lr:
        :param betas:
        :param eps:
        :param weight_decay:
        :param n_sma_threshold:
        :param alpha:
        :param k:
        """

        radam = RAdam(params, lr, betas, eps, weight_decay, n_sma_threshold)
        self.lookahead = Lookahead(radam, alpha, k)

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        loss = self.lookahead.step(closure)
        return loss

    def state_dict(self) -> dict:
        state_dict = self.lookahead.state_dict()
        return state_dict

    def load_state_dict(self, state_dict: dict) -> None:
        self.lookahead.load_state_dict(state_dict)

    def sync_ranger(self):
        self.lookahead.sync_lookahead()
