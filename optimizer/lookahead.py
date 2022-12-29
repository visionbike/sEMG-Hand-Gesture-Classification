from typing import Callable, Optional
from collections import defaultdict
import torch
from torch.optim.optimizer import Optimizer

__all__ = ['Lookahead']


class Lookahead(Optimizer):
    """
    Lookahead optimizer.

    Reference:
    - https://arxiv.org/abs/1907.08610
    - https://github.com/alphadl/lookahead.pytorch
    - https://github.com/mgrankin/over9000/blob/master/lookahead.py
    """

    def __init__(self,
                 optim_base,
                 alpha: float = 0.5,
                 k: int = 6):
        """

        :param optim_base:
        :param alpha:
        :param k:
        """

        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"Invalid slow update rate: {alpha}.")
        if not (1 <= k):
            raise ValueError(f"Invalid lookahead steps: {k}.")

        defaults = dict(alpha_lookahead=alpha, k_lookahead=k, step_lookahead=0)
        self.optim_base = optim_base
        self.param_groups = self.optim_base.param_groups
        self.defaults = optim_base.defaults
        self.defaults.update(defaults)
        self.state = defaultdict(dict)
        # manually add default to the param_groups
        for name, default in defaults.items():
            for group in self.param_groups:
                group.setdefault(name, default)

    def update_slow(self, group: dict):
        for p_fast in group['params']:
            if p_fast.grad is None:
                continue
            state = self.state[p_fast]
            if 'buffer_slow' not in state:
                state['buffer_slow'] = torch.empty_like(p_fast.data)
                state['buffer_slow'].copy_(p_fast.data)
            state_slow = state['buffer_slow']
            state_slow.add_(group['alpha_lookahead'], p_fast.data - state_slow)

    def sync_lookahead(self):
        for group in self.param_groups:
            self.update_slow(group)

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        loss = self.optim_base.step(closure)
        for group in self.param_groups:
            group['step_lookahead'] += 1
            if group['step_lookahead'] % group['k_lookahead'] == 0:
                self.update_slow(group)
        return loss

    def state_dict(self) -> dict:
        state_fast = self.optim_base.state_dict()
        state_slow = {
            (id(k) if isinstance(k, torch.Tensor) else k): v for k, v in self.state.items()
        }
        state_fast = state_fast['state']
        param_groups = state_fast['param_groups']
        return dict(state=state_fast, state_slow=state_slow, param_groups=param_groups)

    def load_state_dict(self, state_dict: dict) -> None:
        state_fast = dict(state=state_dict['state'], param_groups=state_dict['param_groups'])
        self.optim_base.load_state_dict(state_fast)

        # to restore the slow state, but share param_groups reference
        # with base_optimizer. This is a bit redundant but least code
        is_state_slow_new = False
        if 'state_slow' not in state_dict:
            print(f"Loading state_dict from optimizer without Lookahead applied.")
            state_dict['state_slow'] = defaultdict(dict)
            is_state_slow_new = True
        state_slow = dict(state=state_dict['state_slow'], param_groups=state_dict['param_groups'])

        super(Lookahead, self).load_state_dict(state_slow)
        self.param_groups = self.optim_base.param_groups    # make both ref same container
        if is_state_slow_new:
            # reapply defaults to catch missing lookahead specific ones
            for name, default in self.defaults.items():
                for group in self.param_groups:
                    group.setdefault(name, default)
