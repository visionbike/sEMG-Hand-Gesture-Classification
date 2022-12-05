from numpy.typing import NDArray
import numpy as np

__all__ = ['u_law_norm', 'minmax_norm']


def u_law_norm(x: NDArray, mu: float = 2048):
    """
    The mu-law normalization that scales the data points were distributed between 0 to mu.
    :param x: the input signal.
    :param mu: the mu value. Default: 2048
    :return:
    """

    return np.sign(x) * (np.log(1 + mu * np.abs(x)) / np.log(1 + mu))


def minmax_norm(x: NDArray):
    """
    Min-max normalization.
    :param x: the input signal.
    :return:
    """

    return (x - np.min(x, axis=0)) / (np.max(x, axis=0) - np.min(x, axis=0))
