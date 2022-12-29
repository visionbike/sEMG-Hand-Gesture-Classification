from numpy.typing import NDArray
import numpy as np

__all__ = ['ulaw_norm', 'minmax_norm']


def ulaw_norm(x: NDArray, mu: float = 2048):
    """
    The mu-law normalization that scales the data points were distributed between 0 to mu value.
    :param x: the input signal.
    :param mu: the mu value. Default: 2048.
    :return: the normalized signal.
    """

    return np.sign(x) * (np.log(1 + mu * np.abs(x)) / np.log(1 + mu))


def minmax_norm(x: NDArray):
    """
    Min-max normalization.
    :param x: the input signal.
    :return: the normalized signal.
    """

    return (x - np.min(x, axis=0)) / (np.max(x, axis=0) - np.min(x, axis=0))


def z_score_norm(x: NDArray, mean: NDArray, std: NDArray):
    """
    Z-Score normalization.
    :param x: the input signal.
    :param mean: the mean value of the input.
    :param std: the standard deviation of the input.
    :return: the normalized signal.
    """

    return np.divide(np.subtract(x, mean, axis=0), std, axis=0)
