from typing import Callable, Any
from numpy.typing import NDArray
import random
import numpy as np
import torch
from data.preprocessing.utils import *

__all__ = ['NinaCompose', 'NinaToTensor', 'NinaMovingAverage', 'NinaRandomSNR', 'NinaTranspose']


class NinaCompose:
    """
    Compose class of transform functions for NinaPro datasets.
    """

    def __init__(self, transforms: [Any | list[Callable[..., Any]]]):
        """

        :param transforms: list of transform functions.
        """

        self.transforms = transforms

    def __call__(self, x: NDArray, y: NDArray):
        """

        :param x: the input signal.
        :param y: the input label.
        :return: the transformed signal and its label.
        """

        for transform in self.transforms:
            x, y = transform(x, y)
        return x, y


class NinaToTensor:
    """
    ToTensor class for NinaPro datasets.
    """

    def __call__(self, x: NDArray, y: NDArray):
        """

        :param x: the input signal.
        :param y: the input label.
        :return: the transformed signal and its label.
        """

        x = torch.from_numpy(x).float()
        y = torch.tensor(y).long()
        return x, y


class NinaRandomSNR:
    """
    Adding white noise class for NinaPro datasets.
    """

    def __init__(self, use_rest_label: bool = True):
        """

        :param use_rest_label: whether to use 'rest' label. Default: True.
        """

        # noise factors to sample from, outside the function
        # because this will be called millions of times
        self.use_rest_label = use_rest_label
        self.rlist = sum([[(x / 2) % 30] * ((x // 2) % 30) for x in range(120)], [])

    def _add_noise_snr(self, x: NDArray, snr: float = 2.) -> NDArray:
        """
        Function to add white noise to the signal.

        :param x: the input signal.
        :param snr: signal-to-noise factor. Default: 2.
        :return: the noise-added signal.
        """

        # convert freq to db
        x_db = np.log10((x ** 2).mean(axis=0)) * 10
        # avg noise in db
        noise_avg_db = x_db - snr
        # variance noise in db
        noise_var_db = 10 ** (noise_avg_db / 10)
        # make white noise
        noise = np.random.normal(0, np.sqrt(noise_var_db), x.shape)
        return x + noise

    def __call__(self, x: NDArray, y: NDArray):
        """

        :param x: the input signal.
        :param y: the input label.
        :return: the transformed signal and its label.
        """

        # get random signal-to-noise factor
        snr = random.choice(self.rlist)
        x = self._add_noise_snr(x, snr)
        return x, y


class NinaTranspose:
    """
    Transpose class for NinaPro class
    """

    def __call__(self, x: NDArray, y: NDArray):
        """

        :param x: the input signal.
        :param y: the input label.
        :return: the transformed signal and its label.
        """
        return x.T, y


class NinaMovingAverage:
    """
    Moving Average class for NinaPro class
    """

    def __init__(self, wsize: int = 3):
        """

        :param wsize: window size. Default: 3.
        """

        self.wsize = wsize

    def __call__(self, x: NDArray, y: NDArray):
        """

        :param x: the input signal.
        :param y: the input label.
        :return: the transformed signal and its label.
        """

        x = np.moveaxis(moving_average(x, self.wsize), -1, 0)
        return x, y
