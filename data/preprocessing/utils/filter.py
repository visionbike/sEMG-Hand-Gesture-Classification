import gc
import os
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from numpy.typing import NDArray
from numba import njit
import numpy as np
import scipy.signal as signal

__all__ = ['process_butter_low', 'process_butter_high', 'process_butter_band', 'butter_low', 'butter_high', 'butter_band', 'moving_average']


def process_butter_low(x: NDArray, cutoff=2., fs=200., order=4) -> NDArray:
    """
    Multiprocessing function to process butterworth lowpass filter for multiple signals.

    :param x: the input signal list.
    :param cutoff: the cut-off frequency. Default: 2.
    :param fs: the sampling rate. Default: 200.
    :param order: the order of the filter. Default: 4.
    :return: the filtered signals.
    """

    inn = [x[i] for i in range(x.shape[0])]
    with ThreadPoolExecutor(max_workers=min(32, os.cpu_count() + 4)) as executor:
        z = [r for r in executor.map(partial(butter_low, cutoff=cutoff, fs=fs, order=order), inn)]
    del inn
    gc.collect()
    return np.asarray(z)


def process_butter_high(x: NDArray, cutoff=2., fs=200., order=4) -> NDArray:
    """
    Multiprocessing function to process butterworth highpass filter for multiple signals.

    :param x: the input signal list.
    :param cutoff: the cut-off frequency. Default: 2.
    :param fs: the sampling rate. Default: 200.
    :param order: the order of the filter. Default: 4.
    :return: the filtered signals.
    """

    inn = [x[i] for i in range(x.shape[0])]
    with ThreadPoolExecutor(max_workers=min(32, os.cpu_count() + 4)) as executor:
        z = [r for r in executor.map(partial(butter_high, cutoff=cutoff, fs=fs, order=order), inn)]
    del inn
    gc.collect()
    return np.asarray(z)


def process_butter_band(x: NDArray, lcut=5., hcut=99., fs=200., order=4) -> NDArray:
    """
    Multiprocessing function to process butterworth bandpass filter for multiple signals.

    :param x: the input signal list.
    :param lcut: the low-cut frequency. Default: 5.
    :param hcut: the high-cut frequency. Default: 99.
    :param fs: the sampling rate. Default: 200.
    :param order: the order of the filter. Default: 4.
    :return: the filtered signals.
    """

    inn = [x[i] for i in range(x.shape[0])]
    with ThreadPoolExecutor(max_workers=min(32, os.cpu_count() + 4)) as executor:
        z = [r for r in executor.map(partial(butter_low, lcut=lcut, hcut=hcut, fs=fs, order=order), inn)]
    del inn
    gc.collect()
    return np.asarray(z)


def butter_low(x: NDArray, cutoff: float = 2., fs: float = 200., order: int = 4) -> NDArray:
    """
    Butterworth lowpass filter

    :param x: the input signal.
    :param cutoff: the cut-off frequency. Default: 2.
    :param fs: the sampling rate. Default: 200.
    :param order: the order of the filter. Default: 4.
    :return: the output signal.
    """

    nyq = 0.5 * fs  # nyquist frequency
    cutoff = cutoff / nyq
    sos = signal.butter(N=order, Wn=cutoff, btype='low', analog=False, output='sos')
    z = signal.sosfilt(sos, x)
    return z


def butter_high(x: NDArray, cutoff: float = 2., fs: float = 200., order: int = 4) -> NDArray:
    """
    Butterworth highpass filter

    :param x: the input signal.
    :param cutoff: the cut-off frequency. Default: 2.
    :param fs: the sampling rate. Default: 200.
    :param order: the order of the filter. Default: 4.
    :return: the output signal.
    """

    nyq = 0.5 * fs  # nyquist frequency
    cutoff = cutoff / nyq
    sos = signal.butter(N=order, Wn=cutoff, btype='high', analog=False, output='sos')
    z = signal.sosfilt(sos, x)
    return z


def butter_band(x: NDArray, lcut: float = 2., hcut: float = 5., fs: float = 200., order: int = 4) -> NDArray:
    """
    Butterworth bandpass filter

    :param x: the input signal.
    :param lcut: the low-cut frequency. Default: 2.
    :param hcut: the high-cut frequency. Default: 5.
    :param fs: the sampling rate. Default: 200.
    :param order: the order of the filter. Default: 4.
    :return:
    """

    nyq = 0.5 * fs  # nyquist frequency
    lcut = lcut / nyq
    hcut = hcut / nyq
    sos = signal.butter(N=order, Wn=[lcut, hcut], btype='band', analog=False, output='sos')
    z = signal.sosfilt(sos, x)
    return z


@njit
def moving_average(x: NDArray, ksize: int = 3) -> NDArray:
    """
    Moving average filter

    :param x: the input signal in shape of (C, N).
    :param ksize: the kernel size. Default: 3.
    :return:  the filtered signal in shape of (C, N').
    """

    w = np.ones(ksize) / ksize
    z = np.vstack([np.convolve(x[:, i], w, mode='valid') for i in range(x.shape[-1])])
    return z
