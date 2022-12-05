import multiprocessing as multproc
from functools import partial
from numpy.typing import NDArray
import numpy as np
import scipy.signal as signal

__all__ = ['butter_low', 'butter_high', 'butter_band', 'moving_average']


def butter_low(x: NDArray, cutoff: float = 2., fs: float = 200., order: int = 4, zero_phase: bool = True) -> NDArray:
    """
    Butterworth lowpass filter

    :param x: the input signal in shape of (N, C).
    :param cutoff: the cut-off frequency. Default: 2.
    :param fs: the sampling rate. Default: 200.
    :param order: the order of the filter. Default: 4.
    :param zero_phase: whether to apply zero-phase filter (for offline data). Default: True.
    :return: the output signal.
    """

    nyq = 0.5 * fs  # nyquist frequency
    cutoff = cutoff / nyq
    [b, a] = signal.butter(N=order, Wn=cutoff, btype='low', analog=False)
    return signal.filtfilt(b, a, x, axis=0, padtype='odd', padlen=3 * (max(len(b), len(a)) - 1)) if zero_phase else signal.lfilter(b, a, x, axis=0)


def butter_high(x: NDArray, cutoff: float = 2., fs: float = 200., order: int = 4, zero_phase=True) -> NDArray:
    """
    Butterworth highpass filter

    :param x: the input signal in shape of (N, C).
    :param cutoff: the cut-off frequency. Default: 2.
    :param fs: the sampling rate. Default: 200.
    :param order: the order of the filter. Default: 4.
    :param zero_phase: whether to apply zero-phase filter (for offline data). Default: True.
    :return: the output signal.
    """

    nyq = 0.5 * fs  # nyquist frequency
    cutoff = cutoff / nyq
    [b, a] = signal.butter(N=order, Wn=cutoff, btype='high', analog=False)
    return signal.filtfilt(b, a, x, axis=0, padtype='odd', padlen=3 * (max(len(b), len(a)) - 1)) if zero_phase else signal.lfilter(b, a, x, axis=0)


def butter_band(x: NDArray, lcut: float = 2., hcut: float = 5., fs: float = 200., order: int = 4, zero_phase: bool = True) -> NDArray:
    """
    Butterworth bandpass filter

    :param x: the input signal in shape of (N, C).
    :param lcut: the low-cut frequency. Default: 2.
    :param hcut: the high-cut frequency. Default: 5.
    :param fs: the sampling rate. Default: 200.
    :param order: the order of the filter. Default: 4.
    :param zero_phase: whether to apply zero-phase filter (for offline data). Default: True.
    :return:
    """

    nyq = 0.5 * fs  # nyquist frequency
    lcut = lcut / nyq
    hcut = hcut / nyq
    [b, a] = signal.butter(N=order, Wn=[lcut, hcut], btype='band', analog=False)
    return signal.filtfilt(b, a, x, axis=0, padtype='odd', padlen=3 * (max(len(b), len(a)) - 1)) if zero_phase else signal.lfilter(b, a, x, axis=0)


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
