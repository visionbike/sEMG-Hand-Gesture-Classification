from numpy.typing import NDArray
import numpy as np
import scipy.signal as signal

__all__ = ['butter_low', 'butter_high', 'butter_band', 'moving_average']


def butter_low(x: NDArray, cutoff: float = 2., fs: float = 200., order: int = 4, zero_phase: bool = True) -> NDArray:
    """
    Butterworth IIR low-pass filter.

    :param x: the input signal in shape of (N, C).
    :param cutoff: the cut-off frequency. Default: 2.
    :param fs: the sampling rate. Default: 200.
    :param order: the order of the filter. Default: 4.
    :param zero_phase: whether to apply zero-phase filter (for offline data). Default: True.
    :return: the filtered signal.
    """

    nyq = 0.5 * fs  # nyquist frequency
    cutoff = cutoff / nyq
    sos = signal.butter(N=order, Wn=cutoff, btype='lowpass', analog=False, output='sos')
    z = signal.sosfiltfilt(sos, x, axis=0) if zero_phase else signal.sosfilt(sos, x, axis=0)
    return z


def butter_high(x: NDArray, cutoff: float = 2., fs: float = 200., order: int = 4, zero_phase=True) -> NDArray:
    """
    Butterworth IIR high-pass filter.

    :param x: the input signal in shape of (N, C).
    :param cutoff: the cut-off frequency. Default: 2.
    :param fs: the sampling rate. Default: 200.
    :param order: the order of the filter. Default: 4.
    :param zero_phase: whether to apply zero-phase filter (for offline data). Default: True.
    :return: the output signal.
    """

    nyq = 0.5 * fs  # nyquist frequency
    cutoff = cutoff / nyq
    sos = signal.butter(N=order, Wn=cutoff, btype='highpass', analog=False, output='sos')
    z = signal.sosfiltfilt(sos, x, axis=0) if zero_phase else signal.sosfilt(sos, x, axis=0)
    return z


def butter_band(x: NDArray, lcut: float = 2., hcut: float = 5., fs: float = 200., order: int = 4, zero_phase: bool = True) -> NDArray:
    """
    Butterworth IIR band-pass filter.

    :param x: the input signal in shape of (N, C).
    :param lcut: the low-cut frequency. Default: 2.
    :param hcut: the high-cut frequency. Default: 5.
    :param fs: the sampling rate. Default: 200.
    :param order: the order of the filter. Default: 4.
    :param zero_phase: whether to apply zero-phase filter (for offline data). Default: True.
    :return: the filtered signal.
    """

    nyq = 0.5 * fs  # nyquist frequency
    lcut = lcut / nyq
    hcut = hcut / nyq
    sos = signal.butter(N=order, Wn=[lcut, hcut], btype='bandpass', analog=False, output='sos')
    z = signal.sosfiltfilt(sos, x, axis=0) if zero_phase else signal.sosfilt(sos, x, axis=0)
    return z


def moving_average(x: NDArray, ksize: int = 3) -> NDArray:
    """
    Moving average filter

    :param x: the input signal in shape of (C, N).
    :param ksize: the kernel size. Default: 3.
    :return:  the filtered signal in shape of (C, N - ksize + 1).
    """

    w = np.ones(ksize) / ksize
    z = np.vstack([np.convolve(x[:, i], w, mode='valid') for i in range(x.shape[-1])])
    return z
