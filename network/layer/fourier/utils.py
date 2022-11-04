from typing import Optional, Any, Union
from numpy import ndarray
from numpy.typing import NDArray
import numpy as np
from scipy.signal import get_window
import torch
import torch.nn.functional as Fn

__all__ = ['make_pad_center', 'window_sumsquare', 'overlap_add_filter', 'extend_fbins', 'broadcast_dim', 'reshape_dim', 'create_fourier_kernels']


def make_pad_center(x: np.ndarray, size: int, axis: int = -1, **kwargs) -> NDArray:
    """
    Wrapper for np.pad to automatically center an array prior to padding.
    The function to make STFT as same as librosa.

    :param x: the array to be padded and centered.
    :param size: the length to pad, size >= len(x).
    :param axis: the axis along which to pad and center the data.
    :param kwargs: additional keyword arguments passed to `np.pad()`.
    :return: the centered and padded to length `size` along specified axis.
    """

    kwargs.setdefault('mode', 'constant')

    n = x.shape[axis]
    pad_left = int((size - n) // 2)
    lens = [(0, 0)] * x.ndim
    lens[axis] = (pad_left, int(size - n - pad_left))
    if pad_left < 0:
        raise ValueError(f"Target size ({size}) must be at least input size ({n})")
    return np.pad(x, pad_width=lens, **kwargs)


def window_sumsquare(win: torch.Tensor, num_frames: int, win_hop: int, power: int = 2) -> torch.Tensor:
    """
    Compute the sum-square envelope of a window at the given hop length.

    :param win: the input window.
    :param num_frames: the number of frames.
    :param win_hop: the hop (or stride) size.
    :param power: the power value. Default: 2.
    :return:
    """

    win_stacks = win.unsqueeze(-1).repeat((1, num_frames)).unsqueeze(0)
    win_len = win_stacks.shape[1]
    out_len = win_len + win_hop * (num_frames - 1)  # window length + stride * (frames - 1)

    return Fn.fold(win_stacks ** power, (1, out_len), kernel_size=(1, win_len), stride=win_hop)


def overlap_add_filter(win: torch.Tensor, win_hop: int):
    """
    Compute FFT-based overlap-add filter.
    :param win: the input window.
    :param win_hop: the hop (or stride) size.
    :return:
    """

    _, win_len, num_frames = win.shape
    out_len = win_len + win_hop * (num_frames - 1)

    return Fn.fold(win, (1, out_len), kernel_size=(1, win_len), stride=win_hop).flatten(1)


def extend_fbins(x: torch.Tensor) -> torch.Tensor:
    """
    Extending the number of frequency bins from (fft_len // 2 + 1) back to (fft_len) by
    reversing all bins except DC and Nyquist and append it on top of existing spectrogram.

    :param x: the input spectrogram.
    :return:
    """

    x_upper = x[:, 1:, -1].flip(1)
    x_upper[:, :, :, 1] *= -1   # for the imaginary part, it is an odd function

    return torch.cat((x[:, :, :], x_upper), dim=1)


def broadcast_dim(x: torch.Tensor) -> tuple[Union[torch.Tensor, Any], Any]:
    """
    Auto broadcast input so that it can fit into `nn.Conv1d`.

    :param x: the input tensor.
    :return: broadcasting tensor of shape (B * C, 1, N) and the origin shape.
    """
    shape = x.shape
    if x.dim() == 1:
        # if nn.DataParallel is used, this broadcast doesn't work
        x = x[None, None, :]
    elif x.dim() == 2:
        x = x[:, None, :]
    elif x.dim() == 3:
        if x.shape[1] > 1:
            x = x.view(-1, 1, x.shape[-1])
    else:
        raise ValueError(f"Only support input in shape (N, ) or (B, N) or (B, C, N)")
    return x, shape


def reshape_dim(x: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """
    Convert the input tensor of shape (B * C, F, T) to (B, C, F, T)

    :param x: the input tensor.
    :param shape: the original shape to reshape
    :return: reshaped tensor of shape (B * C, 1, N) and the origin shape.
    """

    if len(shape) == 1:
        # if the original shape is (N,)
        return torch.squeeze(torch.squeeze(x, dim=0), dim=0)
    elif len(shape) == 2:
        # if the origin shape is (B, N)
        return torch.squeeze(x, dim=0)
    elif len(shape) == 3:
        # if the origin shape is (B, C, N)
        return x.view(shape[0], shape[1], x.shape[1], x.shape[2])
    else:
        raise ValueError(f"Only support input in shape (N, ) or (B, N) or (B, C, N)")


def create_fourier_kernels(fft_len: int,
                           win_len: Optional[int] = None,
                           win_type: str = 'hann',
                           fbins: Optional[int] = None,
                           fmin: int = 50,
                           fmax: int = 6000,
                           fscale: str = 'linear',
                           sampling_rate: int = 44100) -> tuple[ndarray, ndarray, list[float | Any], list[float | int | Any], ndarray]:
    """
    This function creates the Fourier Kernel for STFT, Melspectrogram and CQT.
    Most of the parameters follow librosa conventions.
    Part of the code comes from pytorch_musicnet (Reference: https://github.com/jthickstun/pytorch_musicnet.)

    :param fft_len: the size of FFT.
    :param win_len: the size of window frame and STFT filter. Default: None (treated as equal to fft_len).
    :param win_type: the windowing type for STFT.
            It uses `scipy.signal.get_window`, please refer to scipy documentation for possible windowing functions.
            Default: 'hann'.
    :param fbins: the number of frequency bins. Default: None (be equivalent to fft_len // 2 + 1 bins).
    :param fmin: the starting frequency for the lowest frequency bin.
            If `freq_scale` is 'no', this argument does nothing.
            Default: 50.
    :param fmax: the ending frequency for the highest frequency bin.
            If `freq_scale` is 'no', this argument does nothing.
            Default: 6000.
    :param fscale: determine the spacing between each frequency bin.
            When 'linear', 'log' or 'log2' is used, the bin spacing can be controlled by `fmin` and `fmax`.
            If 'no' is used, the bin will start at 0Hz and end at Nyquist frequency with linear spacing.
            Default: 'linear'.
    :param sampling_rate: the sampling rate for the input.
            It is used to calculate the correct `fmin` and `fmax`.
            Setting the correct sampling rate is very important for calculating the correct frequency.
    :returns:
        kernel_sin : imaginary Fourier Kernel array with the shape (num_fbins, 1, fft_len).
        kernel_cos : real Fourier Kernel array with the shape (num_fbins, 1, fft_len).
        bins2freq : a list that maps each frequency bin to frequency in Hz.
        bins : a list of the normalized frequency `k` in digital domain.
            This `k` is in the Discrete Fourier Transform equation.
    """

    if not fbins:
        fbins = fft_len // 2 + 1
    if not win_len:
        win_len = fft_len

    s = np.arange(0, fft_len, 1.0)
    kernel_sin = np.empty((fbins, 1, fft_len))
    kernel_cos = np.empty((fbins, 1, fft_len))
    bins2freqs = []
    bins = []

    # choose window shape
    win_mask = get_window(win_type, int(win_len), fftbins=True)
    win_mask = make_pad_center(win_mask, fft_len)

    if fscale == 'linear':
        bin_start = fmin * fft_len / sampling_rate
        scale_ind = (fmax - fmin) * (fft_len / sampling_rate) / fbins
        # only half of the bins contain useful info
        for k in range(fbins):
            bins2freqs.append((k * scale_ind + bin_start) * sampling_rate / fft_len)
            bins.append(k * scale_ind + bin_start)
            kernel_sin[k, 0, :] = np.sin(2 * np.pi * (k * scale_ind + bin_start) * s / fft_len)
            kernel_cos[k, 0, :] = np.cos(2 * np.pi * (k * scale_ind + bin_start) * s / fft_len)
    elif fscale == 'log':
        bin_start = fmin * fft_len / sampling_rate
        scale_ind = np.log(fmax / fmin) / fbins
        # only half of the bins contain useful info
        for k in range(fbins):
            bins2freqs.append(np.exp(k * scale_ind) * bin_start * sampling_rate / fft_len)
            bins.append(np.exp(k * scale_ind) * bin_start)
            kernel_sin[k, 0, :] = np.sin(2 * np.pi * (np.exp(k * scale_ind) * bin_start) * s / fft_len)
            kernel_cos[k, 0, :] = np.cos(2 * np.pi * (np.exp(k * scale_ind) * bin_start) * s / fft_len)
    elif fscale == 'log2':
        bin_start = fmin * fft_len / sampling_rate
        scale_ind = np.log2(fmax / fmin) / fbins
        # only half of the bins contain useful info
        for k in range(fbins):
            bins2freqs.append(2 ** (k * scale_ind) * bin_start * sampling_rate / fft_len)
            bins.append(2 ** (k * scale_ind) * bin_start)
            kernel_sin[k, 0, :] = np.sin(2 * np.pi * (2 ** (k * scale_ind) * bin_start) * s / fft_len)
            kernel_cos[k, 0, :] = np.cos(2 * np.pi * (2 ** (k * scale_ind) * bin_start) * s / fft_len)
    elif fscale == 'no':
        # only half of the bins contain useful info
        for k in range(fbins):
            bins2freqs.append(k * sampling_rate / fft_len)
            bins.append(k)
            kernel_sin[k, 0, :] = np.sin(2 * np.pi * k * s / fft_len)
            kernel_cos[k, 0, :] = np.cos(2 * np.pi * k * s / fft_len)
    else:
        print(f"Please select the correct frequency scale, 'linear' or 'log'.")

    kernel_sin = kernel_sin.astype(np.float32)
    kernel_cos = kernel_cos.astype(np.float32)
    win_mask = win_mask.astype(np.float32)

    return kernel_sin, kernel_cos, bins2freqs, bins, win_mask
