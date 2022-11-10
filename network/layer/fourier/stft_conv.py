from typing import Optional
import numpy as np
from scipy.signal import get_window
import torch
import torch.nn as nn
import torch.nn.functional as Fn
from network.layer.fourier.utils import *

__all__ = ['StftConv', 'iStftConv']


class StftConv(nn.Module):
    """
    This function is to calculate the short-time Fourier transform (STFT) of the input signal.
    Input signal should be in either of the following shapes.
    1. (N)
    2. (B, 1, N)
    3. (B, C, N)
    The correct shape will be inferred automatically if the input follows these 3 shapes.
    Most of the arguments follow the convention from librosa.

    Reference:
    - https://github.com/KinWaiCheuk/nnAudio/blob/master/Installation/nnAudio/features/stft.py
    """

    def __init__(self,
                 fft_len: int = 2048,
                 win_len: Optional[int] = None,
                 win_hop: Optional[int] = None,
                 win_type: str = 'hann',
                 fbins: Optional[int] = None,
                 fscale: str = 'no',
                 fmin: int = 50,
                 fmax: int = 6000,
                 pad_center: bool = True,
                 pad_mode: str = 'reflect',
                 sampling_rate: int = 200,
                 trainable: bool = True,
                 feature_type: str = 'complex'):
        """

        :param fft_len: size of fourier transform. Default: 2048.
        :param win_len: the size of window frame and STFT filter. Default: None (treated as equal to fft_len).
        :param win_hop: the hop (or stride) size. Default: None (be equivalent to fft_len // 4).
        :param win_type: win_type: the windowing type for STFT.
                It uses `scipy.signal.get_window`, please refer to scipy documentation for possible windowing functions.
                Default: 'hann'.
        :param fbins: the number of frequency bins. Default: None (be equivalent fft_len // 2 + 1 bins).
        :param fscale: determine the spacing between each frequency bin.
                When `linear`, 'log' or `log2` is used, the bin spacing can be controlled by `freq_min` and `freq_max`.
                If 'no' is used, the bin will start at 0Hz and end at Nyquist frequency with linear spacing.
                Default: 'no'.
        :param fmin: the starting frequency for the lowest frequency bin.
                If `freq_scale` is 'no', this argument does nothing.
                Default: 50.
        :param fmax: freq_max: the ending frequency for the highest frequency bin.
                If `freq_scale` is 'no', this argument does nothing.
                Default: 6000.
        :param pad_center: pad_center: Putting the STFT kernel at the center of the time-step or not.
                If False, the time index is the beginning of the STFT kernel.
                If True, the time index is the center of the STFT kernel.
                Default: True.
        :param pad_mode: pad_mode: the padding method. Default: 'reflect'.
        :param sampling_rate: the sampling rate for the input. It is used to calculate the correct `fmin` and `fmax`.
                Setting the correct sampling rate is very important for calculating the correct frequency.
                Default: 200.
        :param trainable: Determine if the STFT kernels are trainable or not.
                If True, the gradients for STFT kernels will also be calculated and the STFT kernels will be updated during model training.
                Default: False.
        :param feature_type: control the spectrogram output type, either 'magnitude', 'complex', or 'phase'.
                The output_format can also be changed during the `forward` method.
                Default: 'complex'.
        """

        if feature_type not in ['complex', 'magnitude', 'phase']:
            raise ValueError(f"Expected values: 'complex', 'magnitude', 'phase', but got 'feature_type' = {feature_type}.")

        super(StftConv, self).__init__()

        # trying to make the default setting same as librosa
        if not win_len:
            win_len = fft_len
        if not win_hop:
            win_hop = int(win_len // 4)

        self.fft_len = fft_len
        self.win_len = win_len
        self.win_hop = win_hop
        self.win_type = win_type
        self.pad_center = pad_center
        self.pad_mode = pad_mode
        self.pad_amount = self.fft_len // 2
        self.fbins = fbins
        self.fscale = fscale
        self.fmin = fmin
        self.fmax = fmax
        self.sampling_rate = sampling_rate
        self.trainable = trainable
        self.feature_type = feature_type

        # create filtering windows for stft
        kernel_sin, kernel_cos, bins2freqs, bins, win_mask = create_fourier_kernels(
            self.fft_len, self.win_len, self.win_type, self.fbins, self.fmin, self.fmax, self.fscale, self.sampling_rate
        )
        self.bins2freqs = bins2freqs
        self.bins = bins
        kernel_sin = torch.tensor(kernel_sin, dtype=torch.float)
        kernel_cos = torch.tensor(kernel_cos, dtype=torch.float)

        # apply window functions to the fourier kernels
        win_mask = torch.tensor(win_mask)
        kernel_sin = kernel_sin * win_mask
        kernel_cos = kernel_cos * win_mask

        if self.trainable:
            kernel_sin = nn.Parameter(kernel_sin, requires_grad=self.trainable)
            kernel_cos = nn.Parameter(kernel_cos, requires_grad=self.trainable)
            self.register_parameter('kernel_sin', kernel_sin)
            self.register_parameter('kernel_cos', kernel_cos)
        else:
            self.register_buffer('kernel_sin', kernel_sin)
            self.register_buffer('kernel_cos', kernel_cos)

        # prepare the shape of window mask so that it can be used later in inverse
        self.register_buffer('win_mask', win_mask.unsqueeze(0).unsqueeze(-1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        :param x: the input tensor of shape (N,) or (B, N) or (B, C, N).
        :return: the spectrogram tensor in shape (*, *, F, T) if `feature_type == 'magnitude`, otherwise, output in shape (*, *, F, T, 2).
        """

        # reshape input to form (B, C, T) or (B * C, 1, T)
        z, shape_ori = broadcast_dim(x)

        # padding
        if self.pad_center:
            padding = None
            if self.pad_mode == 'constant':
                padding = nn.ConstantPad1d(self.pad_amount, 0)
            elif self.pad_mode == 'reflect':
                if x.shape[-1] < self.pad_amount:
                    raise AssertionError("Expected signal length larger than reflect padding length (fft_len // 2).")
                padding = nn.ReflectionPad1d(self.pad_amount)
            z = padding(z)

        # do STFT using conv1d
        z_real = Fn.conv1d(z, self.kernel_cos, stride=self.win_hop)
        z_imag = Fn.conv1d(z, self.kernel_sin, stride=self.win_hop)

        # remove redundant parts
        z_real = z_real[:, : self.fbins, :]
        z_imag = z_imag[:, : self.fbins, :]

        # reshape back to original shape
        z_real = reshape_dim(z_real, shape_ori)
        z_imag = reshape_dim(z_imag, shape_ori)

        if self.feature_type == 'magnitude':
            z = z_real.pow(2) + z_imag.pow(2)
            if self.trainable:
                # prevent Nan gradient when sqrt(0) due to output=0
                z += 1e-8
            return torch.sqrt(z)
        elif self.feature_type == 'complex':
            return torch.stack((z_real, -z_imag), dim=-1)
        elif self.feature_type == 'phase':
            # +0.0 removes -0.0 elements, which leads to error in calculating phase
            return torch.atan2(-z_imag + 0.0, z_real)


class iStftConv(nn.Module):
    """
    This class is to convert spectrograms back to waveforms.
    It only works for the complex value spectrograms.
    The parameters (e.g. `fft_len`, `win_type`) need to be the same as the STFT in order to obtain the correct inverse.
    When `trainable` = True and `fscale` != 'no', there is no guarantee that the inverse is perfect, please use with extra care.

    Reference:
    - https://github.com/KinWaiCheuk/nnAudio/blob/master/Installation/nnAudio/features/stft.py
    - https://github.com/KinWaiCheuk/nnAudio/blob/master/Installation/nnAudio/features/griffin_lim.py
    """

    def __init__(self,
                 fft_len: int = 2048,
                 win_len: Optional[int] = None,
                 win_hop: Optional[int] = None,
                 win_type: str = 'hann',
                 fbins: Optional[int] = None,
                 fscale: str = 'no',
                 fmin: int = 50,
                 fmax: int = 6000,
                 pad_center: bool = True,
                 pad_mode: str = 'reflect',
                 sampling_rate: int = 200,
                 trainable: bool = True,
                 feature_type: str = 'complex'):
        """

        :param fft_len: size of fourier transform. Default: 2048.
        :param win_len: the size of window frame and STFT filter. Default: None (treated as equal to fft_len).
        :param win_hop: the hop (or stride) size.
                Please make sure the value is the same as the forward STFT.
                Default: None (be equivalent to fft_len // 4).
        :param win_type: win_type: the windowing type for STFT.
                It uses `scipy.signal.get_window`, please refer to scipy documentation for possible windowing functions.
                Please make sure the value is the same as the forward STFT.
                Default: 'hann'.
        :param fbins: the number of frequency bins.
                Please make sure the value is the same as the forward STFT.
                Default: None (be equivalent fft_len // 2 + 1 bins).
        :param fscale: determine the spacing between each frequency bin.
                When `linear`, 'log' or `log2` is used, the bin spacing can be controlled by `freq_min` and `freq_max`.
                If 'no' is used, the bin will start at 0Hz and end at Nyquist frequency with linear spacing.
                Please make sure the value is the same as the forward STFT.
                Default: 'no'.
        :param fmin: the starting frequency for the lowest frequency bin.
                If `freq_scale` is 'no', this argument does nothing.
                Please make sure the value is the same as the forward STFT.
                Default: 50.
        :param fmax: freq_max: the ending frequency for the highest frequency bin.
                If `freq_scale` is 'no', this argument does nothing.
                Please make sure the value is the same as the forward STFT.
                Default: 6000.
        :param pad_center: pad_center: Putting the STFT kernel at the center of the time-step or not.
                If False, the time index is the beginning of the STFT kernel.
                If True, the time index is the center of the STFT kernel.
                Please make sure the value is the same as the forward STFT.
                Default: True.
        :param pad_mode: pad_mode: the padding method. Default: 'reflect'.
        :param sampling_rate: the sampling rate for the input. It is used to calculate the correct `fmin` and `fmax`.
                Setting the correct sampling rate is very important for calculating the correct frequency.
                Please make sure the value is the same as the forward STFT.
                Default: 22050.
        :param trainable: Determine if the STFT kernels are trainable or not.
                If True, the gradients for STFT kernels will also be calculated and the STFT kernels will be updated during model training.
                Please make sure the value is the same as the forward STFT.
                Default: False.
        :param feature_type: control the spectrogram output type, either 'magnitude', 'complex', or 'phase'.
                The output_format can also be changed during the `forward` method.
                Please make sure the value is the same as the forward STFT.
                Default: 'complex'.
        """

        super(iStftConv, self).__init__()

        if not win_len:
            win_len = fft_len
        if not win_hop:
            win_hop = int(win_len // 4)

        self.fft_len = fft_len
        self.win_len = win_len
        self.win_hop = win_hop
        self.win_type = win_type
        self.pad_center = pad_center
        self.pad_mode = pad_mode
        self.pad_amount = self.fft_len // 2
        self.fbins = fbins
        self.fscale = fscale
        self.fmin = fmin
        self.fmax = fmax
        self.sampling_rate = sampling_rate
        self.trainable = trainable
        self.feature_type = feature_type

        # create filtering windows for inverse
        kernel_sin, kernel_cos, _, _, win_mask = create_fourier_kernels(
            self.fft_len, self.win_len, self.win_type, self.fbins, self.fmin, self.fmax, self.fscale, self.sampling_rate
        )
        win_mask = get_window(self.win_type, self.win_len, fftbins=True)
        win_mask = torch.tensor(win_mask).unsqueeze(0).unsqueeze(-1)
        # kernel_sin and kernel_cos have the shape (freq_bins, 1, n_fft, 1) to support conv2D
        kernel_sin = torch.tensor(kernel_sin, dtype=torch.float).unsqueeze(-1)
        kernel_cos = torch.tensor(kernel_cos, dtype=torch.float).unsqueeze(-1)

        if self.trainable:
            kernel_sin = nn.Parameter(kernel_sin, requires_grad=self.trainable)
            kernel_cos = nn.Parameter(kernel_cos, requires_grad=self.trainable)
            self.register_parameter('kernel_sin', kernel_sin)
            self.register_parameter('kernel_cos', kernel_cos)
        else:
            self.register_buffer('kernel_sin', kernel_sin)
            self.register_buffer('kernel_cos', kernel_cos)
        self.register_buffer('win_mask', win_mask)

    def _inverse_stft(self,
                      x: torch.Tensor,
                      kernel_cos: torch.Tensor,
                      kernel_sin: torch.Tensor,
                      one_sided: bool = True,
                      length: Optional[int] = None,
                      refresh_win: Optional[bool] = None) -> torch.Tensor:
        """

        :param x: the input tensor in shape of (*, *, F, T).
        :param kernel_cos: the cosine kernel tensor.
        :param kernel_sin: the sine kernel tensor.
        :param one_sided: whether the padding in one-sided or both-sided.
        :param length: the length of reconstructed signal.
        :param refresh_win: to reduce the loaded memory.
        :return:
        """

        if one_sided:
            # extend frequency
            x = extend_fbins(x)
        x_real, x_imag = x[:, :, :, 0], x[:, :, :, 1]
        # broadcast dimensions to support Conv2D
        x_real = x_real.unsqueeze(1)
        x_imag = x_imag.unsqueeze(1)
        a1 = Fn.conv2d(x_real, kernel_cos, stride=(1, 1))
        b2 = Fn.conv2d(x_imag, kernel_sin, stride=(1, 1))
        # compute the real and imaginary parts
        # the signal lies in the real part
        real = a1 - b2
        real = real.squeeze(-2) * self.win_mask
        # normalize the amplitude with fft_len
        real /= self.fft_len
        # overlap and add algorithm to connect all the frames
        real = overlap_add_filter(real, self.win_hop)
        # prepare the window sumsquare for division
        # only need to create this window once to save times
        # unless the input spectrogram have different time steps
        if (not hasattr(self, 'w_sum')) or refresh_win:
            self.win_sum = window_sumsquare(self.win_mask.flatten(), x.shape[2], self.win_hop, self.fft_len).flatten()
            self.nonzero_indices = self.win_sum > 1e-10
        else:
            pass
        real[:, self.nonzero_indices] = real[:, self.nonzero_indices].div(self.win_sum[self.nonzero_indices])

        # remove padding
        if not length:
            if self.pad_center:
                real = real[:, self.pad_amount: -self.pad_amount]
        else:
            if self.pad_center:
                real = real[:, self.pad_amount: self.pad_amount + length]
            else:
                real = real[:, : length]

        return real

    def _griffin_lim(self,
                     x: torch.Tensor,
                     length: int,
                     num_iters: int = 32,
                     momentum: float = 0.99) -> torch.Tensor:
        """

        :param x: the input tensor
        :param length:
        :param num_iters:
        :param momentum:
        :return:
        """

        # creating window function for stft and istft later
        win = torch.tensor(get_window(self.win_type, int(self.win_len), fftbins=True), device=x.device).float()
        # initialize random phase
        phase_random = torch.randn(*x.shape, device=x.device)
        angles = torch.empty((*x.shape, 2), device=x.device)
        angles[:, :, :, 0] = torch.cos(2 * np.pi * phase_random)
        angles[:, :, :, 1] = torch.sin(2 * np.pi * phase_random)
        # initialize the rebuilt magnitude spectrogram
        rebuilt = torch.zeros(*angles.shape, device=x.device)

        for _ in range(num_iters):
            # save previous rebuilt magnitude spec
            tprev = rebuilt
            inverse = torch.istft(
                x.unsqueeze(-1) * angles, self.fft_len, self.win_hop, self.win_len, win, self.pad_center, length=length
            )
            # wav2spec conversion
            rebuilt = torch.stft(
                inverse, self.fft_len, self.win_hop, self.win_len, win, pad_mode=self.pad_mode,
            )
            # phase update rule
            angles[:, :, :] = rebuilt - (momentum / (1 + momentum)) * tprev
            # phase normalization
            angles = angles.div(
                torch.sqrt(angles.pow(2).sum(-1)).unsqueeze(-1) + 1e-16
            )

        # use the final phase to reconstruct the waveforms
        inverse = torch.istft(
            x.unsqueeze(-1) * angles, self.fft_len, self.win_hop, self.win_len, win, self.pad_center, length=length
        )
        return inverse

    def forward(self, x: torch.Tensor, N: int) -> torch.Tensor:
        """

        :param x: the input tensor in shape (F, T) or (B, F, T) or (B, C, F, T) or (B, C, F, T, 2).
        :param N: the feature dim.
        :return: the output tensor in shape of (*, *, N).
        """

        # broadcast dim
        z = None
        if x.dim() == 2:
            z = x[None, :]
        elif x.dim() == 4:
            z = x.view(-1,  x.shape[2], x.shape[3])
        elif x.dim() == 5:
            z = x.view(-1, x.shape[2], x.shape[3], x.shape[4])

        if z.dim() == 3:
            z = self._griffin_lim(z, N)
        elif z.dim() == 4:
            # work with complex number in shape (B, F, T, 2)
            B, F, T, _ = z.shape
            one_sided = True if (F == (self.fft_len // 2 + 1)) else False
            z = self._inverse_stft(x, self.kernel_cos, self.kernel_sin, one_sided, N, True)
        return z
