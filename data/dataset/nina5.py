import gc
import numpy as np
from numpy.typing import NDArray
from torch.utils.data import Dataset
from data.transform import *

__all__ = ['Nina5Dataset']


class Nina5Dataset(Dataset):
    """
    Pytorch implementation for NinaPro DB5.
    """

    def __init__(self,
                 data: NDArray,
                 msize: [None | int] = 15,
                 use_augment: bool = False,
                 use_rest_label: bool = True):
        """

        :param data: the Nina DB5 data, include signal and its label.
        :param msize: the window size for moving average. Default: 15.
        :param use_augment: whether to use data augmentation. Default: False.
        :param use_rest_label: whether to use 'rest' label. Default: True.
        """

        super(Nina5Dataset, self).__init__()

        # load data
        if not data['imu']:
            self.data = data['emg'].copy()
        else:
            self.data = np.concatenate([data['emg'].copy(), data['imu'].copy()], axis=-1)
        self.lbls = self.data['lbl'].copy()

        # release memory
        del data
        gc.collect()

        # get transformation
        transforms = []
        if use_augment:
            transforms += [NinaRandomSNR(use_rest_label)]
        if msize is not None:
            transforms += [NinaMovingAverage(msize)]
        transforms += [NinaToTensor()]
        self.transforms = NinaCompose(transforms)

    def __len__(self):
        return self.lbls.shape[0]

    def __getitem__(self, idx: int):
        x = self.data[idx]
        y = self.lbls[idx]
        x, y = self.transforms(x, y)
        return x, y
