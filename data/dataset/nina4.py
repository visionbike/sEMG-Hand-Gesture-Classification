import gc
from numpy.typing import NDArray
from torch.utils.data import Dataset
from data.transform import *

__all__ = ['Nina4Dataset']


class Nina4Dataset(Dataset):
    """
    Pytorch implementation for NinaPro DB5.
    """

    def __init__(self,
                 data: NDArray,
                 msize: [None | int] = 150,
                 use_augment: bool = False,
                 use_rest_label: bool = True, **kwargs):
        """

        :param data: the Nina DB4 data, include signal and its label.
        :param msize: the window size for moving average. Default: 150.
        :param use_augment: whether to use data augmentation. Default: False.
        :param use_rest_label: whether to use 'rest' label. Default: True.
        """

        super(Nina4Dataset, self).__init__()

        # load data
        self.data = data['emg'].copy()
        self.lbls = data['lbl'].copy()

        # release memory
        del data
        gc.collect()

        # get transformation
        transforms = []
        if use_augment:
            transforms += [NinaRandomSNR(use_rest_label)]
        if msize is not None:
            transforms += [NinaMovingAverage(msize)]
        transforms += [NinaTranspose(), NinaToTensor()]
        self.transforms = NinaCompose(transforms)

    def __len__(self):
        return self.lbls.shape[0]

    def __getitem__(self, idx: int):
        x = self.data[idx]
        y = self.lbls[idx]
        x, y = self.transforms(x, y)
        return x, y
