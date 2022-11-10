import gc
from pathlib import Path
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from .preprocessing import *
from .transform import *
from .dataset import *

__all__ = ['get_data']


def get_data(name: str, **kwargs: dict) -> pl.LightningDataModule:
    """

    :param name: the dataset name.
    :param kwargs:
    :return:
    """

    if name == 'nina1':
        return Nina1Data(**kwargs)
    elif name == 'nina4':
        return Nina4Data(**kwargs)
    elif name == 'nina5':
        return Nina5Data(**kwargs)
    else:
        raise ValueError(f"Invalid value 'name' = {name}. Valid values: 'nina1' | 'nina4' | 'nina5'.")


class Nina1Data(pl.LightningDataModule):
    """
    PyTorchlightning data implementation for NinaPro DB1.
    """

    def __init__(self, **kwargs):
        """

        :param kwargs:
        """

        # check parameters
        if 'path' not in kwargs.keys():
            raise ValueError("Not found 'path' argument.")
        if 'batch_size' not in kwargs.keys():
            raise ValueError("Not found 'batch_size' argument.")
        if 'num_workers' not in kwargs.keys():
            raise ValueError("Not found 'num_workers' argument.")
        if 'msize' not in kwargs.keys():
            raise ValueError("Not found 'msize' argument.")
        if 'use_augment' not in kwargs.keys():
            raise ValueError("Not found 'use_augment' argument.")
        if 'use_rest_label' not in kwargs.keys():
            raise ValueError("Not found 'use_rest_label' argument.")

        super(Nina1Data, self).__init__()

        self.path = Path(kwargs.pop('path'))
        if not self.path.exists():
            raise FileExistsError(f"Path: {str(self.path)} does not exist.")

        self.batch_size = kwargs.pop('batch_size')
        self.num_workers = kwargs.pop('num_workers')

        self.kwargs = kwargs

        self.train = None
        self.val = None
        self.test = None

    def setup(self, stage: [None | str] = None) -> None:
        if stage in (None, 'train'):
            data_train = np.load(str(self.path / 'train.npz'), allow_pickle=True)
            data_val = np.load(str(self.path / 'val.npz'), allow_pickle=True)
            data_test = np.load(str(self.path / 'test.npz'), allow_pickle=True)
            #
            self.train = Nina1Dataset(data_train, self.kwargs)
            self.val = Nina1Dataset(data_val, self.kwargs)
            self.test = Nina1Dataset(data_test, self.kwargs)
            #
            del data_train, data_val, data_test
            gc.collect()
        elif stage == 'test':
            data_test = np.load(str(self.path / 'test.npz'), allow_pickle=True)
            self.test = Nina1Dataset(data_test, self.kwargs)
            #
            del data_test
            gc.collect()
        else:
            raise ValueError(f"Invalid value 'stage' = {stage}. Valid values: 'train' | 'test'.")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train,
                          batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val,
                          batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test,
                          batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)


class Nina4Data(pl.LightningDataModule):
    """
    PyTorchlightning data implementation for NinaPro DB4.
    """

    def __init__(self, **kwargs):
        """

        :param kwargs:
        """

        # check parameters
        if 'path' not in kwargs.keys():
            raise ValueError("Not found 'path' argument.")
        if 'batch_size' not in kwargs.keys():
            raise ValueError("Not found 'batch_size' argument.")
        if 'num_workers' not in kwargs.keys():
            raise ValueError("Not found 'num_workers' argument.")
        if 'msize' not in kwargs.keys():
            raise ValueError("Not found 'msize' argument.")
        if 'use_augment' not in kwargs.keys():
            raise ValueError("Not found 'use_augment' argument.")
        if 'use_rest_label' not in kwargs.keys():
            raise ValueError("Not found 'use_rest_label' argument.")

        super(Nina4Data, self).__init__()

        self.path = Path(kwargs.pop('path'))
        if not self.path.exists():
            raise FileExistsError(f"Path: {str(self.path)} does not exist.")

        self.batch_size = kwargs.pop('batch_size')
        self.num_workers = kwargs.pop('num_workers')

        self.kwargs = kwargs

        self.train = None
        self.val = None
        self.test = None

    def setup(self, stage: [None | str] = None) -> None:
        if stage in (None, 'train'):
            data_train = np.load(str(self.path / 'train.npz'), allow_pickle=True)
            data_val = np.load(str(self.path / 'val.npz'), allow_pickle=True)
            data_test = np.load(str(self.path / 'test.npz'), allow_pickle=True)
            #
            self.train = Nina4Dataset(data_train, self.kwargs)
            self.val = Nina4Dataset(data_val, self.kwargs)
            self.test = Nina4Dataset(data_test, self.kwargs)
            #
            del data_train, data_val, data_test
            gc.collect()
        elif stage == 'test':
            data_test = np.load(str(self.path / 'test.npz'), allow_pickle=True)
            self.test = Nina4Dataset(data_test, self.kwargs)
            #
            del data_test
            gc.collect()
        else:
            raise ValueError(f"Invalid value 'stage' = {stage}. Valid values: 'train' | 'test'.")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train,
                          batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val,
                          batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test,
                          batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)


class Nina5Data(pl.LightningDataModule):
    """
    PyTorchlightning data implementation for NinaPro DB5.
    """

    def __init__(self, **kwargs):
        """

        :param kwargs:
        """

        # check parameters
        if 'path' not in kwargs.keys():
            raise ValueError("Not found 'path' argument.")
        if 'batch_size' not in kwargs.keys():
            raise ValueError("Not found 'batch_size' argument.")
        if 'num_workers' not in kwargs.keys():
            raise ValueError("Not found 'num_workers' argument.")
        if 'msize' not in kwargs.keys():
            raise ValueError("Not found 'msize' argument.")
        if 'use_augment' not in kwargs.keys():
            raise ValueError("Not found 'use_augment' argument.")
        if 'use_rest_label' not in kwargs.keys():
            raise ValueError("Not found 'use_rest_label' argument.")

        super(Nina5Data, self).__init__()

        self.path = Path(kwargs.pop('path'))
        if not self.path.exists():
            raise FileExistsError(f"Path: {str(self.path)} does not exist.")

        self.batch_size = kwargs.pop('batch_size')
        self.num_workers = kwargs.pop('num_workers')

        self.kwargs = kwargs

        self.train = None
        self.val = None
        self.test = None

    def setup(self, stage: [None | str] = None) -> None:
        if stage in (None, 'train'):
            data_train = np.load(str(self.path / 'train.npz'), allow_pickle=True)
            data_val = np.load(str(self.path / 'val.npz'), allow_pickle=True)
            data_test = np.load(str(self.path / 'test.npz'), allow_pickle=True)
            #
            self.train = Nina5Dataset(data_train, **self.kwargs)
            self.val = Nina5Dataset(data_val, **self.kwargs)
            self.test = Nina5Dataset(data_test, **self.kwargs)
            #
            del data_train, data_val, data_test
            gc.collect()
        elif stage == 'test':
            data_test = np.load(str(self.path / 'test.npz'), allow_pickle=True)
            self.test = Nina5Dataset(data_test, **self.kwargs)
            #
            del data_test
            gc.collect()
        else:
            raise ValueError(f"Invalid value 'stage' = {stage}. Valid values: 'train' | 'test'.")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train,
                          batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val,
                          batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test,
                          batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)
