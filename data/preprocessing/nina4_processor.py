import sys
import gc
from typing import Any
from pathlib import Path
import numpy as np
import scipy.io as scio
from tqdm import tqdm
from .base_processor import BaseProcessor
from .utils import *

__all__ = ['Nina4Processor']


class Nina4Processor(BaseProcessor):
    """
    The preprocessing class for NinaPro DB4 data.
    """

    BASE_LABEL_IDS = dict(a=1, b=13, c=3)
    NUM_SUBJECTS = 10

    def __init__(self,
                 path: str,
                 use_butter: bool = True,
                 use_rectify: bool = True,
                 ssize: int = 5,
                 wsize: int = 52,
                 use_first_appearance: bool = False,
                 use_rest_label: bool = True):
        """

        :param path: the input data path.
        :param use_butter: whether to use butterworth filter. Default: True.
        :param use_rectify: whether to use rectifying. Default: True.
        :param ssize: step size for window rolling. Default: 5.
        :param wsize: window size for window rolling. Default: 52.
        :param use_first_appearance: if True, using first appearance strategy; otherwise, using major appearance strategy. Default: True.
        :param use_rest_label: whether to use the 'rest' label. Default: True.
        """

        if not Path(path).exists():
            raise FileExistsError(f"Directory {path} does not exist.")

        if not isinstance(ssize, int):
            raise ValueError(f"'ssize' requires Integer type, but got {type(ssize)}.")

        if not isinstance(wsize, int):
            raise ValueError(f"'wsize' requires Integer type, but got {type(wsize)}.")

        if (ssize <= 0) or (ssize > wsize):
            raise ValueError(f"'ssize' should be in range of (1, wsize, but got 'ssize' = {ssize}.")

        if wsize <= 0:
            raise ValueError(f"'wsize' should be larger than 0, but got 'wsize' = {wsize}.")

        self.path = Path(path)
        self.use_butter = use_butter
        self.use_rectify = use_rectify
        self.step_size = ssize
        self.window_size = wsize
        self.use_first_appearance = use_first_appearance
        self.use_rest_label = use_rest_label

        super(Nina4Processor, self).__init__()

        self.emgs, self.lbls, self.reps = None, None, None

        self.load_data()

    def _load_file(self, path: str, ex: str) -> tuple[Any, int | Any, Any]:
        """
        Load *.mat file and remap the class indices.

        :param path: the data file name.
        :param ex: the exercise name.
        :return: the data, label and repetition numpy array.
        """

        res = scio.loadmat(path)
        emgs = res['emg'].copy()
        # repetition labeled by a machine (more accurate labels, this is what we will use to split the data by)
        reps = res['rerepetition'].copy()
        # machine class exercises
        lbls = res['restimulus'].copy()
        # remap the labels to {0,..., 52}
        lbls = (lbls > 0).astype('int') * (self.BASE_LABEL_IDS[ex] + lbls - 1)
        # release memory
        del res
        gc.collect()
        return emgs, lbls, reps

    def _load_by_exercises(self, idx: int, ex: str) -> tuple[list[Any], list[Any], list[Any]]:
        """
        Load *.mat file by exercises.

        :param idx: the exercise index.
        :param ex: the exercise name.
        :return: emgs, imus, lbls and rep list.
        """

        emgs = []
        lbls = []
        reps = []
        for sub in tqdm(range(1, self.NUM_SUBJECTS + 1), file=sys.stdout):
            path = self.path / f's{sub}' / f'S{sub}_E{idx}_A1.mat'
            emg, lbl, rep = self._load_file(str(path), ex)
            emgs += [emg]
            lbls += [lbl]
            reps += [rep]
        return emgs, lbls, reps

    def load_data(self) -> None:
        """
        Load Nina5 data from *.mat files by exercises.

        :return:
        """

        self.emgs = []
        self.lbls = []
        self.reps = []
        print('### Loading data...')
        for i, ex in enumerate(self.BASE_LABEL_IDS.keys()):
            print(f"Exercise {ex}...")
            # exercise idx start from 1
            emgs, lbls, reps = self._load_by_exercises((i + 1), ex)
            self.emgs += emgs
            self.lbls += lbls
            self.reps += reps
        print('Done!')

    def _process_data_v1(self) -> None:
        """
        Preprocess sEMG data version 1:
        1. Processing data (rectifying, filtering, etc.)
        2. Rolling data
        3. Removing winding containing more than one repetition
        4. Relabel for multiple label-containing window
        5. Post-processing (quantize, do/do not remove rest label, etc.)

        :return:
        """

        print('### Processing data...')
        if self.use_rectify:
            print('Rectifying...')
            self.emgs = [np.abs(emg) for emg in self.emgs]
        if self.use_butter:
            print('Butterworth filtering...')
            self.emgs = [butter_band(emg, lcut=20., hcut=40., fs=2000., order=4) for emg in self.emgs]

        print('### Rolling data...')
        self.emgs = [window_rolling(emg, self.step_size, self.window_size) for emg in self.emgs]
        self.lbls = [window_rolling(lab, self.step_size, self.window_size) for lab in self.lbls]
        self.reps = [window_rolling(rep, self.step_size, self.window_size) for rep in self.reps]
        # reshape the data to have the axes in the proper order
        self.emgs = np.moveaxis(np.concatenate(self.emgs, axis=0), 2, 1)
        self.lbls = np.moveaxis(np.concatenate(self.lbls, axis=0), 2, 1)[..., -1]
        self.reps = np.moveaxis(np.concatenate(self.reps, axis=0), 2, 1)[..., -1]

        print('### Removing windows that contain multiple repetitions...')
        # split by repetition, but do not want any data leaks
        # sim ply drop any window that has more than one repetition in it
        no_leaks = np.array([
            i
            for i in range(self.reps.shape[0])
            if np.unique(self.reps[i]).shape[0] == 1
        ])
        self.emgs = self.emgs[no_leaks, :, :]
        self.lbls = self.lbls[no_leaks, :]
        self.reps = self.reps[no_leaks, :]
        # release memory
        del no_leaks
        gc.collect()

        print('### Replacing by first/major label appearance...')
        # next we want to make sure there aren't multiple labels
        # do this using the first class that appears in a window
        inn_lbls = [self.lbls[i] for i in range(self.lbls.shape[0])]
        inn_reps = [self.reps[i] for i in range(self.reps.shape[0])]
        if self.use_first_appearance:
            self.lbls = replace_by_first_label(inn_lbls)
            self.reps = replace_by_first_label(inn_reps)
        else:
            self.lbls = replace_by_major_label(inn_lbls)
            self.reps = replace_by_major_label(inn_reps)
        # release memory
        del inn_lbls, inn_reps
        gc.collect()

        print('### Post-processing...')
        print('Quantifying to float16...')
        self.emgs = self.emgs.astype(np.float16)
        print('Processing label...')
        if not self.use_rest_label:
            print(f"'use_rest_label' = {self.use_rest_label}. Removing 'rest' label data...")
            self.emgs = self.emgs[np.where(self.lbls != 0)[0]]
            self.reps = self.reps[np.where(self.lbls != 0)[0]]
            self.lbls = self.lbls[np.where(self.lbls != 0)[0]]
            self.lbls -= 1
        print('Done!')

    def _process_data_v2(self) -> None:
        """
        Preprocess sEMG data version 2:
        1. Rolling data
        2. Processing data (rectifying, filtering, etc.)
        3. Removing winding containing more than one repetition
        4. Relabel for multiple label-containing window
        5. Post-processing (quantize, do/do not remove rest label, etc.)

        :return:
        """

        print('### Rolling data...')
        self.emgs = [window_rolling(emg, self.step_size, self.window_size) for emg in self.emgs]
        self.lbls = [window_rolling(lab, self.step_size, self.window_size) for lab in self.lbls]
        self.reps = [window_rolling(rep, self.step_size, self.window_size) for rep in self.reps]
        # reshape the data to have the axes in the proper order
        self.emgs = np.moveaxis(np.concatenate(self.emgs, axis=0), 2, 1)
        self.lbls = np.moveaxis(np.concatenate(self.lbls, axis=0), 2, 1)[..., -1]
        self.reps = np.moveaxis(np.concatenate(self.reps, axis=0), 2, 1)[..., -1]

        print('### Removing windows that contain multiple repetitions...')
        # split by repetition, but do not want any data leaks
        # simply drop any window that has more than one repetition in it
        no_leaks = np.array([
            i
            for i in range(self.reps.shape[0])
            if np.unique(self.reps[i]).shape[0] == 1
        ])
        self.emgs = self.emgs[no_leaks, :, :]
        self.lbls = self.lbls[no_leaks, :]
        self.reps = self.reps[no_leaks, :]
        # release memory
        del no_leaks
        gc.collect()

        print('### Replacing by first/major appearance...')
        # next we want to make sure there aren't multiple labels
        # do this using the first/major appearance in a window
        inn_lbls = [self.lbls[i] for i in range(self.lbls.shape[0])]
        inn_reps = [self.reps[i] for i in range(self.reps.shape[0])]
        if self.use_first_appearance:
            self.lbls = replace_by_first_label(inn_lbls)
            self.reps = replace_by_first_label(inn_reps)
        else:
            self.lbls = replace_by_major_label(inn_lbls)
            self.reps = replace_by_major_label(inn_reps)
        # release memory
        del inn_lbls, inn_reps
        gc.collect()

        print('### Processing data...')
        if self.use_rectify:
            print('Rectifying...')
            self.emgs = np.abs(self.emgs)
        # return self.processed_emgs
        if self.use_butter:
            # because the processed samples are large, therefore multiprocessing is needed
            print('Butterworth filtering...')
            inn_emgs = [self.emgs[i] for i in range(self.emgs.shape[0])]
            self.emgs = process_butter_band(inn_emgs, lcut=20., hcut=40., fs=2000., order=4)
            # release memory
            del inn_emgs
            gc.collect()

        print('### Post processing...')
        print('Quantifying to float16...')
        self.emgs = self.emgs.astype(np.float16)
        print('Processing label...')
        if not self.use_rest_label:
            print(f"'use_rest_label' = {self.use_rest_label}. Removing 'rest' label data...")
            self.emgs = self.emgs[np.where(self.lbls != 0)[0]]
            self.reps = self.reps[np.where(self.lbls != 0)[0]]
            self.lbls = self.lbls[np.where(self.lbls != 0)[0]]
            self.lbls -= 1
        print('Done!')

    def process_data(self, ver: int = 1) -> None:
        """
        Processing sEMG data.

        :param ver: Process data by the 1st or 2nd way. Default: 1.
        :return:
        """

        if ver == 1:
            self._process_data_v1()
        else:
            self._process_data_v2()

    def split_data(self, split: str) -> dict[str, Any]:
        """
        Split into train/val/test datasets by repetitions.

        :param split: include `train`, `val` or `test`.
        :return: the split data and corresponding labels.
        """

        print(f"'split' = {split}. Splitting datasets by repetitions...")
        reps_unique = np.unique(self.reps)
        val_reps = reps_unique[3::2]
        if split == 'train':
            reps = reps_unique[np.where(np.isin(reps_unique, val_reps, invert=True))]
            idxs = np.where(np.isin(self.reps, np.array(reps)))
            data = dict(emg=self.emgs[idxs].copy(),
                        lbl=self.lbls[idxs].copy())
        elif split == 'test':
            reps = [val_reps[-1]]
            idxs = np.where(np.isin(self.reps, np.array(reps)))
            data = dict(emg=self.emgs[idxs].copy(),
                        lbl=self.lbls[idxs].copy())
        elif split == 'val':
            reps = val_reps[:-1]
            idxs = np.where(np.isin(self.reps, np.array(reps)))
            data = dict(emg=self.emgs[idxs].copy(),
                        lbl=self.lbls[idxs].copy())
        else:
            raise ValueError(f"Expected values: 'train'|'val'|'test', but got 'split' = {split}.")
        return data
