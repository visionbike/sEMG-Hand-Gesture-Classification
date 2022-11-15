from typing import Union, Optional
import multiprocessing as multproc
# from concurrent.futures import ThreadPoolExecutor
from numpy.typing import NDArray
import numpy as np

__all__ = ['replace_by_first_label', 'replace_by_major_label', 'compute_class_weights', 'get_class_weights', 'window_rolling']


def _get_first_label(x: NDArray) -> NDArray:
    """
    Get the first label in the overlapped windowing case.
    :param x: the input label array.
    :return: the first label in the label array.
    """

    return np.unique(x)[0]


def _get_major_label(x: NDArray) -> NDArray:
    """
    Get the major label in the overlapped windowing case.

    :param x: the input label array.
    :return: the major label in the label array.
    """

    # get dictionary with keys are label and values are corresponding counts
    unique = dict(zip(*np.unique(x, return_counts=True)))
    return max(unique.keys())


def replace_by_first_label(x: list) -> NDArray:
    """
    Multiprocessing function to process butterworth filter for multiple signals.

    :param x: the input label list.
    :return: the first labeling array.
    """

    # with ThreadPoolExecutor(max_workers=min(32, os.cpu_count() + 4)) as executor:
    #     z = [r for r in executor.map(_get_first_label, x)]
    num_workers = multproc.cpu_count() - 1
    num_samples = len(x)
    with multproc.Pool(processes=num_workers) as p:
        z = list(p.imap(_get_first_label, x, chunksize=num_samples // num_workers))
    return np.asarray(z)


def replace_by_major_label(x: list) -> NDArray:
    """
    Multiprocessing function to process butterworth filter for multiple signals.

    :param x: the input label list.
    :return: the major labeling array.
    """

    # with ThreadPoolExecutor(max_workers=min(32, os.cpu_count() + 4)) as executor:
    #     z = [r for r in executor.map(_get_major_label, list)]
    num_workers = multproc.cpu_count() - 1
    num_samples = len(x)
    with multproc.Pool(processes=num_workers) as p:
        z = list(p.imap(_get_major_label, x, chunksize=num_samples // num_workers))
    return np.asarray(z)


def compute_class_weights(x: NDArray) -> dict:
    """
    Compute the class weights.

    :param x: the list of labels.
    :return: the class weights.
    """

    unique = dict(zip(*np.unique(x, return_counts=True)))
    total = sum(unique.values(), 0.)
    props = {k: v / total for k, v in unique.items()}
    props_min = min(props.values())
    props_norm = {k: v / props_min for k, v in unique.items()}
    return props_norm


def get_class_weights(num_classes: int, use_weight: bool, based_weights: Optional[Union[str, float, int]]) -> Optional[NDArray]:
    """
    Get the clas weights form configuration.

    :param num_classes: the number of classes.
    :param use_weight: whether to apply class weights or not.
    :param based_weights: the weight file path or weight array.
        If 'base_weights' is *. npy file path, load the class weights from the file.
        If 'base_weights' is float or integer value, return a list of these values.
        If 'base_weights' is None, return a list of ones.
    :return: the class weight array or None.
    """

    if use_weight:
        if isinstance(based_weights, str):
            weights = np.load(based_weights).tolist()
        elif isinstance(based_weights, (float, int)):
            weights = np.full(num_classes, based_weights).tolist()
        else:
            weights = np.ones(num_classes).tolist()
    else:
        weights = None
    return weights


def window_rolling(x: NDArray, ssize: int = 5, wsize: int = 52) -> NDArray:
    """
    Window rolling implementation

    :param x: the input signal
    :param ssize: the rolling step size. Default: 5.
    :param wsize: the window size of each segment. Default: 52.
    :return:
    """

    z = np.dstack([x[i: i + x.shape[0] - wsize + 1: ssize] for i in range(wsize)])
    return z
