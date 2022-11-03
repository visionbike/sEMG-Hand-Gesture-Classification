import gc
import multiprocessing as multproc
from numpy.typing import NDArray
import numpy as np

__all__ = ['replace_by_first_label', 'replace_by_major_label', 'compute_class_weights', 'window_rolling']


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


def replace_by_first_label(x: NDArray) -> NDArray:
    """
    Multiprocessing function to process butterworth filter for multiple signals.

    :param x: the input label array.
    :return: the first labeling array.
    """

    inn = [x[i] for i in range(x.shape[0])]
    with multproc.Pool(None) as p:
        z = p.map(_get_first_label, inn)
    del inn
    gc.collect()
    return np.asarray(z)


def replace_by_major_label(x: NDArray) -> NDArray:
    """
    Multiprocessing function to process butterworth filter for multiple signals.

    :param x: the input label array.
    :return: the first labeling array.
    """

    inn = [x[i] for i in range(x.shape[0])]
    with multproc.Pool(None) as p:
        z = p.map(_get_first_label, inn)
    del inn
    gc.collect()
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


def window_rolling(x: NDArray, ssize: int = 5, wsize: int = 52) -> NDArray:
    """
    Window rolling implementation

    :param x: the input signal
    :param ssize: the rolling step size. Default: 5.
    :param wsize: the window size of each segment. Default: 52.
    :return:
    """

    z = np.dstack([x[i: i - wsize + 1 or None: ssize] for i in range(wsize)])
    return z
