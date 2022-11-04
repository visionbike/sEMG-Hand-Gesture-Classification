from typing import Optional
import torchmetrics as tm

__all__ = ['get_metrics']


def get_metrics(num_classes: int = 1,
                prefix: Optional[str] = None,
                postfix: Optional[str] = None):
    """
    Get classification metric collection.

    :param num_classes: the number of classes. Default: 1.
    :param prefix: the string to append in front of keys of the output dict. Default: None.
    :param postfix: the string to append after keys of the output dict. Default: None.
    :return: the classification metrics.
    """

    metrics = tm.MetricCollection(
        dict(acc=tm.Accuracy(num_classes),
             mcc=tm.MatthewsCorrCoef(num_classes),
             bacc=tm.Recall(num_classes, average='macro')),
        prefix=prefix,
        postfix=postfix
    )
    return metrics
