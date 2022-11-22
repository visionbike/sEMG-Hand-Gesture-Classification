from typing import Optional
import torchmetrics as tm
import torchmetrics.classification as tmc

__all__ = ['get_metrics']


def get_metrics(num_classes: int = 1,
                prefix: Optional[str] = None,
                postfix: Optional[str] = None,
                cf_mat: bool = False):
    """
    Get classification metric collection.

    :param num_classes: the number of classes. Default: 1.
    :param prefix: the string to append in front of keys of the output dict. Default: None.
    :param postfix: the string to append after keys of the output dict. Default: None.
    :param cf_mat: whether to apply the confusion matrix. Default: False.
    :return: the classification metrics.
    """
    metric_dict = dict(
        accuracy=tmc.MulticlassAccuracy(num_classes, average='micro'),
        balanced_accuracy=tmc.MulticlassRecall(num_classes, average='macro'),
        mathews_corr_coef=tmc.MulticlassMatthewsCorrCoef(num_classes),
        cohen_kappa=tmc.MulticlassCohenKappa(num_classes),
    )
    if cf_mat:
        conf_mat = tmc.MulticlassConfusionMatrix(num_classes, normalize='all')
    else:
        conf_mat = None

    metrics = tm.MetricCollection(
        metrics=metric_dict,
        prefix=prefix,
        postfix=postfix,
    )
    return metrics, conf_mat
