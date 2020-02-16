from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import abc

import numpy as np
from sklearn import metrics
import torch


class MetricHookBase(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __call__(self, distances, labels, is_train, split):
        pass

class DefaultMetricHook(MetricHookBase):
    def __call__(self, distances, labels, is_train, split):
        if isinstance(labels, torch.Tensor):
            labels = labels.numpy()
        assert len(distances.shape) == 2
        preds = np.min(distances, axis=1) # min distance is anomaly score
        fpr, tpr, thresholds = metrics.roc_curve(labels, preds)
        auc = metrics.auc(fpr, tpr)

        return {'score': auc, 'thresholds': thresholds}