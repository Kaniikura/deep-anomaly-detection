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
    def __call__(self, distances, labels, split, train_labels=None):
        pass

class DefaultMetricHook(MetricHookBase):
    def __call__(self, distances, labels, split, train_labels=None):
        if split == 'evaluation':
            # multi-label classification
            if isinstance(labels, torch.Tensor):
                labels = labels.numpy()
            assert len(distances.shape) == 2
            assert train_labels is not None
            pred_ids = np.argmin(distances, axis=1) # get the id of the closest training data
            preds = train_labels[pred_ids]
            correct = (preds==labels).sum()
            total = len(labels)
            acc = correct/ total
            res = {'score': acc}

        if split == 'inference':
            # Binary classification of anomaly or normal
            if isinstance(labels, torch.Tensor):
                labels = labels.numpy()
            assert len(distances.shape) == 2
            preds = np.min(distances, axis=1) # min distance is anomaly score
            fpr, tpr, thresholds = metrics.roc_curve(labels, preds)
            auc = metrics.auc(fpr, tpr)
            res = {'auc': auc, 'thresholds': thresholds, 'anomaly_score':preds, 'label':labels}

        return res