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
    def __call__(self, inputs, labels, split, train_labels=None, k=10):
        pass

class DefaultMetricHook(MetricHookBase):
    def __call__(self, inputs, labels, split, train_labels=None, k=10):
        return

class DMLMetricHook(MetricHookBase):
    def __call__(self, inputs, labels, split, train_labels=None, k=10):
        if split == 'evaluation':
            # multi-label classification
            if isinstance(labels, torch.Tensor):
                labels = labels.numpy()
            assert len(inputs.shape) == 2
            assert train_labels is not None
            pred_ids = np.argmin(inputs, axis=1) # get the id of the closest training data
            preds = train_labels[pred_ids]
            correct = (preds==labels).sum()
            total = len(labels)
            acc = correct/ total
            res = {'score': acc}

        if split == 'inference':
            # Binary classification of anomaly or normal
            if isinstance(labels, torch.Tensor):
                labels = labels.numpy()
            assert len(inputs.shape) == 2
            idx = np.argpartition(inputs, k, axis=1)[:,:k]
            preds = np.take_along_axis(inputs, idx, axis=1).mean(axis=1)
            fpr, tpr, thresholds = metrics.roc_curve(labels, preds)
            auc = metrics.auc(fpr, tpr)
            res = {'auc': auc, 'thresholds': thresholds, 'anomaly_score':preds, 'label':labels}

        return res

class AEMetricHook(MetricHookBase):
    def __call__(self, inputs, labels, split, train_labels=None, k=10):
        if isinstance(labels, torch.Tensor):
                labels = labels.numpy()
        assert len(inputs.shape) == 1
        fpr, tpr, thresholds = metrics.roc_curve(labels, inputs)
        auc = metrics.auc(fpr, tpr)
        res = {'auc': auc, 'thresholds': thresholds, 'anomaly_score':inputs, 'label':labels}

        return res