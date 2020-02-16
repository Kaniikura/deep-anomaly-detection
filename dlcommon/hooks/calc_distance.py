from __future__ import division
from __future__ import print_function


import abc

import numpy as np


class DistanceHookBase(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __call__(self, train_embs, test_embs, is_train, split):
        pass


class DefaultDistanceHook(DistanceHookBase):
    def __call__(self, train_embs, test_embs, is_train, split):
        assert isinstance(train_embs, np.ndarray)
        assert isinstance(test_embs,  np.ndarray)
        # euclidean distance
        return np.array([np.linalg.norm(train_embs - a_i, axis=1) for a_i in test_embs])

class CosineDistanceHook(DistanceHookBase):
    def __call__(self, train_embs, test_embs, is_train, split):
        assert isinstance(train_embs, np.ndarray)
        assert isinstance(test_embs,  np.ndarray)
        l2_test = np.linalg.norm(test_embs , axis=1)
        l2_train = np.linalg.norm(train_embs, axis=1)
        inner = np.dot(test_embs, train_embs.T)
        norms = np.dot(l2_test.reshape(-1, 1), l2_train.reshape(1, -1))

        return 1.0 - (inner / norms)