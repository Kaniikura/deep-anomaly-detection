from __future__ import division
from __future__ import print_function


import abc

import numpy as np
import torch


class DistanceHookBase(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __call__(self, train_embs, test_embs, split):
        pass


class DefaultDistanceHook(DistanceHookBase):
    def __call__(self, train_embs, test_embs, split):
        assert isinstance(train_embs, np.ndarray)
        assert isinstance(test_embs,  np.ndarray)
        # euclidean distance
        return np.array([np.linalg.norm(train_embs - a_i, axis=1) for a_i in test_embs])

class CosineDistanceHook(DistanceHookBase):
    def __call__(self, train_embs, test_embs, split):
        if isinstance(train_embs, np.ndarray):
            assert isinstance(test_embs,  np.ndarray)
            l2_a = np.linalg.norm(test_embs , axis=1)
            l2_b = np.linalg.norm(train_embs, axis=1)
            inner = np.dot(test_embs, train_embs.T)
            norms = np.dot(l2_a.reshape(-1, 1), l2_b.reshape(1, -1))
            return 1.0 - (inner / norms)

        else: # torch.float or torch.cuda.FloatTensor
            eps = 1e-8
            a_n, b_n = train_embs.norm(dim=1)[:, None], test_embs.norm(dim=1)[:, None]
            a_norm = train_embs  / torch.max(a_n, eps * torch.ones_like(a_n))
            b_norm = test_embs / torch.max(b_n, eps * torch.ones_like(b_n))
            sim_mt = torch.mm(b_norm, a_norm.transpose(0, 1))

            return 1 - sim_mt