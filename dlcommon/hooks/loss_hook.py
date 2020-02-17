from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import abc


class LossHookBase(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __call__(self, outputs, labels, data, split):
        pass


class DefaultLossHook(LossHookBase):
    def __call__(self, loss_fn, outputs, labels, data, split):
        if isinstance(outputs, dict):
            return loss_fn(input=outputs['logits'], target=labels)
        return loss_fn(input=outputs, target=labels)
