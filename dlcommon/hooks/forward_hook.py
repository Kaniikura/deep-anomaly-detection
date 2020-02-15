from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import abc

def calc_distance(test_embs, train_embs):
    return test_embs*train_embs

class ForwardHookBase(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __call__(self, model, images, labels=None, data=None, is_train=False):
        pass


class DefaultForwardHook(ForwardHookBase):
    def __call__(self, model, images, labels=None, data=None, is_train=False):
        return model(images)

class MetricForwardHook(ForwardHookBase):
    def __call__(self, model, images, labels=None, data=None, is_train=False):
        if is_train:
            return model(images, labels)
        else:
            return model.encoder(images)


class PostForwardHookBase(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __call__(self, outputs, images=None, labels=None, data=None, is_train=False, train_embs=None):
        pass


class DefaultPostForwardHook(PostForwardHookBase):
    def __call__(self, outputs, images=None, labels=None, data=None, is_train=False, train_embs=None):
        return outputs

class MetricPostForwardHook(PostForwardHookBase):
    def __call__(self, outputs, images=None, labels=None, data=None, is_train=False, train_embs=None):
        if is_train:
            return outputs
        else:
            return calc_distance(outputs,train_embs)

