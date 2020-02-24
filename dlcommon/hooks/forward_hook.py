from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import abc

import torch.nn.functional as F

class ForwardHookBase(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __call__(self, model, images, split, labels=None, data=None):
        pass

class DefaultForwardHook(ForwardHookBase):
    def __call__(self, model, images, split, labels=None, data=None):
        return

class DMLForwardHook(ForwardHookBase):
    def __call__(self, model, images, split, labels=None, data=None):
        if split=='train' or split =='validation':
            return model(images, labels)
        elif split in ['evaluation', 'get_embeddings', 'inference']:
            x = model.get_feature(images)
            return F.normalize(x)

class AEForwardHook(ForwardHookBase):
    def __call__(self, model, images, split, labels=None, data=None):
       return model(images)


class PostForwardHookBase(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __call__(self, outputs, images=None, labels=None, data=None, split=None, train_embs=None):
        pass


class DefaultPostForwardHook(PostForwardHookBase):
    def __call__(self, outputs, images=None, labels=None, data=None, split=None, train_embs=None):
        return outputs

