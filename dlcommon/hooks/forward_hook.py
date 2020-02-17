from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import abc

class ForwardHookBase(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __call__(self, model, images, split, labels=None, data=None):
        pass


class DefaultForwardHook(ForwardHookBase):
    def __call__(self, model, images, split, labels=None, data=None):
        if split=='train' or split =='validation':
            return model(images, labels)
        elif split in ['evaluation', 'get_embeddings', 'inference']:
            return model.encoder(images)

class PostForwardHookBase(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __call__(self, outputs, images=None, labels=None, data=None, split=None, train_embs=None):
        pass


class DefaultPostForwardHook(PostForwardHookBase):
    def __call__(self, outputs, images=None, labels=None, data=None, split=None, train_embs=None):
        return outputs

