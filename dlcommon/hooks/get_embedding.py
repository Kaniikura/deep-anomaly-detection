from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import abc

class GetEmbeddingHookBase(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __call__(self, model, images, labels=None, data=None, is_train=False):
        pass


class DefaultGetEmbeddingHook(GetEmbeddingHookBase):
    def __call__(self, model, images, labels=None, data=None, is_train=False):
        embs = model.encoder(images)

        return embs

