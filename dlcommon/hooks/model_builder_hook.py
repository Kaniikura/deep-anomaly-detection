from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import abc

import dlcommon.registry
from dlcommon.registry import BACKBONES, MODELS
from dlcommon.utils import build_from_config


class ModelBuilderHookBase(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __call__(self, config):
        pass


class DefaultModelBuilderHook(ModelBuilderHookBase):
    def __call__(self, config):
        #######################################################################
        # metlic learning
        #######################################################################
        if BACKBONES.get(config.name) is not None:
            return self.build_metric_model(config)
            print('=====\n'*10)

        #######################################################################
        # GANs
        #######################################################################
        ## WIP

        #######################################################################
        # other models
        #######################################################################
        return build_from_config(config, MODELS)

    def build_metric_model(config):
        model = build_from_config(config, BACKBONES)

        return model
