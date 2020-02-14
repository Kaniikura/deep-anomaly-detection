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

        #######################################################################
        # segmentation models
        #######################################################################
        return build_from_config(config, MODELS)

    def build_metric_model(config):
        model = build_from_config(config, BACKBONES)

        # Adacos
        if config.name.startswith('AdaCos'):
            model.last = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(backbone.last_channel, config.params.num_classes),
            )

        # ArcFace
        if config.name.startswith('squeezenet'):
            model.last = nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Conv2d(512, config.params.num_classes, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1))
            )

        # TODO: other models

        return model
