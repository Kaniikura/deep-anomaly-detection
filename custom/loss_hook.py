from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tqdm
import numpy as np
import torch
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F

import dlcommon
import dlcommon.losses


@dlcommon.HOOKS.register
class MetricLossHook:
    def __init__(self):
        self.loss_cls = nn.BCEWithLogitsLoss()

    def __call__(self, loss_fn, outputs, labels, data, is_train):
       
        loss_dict = {
                'loss': loss,
                'loss_seg': loss_seg,
                'loss_cls': loss_cls
                }
        return loss_dict
