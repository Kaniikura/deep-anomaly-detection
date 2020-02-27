from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import abc

import numpy as np
import torchvision

class LoggerHookBase(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __call__(self, writer, split, outputs, labels, log_dict,
                 epoch, step=None, num_steps_in_epoch=None, is_normalized=True):
        pass


def inv_normalize(imgs, mean=(0.485, 0.456, 0.406), 
                        std=(0.229, 0.224, 0.225)):
    imgs[:, 0, :, :] = imgs[:, 0, :, :] * std[0] + mean[0]
    imgs[:, 1, :, :] = imgs[:, 1, :, :] * std[1] + mean[1]
    imgs[:, 2, :, :] = imgs[:, 2, :, :] * std[2] + mean[2]
    
    return imgs

class DefaultLoggerHook(LoggerHookBase):
    def __call__(self, writer, split, outputs, labels, log_dict,
                 epoch, step=None, num_steps_in_epoch=None, is_normalized=True):
        if step is not None:
            assert num_steps_in_epoch is not None
            log_step = epoch * 10000 + (step / num_steps_in_epoch) * 10000
            log_step = int(log_step)
        else:
            log_step = epoch

        for key, value in log_dict.items():
            if key=='images' or key=='gen_images' or key=='recon_images':
                imgs = log_dict[key]
                if is_normalized:
                    imgs = inv_normalize(imgs)
                grid = torchvision.utils.make_grid(imgs)
                writer.add_image(f'{split}/{key}', grid, log_step)
            else:
                writer.add_scalar(f'{split}/{key}', log_dict[key], log_step)
