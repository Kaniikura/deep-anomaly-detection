from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import pprint
from easydict import EasyDict as edict
from sacred import Experiment
from sacred.utils import apply_backspaces_and_linefeeds

import torch
import custom   # import all custom modules for registering objects.

from dlcommon.initialization import initialize
from dlcommon.utils import ex
from dlcommon.apis.autoencoder.train import run as run_train
from dlcommon.apis.autoencoder.inference import run as run_inference


ex.captured_out_filter = apply_backspaces_and_linefeeds

@ex.main
def main(_run, _config):
    config = edict(_config)
    pprint.PrettyPrinter(indent=2).pprint(config)


@ex.command
def train(_run, _config):
    config = edict(_config)
    print('------------------------------------------------')
    print('train')
    pprint.PrettyPrinter(indent=2).pprint(config)
    run_train(config)

@ex.command
def evaluate(_run, _config):
    config = edict(_config)
    print('------------------------------------------------')
    print('evaluate')
    pprint.PrettyPrinter(indent=2).pprint(config)
    #run_evaluate(config)

@ex.command
def inference(_run, _config):
    config = edict(_config)
    print('------------------------------------------------')
    print('inference')
    pprint.PrettyPrinter(indent=2).pprint(config)
    run_inference(config)
    
if __name__ == '__main__':
    torch.multiprocessing.set_sharing_strategy('file_system')
    torch.backends.cudnn.deterministic=True

    initialize()
    ex.run_commandline()