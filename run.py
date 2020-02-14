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
from dlcommon.apis.train import run as run_train
#from dlcommon.apis.evaluate import run as run_evaluate
#from dlcommon.apis.inference import run as run_inference
#from dlcommon.apis.swa import run as run_swa


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


if __name__ == '__main__':
    torch.multiprocessing.set_sharing_strategy('file_system')
    torch.backends.cudnn.deterministic=True

    initialize()
    ex.run_commandline()