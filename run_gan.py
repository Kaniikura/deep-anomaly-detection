from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import pprint
from easydict import EasyDict as edict
from sacred import Experiment
from sacred.utils import apply_backspaces_and_linefeeds

import os
import torch
import custom   # import all custom modules for registering objects.

from dlcommon.initialization import initialize
from dlcommon.utils import ex
from dlcommon.apis.gan.train import run as run_train
from dlcommon.apis.gan.inference import run as run_inference


ex.captured_out_filter = apply_backspaces_and_linefeeds

# rewrite config here to reduce redundancy,
# since yaml does not support character concatenation, 
def eval_config(config):
    # rewrite config
    dataname = str(config.data.name)
    modelname = str(config.model_name)
    config.dataset.params.data_dir = os.path.join(str(config.dataset.params.data_dir),dataname)
    config.train.dir = os.path.join(str(config.train.dir), dataname, modelname)
    config.inference.output_path = os.path.join(config.inference.output_path,dataname,modelname)
    config.inference.reference_csv_filename = os.path.join('data', dataname, 'csvs', 
                                                            config.inference.reference_csv_filename)

    return config

@ex.main
def main(_run, _config):
    config = edict(_config)
    pprint.PrettyPrinter(indent=2).pprint(config)


@ex.command
def train(_run, _config):
    config = edict(_config)
    config = eval_config(config)
    print('------------------------------------------------')
    print('train')
    pprint.PrettyPrinter(indent=2).pprint(config)
    run_train(config)

@ex.command
def inference(_run, _config):
    config = edict(_config)
    config = eval_config(config)
    print('------------------------------------------------')
    print('inference')
    pprint.PrettyPrinter(indent=2).pprint(config)
    run_inference(config)
    
if __name__ == '__main__':
    torch.multiprocessing.set_sharing_strategy('file_system')
    torch.backends.cudnn.deterministic=True

    initialize()
    ex.run_commandline()