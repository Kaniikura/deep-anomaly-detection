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
from dlcommon.apis.metric_learning.train import run as run_train
from dlcommon.apis.metric_learning.evaluate import run as run_evaluate
from dlcommon.apis.metric_learning.inference import run as run_inference

ex.captured_out_filter = apply_backspaces_and_linefeeds

# rewrite config here to reduce redundancy,
# since yaml does not support character concatenation, 
def eval_config(config):
    # rewrite config
    config.model.name = config.model_name
    config.dataset.params.data_dir = os.path.join(str(config.dataset.params.data_dir),str(config.data.name))
    config.train.dir = os.path.join(str(config.train.dir), str(config.model_name), str(config.fold_idx))
    config.checkpoint = os.path.join(str(config.train.dir),str(config.checkpoint))
    config.inference.output_path = os.path.join(str(config.inference.output_path), str(config.model_name))
    config.inference.reference_csv_filename = os.path.join(str(config.dataset.params.data_dir), str(config.inference.reference_csv_filename))
    splits_list = []
    for i, split in enumerate(config.dataset.splits):
        mode = split['mode']
        split = split['split']
        print(split)
        if split in ['train','get_embeddings',]:
            folds = [i for i in range(config.num_folds) if i != config.fold_idx]
        elif split in ['validation', 'evaluation', 'inference']:
            folds = [config.fold_idx]
        splits_list.append({
                'mode': mode, 'split': split, 'folds': folds
            })
    config.dataset.splits = splits_list

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
def evaluate(_run, _config):
    config = edict(_config)
    config = eval_config(config)
    print('------------------------------------------------')
    print('evaluate')
    pprint.PrettyPrinter(indent=2).pprint(config)
    run_evaluate(config)

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