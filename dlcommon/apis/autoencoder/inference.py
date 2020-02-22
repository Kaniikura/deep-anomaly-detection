import os
import math
from collections import defaultdict

import tqdm

import numpy as np
import torch
from tensorboardX import SummaryWriter

from dlcommon.builder import (
        build_hooks,
        build_model,
        build_dataloaders
)
import dlcommon.utils


def inference_split(config, model, split, dataloader, hooks, score_fn):
    model.eval()

    batch_size = config.inference.batch_size
    total_size = len(dataloader.dataset)
    total_step = math.ceil(total_size / batch_size)
    aggregated_loss_dict = defaultdict(list)
    aggregated_outputs_dict = defaultdict(list)
    aggregated_outputs = []
    aggregated_labels = []
    aggregated_scores = []
    aggregated_indices = []
    tbar = tqdm.tqdm(enumerate(dataloader), total=total_step)
    for i, data in tbar:
        images = data['image'].cuda() #to(device)
        labels = data['is_anomaly']
        indices = data['index']
        
        with torch.no_grad():
            outputs = hooks.forward_fn(model=model, images=images, labels=labels,
                                        data=data, split=split)
            outputs = hooks.post_forward_fn(outputs=outputs, images=images, labels=None,
                                        data=data, split=split)
            scores = score_fn(outputs, images)
            scores = scores.cpu()

        aggregated_labels.append(labels.numpy())
        aggregated_scores.append(scores.numpy())
        aggregated_indices.append(indices.numpy())


    def concatenate(v):
        # not a list or empty
        if not isinstance(v, list) or not v:
            return v

        # ndarray
        if isinstance(v[0], np.ndarray):
            return np.concatenate(v, axis=0)
        
        return v

    aggregated_labels = concatenate(aggregated_labels)
    aggregated_scores = concatenate(aggregated_scores)
    aggregated_indices = concatenate(aggregated_indices)
    metric_dict =  hooks.metric_fn(inputs=aggregated_scores, 
                                    labels=aggregated_labels, split=split, 
                                    train_labels=None)

    return metric_dict, aggregated_indices

def inference(config, model, hooks, dataloaders, score_fn):
    # inference
    config.dataset.splits = [v for v in config.dataset.splits if v.split == config.inference.split]
    dataloaders = [dataloader for dataloader in dataloaders if dataloader['split'] == config.inference.split]
    assert len(dataloaders) == 1, f'len(dataloaders)({len(dataloaders)}) not 1'

    dataloader = dataloaders[0]
    split = dataloader['split']
    dataset_mode = dataloader['mode']
    dataloader = dataloader['dataloader']

    outputs_dict, indices = inference_split(config, model, split, dataloader, hooks, score_fn)
    auc = outputs_dict['auc']
    print(f'[{split}] AUC: {auc}')

    from pathlib import Path
    result_path = Path(config.inference.result_path)
    if not result_path.is_dir():
        result_path.mkdir(parents=True)
    csv_path = result_path/'mvtec.csv'

    if csv_path.exists(): #　overwrite existing csv
        df = pd.read_csv(csv_path)
    else:
        if reference_csv_filename is not None:
            df = pd.read_csv(reference_csv_filename)
        else:
            df = pd.DataFrame(index=indices)
        df['AnomalyScore'] = np.nan
    
    df.loc[indices,'AnomalyScore'] = outputs['anomaly_score']


def run(config):
    # build hooks
    hooks = build_hooks(config)

    # build model
    model = build_model(config, hooks)
    model = model.cuda()
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # load checkpoint
    checkpoint = config.checkpoint
    last_epoch, step = dlcommon.utils.load_checkpoint(model, None, checkpoint)
    print(f'last_epoch:{last_epoch}')

    # build datasets
    dataloaders = build_dataloaders(config)

    # calculation method for anomaly score
    if config.loss.name == 'SSIMLoss':
        from dlcommon.losses import SSIMLoss
        score_fn = SSIMLoss(size_average=False)
    elif config.loss.name == 'MSELoss':
        from torch.nn import MSELoss
        class MSEInstances:
            def __init__(self):
                self.mse_elements = MSELoss(reduction='none')
            def __call__(self, input, target):
                loss_elements = self.mse_elements(input, target)
                loss_instances = loss_elements.mean(axis=1)
                return loss_instances
        score_fn = MSEInstances()  
        
    # train loop
    inference(config=config,
              model=model,
              dataloaders=dataloaders,
              hooks=hooks, score_fn=score_fn)
