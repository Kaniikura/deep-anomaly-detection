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


def inference_split(config, G, D, E, split, dataloader, hooks, score_fn):
    G.eval()
    D.eval()
    E.eval()

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
        real_images = data['image'].cuda() #to(device)
        labels = data['is_anomaly']
        indices = data['index']
        
        with torch.no_grad():
            z = E(real_images)
            recon_images = G(z)
            recon_features = D(recon_images)
            image_features = D(real_images)

        loss_img = score_fn(recon_images, real_images)
        loss_fts = score_fn(recon_features, image_features)
        scores = loss_img #+ config.train.encoder.kappa*loss_fts
        aggregated_labels.append(labels.numpy())
        aggregated_scores.append(scores.cpu().numpy())
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

def inference(config, G, D, E, hooks, dataloaders, score_fn):
    # inference
    config.dataset.splits = [v for v in config.dataset.splits if v.split == config.inference.split]
    dataloaders = [dataloader for dataloader in dataloaders if dataloader['split'] == config.inference.split]
    assert len(dataloaders) == 1, f'len(dataloaders)({len(dataloaders)}) not 1'

    dataloader = dataloaders[0]
    split = dataloader['split']
    dataset_mode = dataloader['mode']
    dataloader = dataloader['dataloader']

    outputs_dict, indices = inference_split(config, G, D, E, split, dataloader, hooks, score_fn)
    auc = outputs_dict['auc']
    print(f'[{split}] AUC: {auc}')

    hooks.write_result_fn(split=split, output_path=config.inference.output_path, 
                            outputs=outputs_dict, indices=indices ,labels=None, data=None,
                            is_train=False, reference_csv_filename=None)


def run(config):
    # build hooks
    hooks = build_hooks(config)

    # build model
    model = build_model(config, hooks, member='gan')
    G = model.G
    D = model.D
    E = build_model(config, hooks, member='encoder')

    def _freeze_model(_model):
        for param in _model.parameters():
            param.requires_grad = False
        
    _freeze_model(G)
    _freeze_model(D)
    _freeze_model(E)

    G = G.cuda()
    D = D.cuda()
    E = E.cuda()
    if torch.cuda.device_count() > 1:
        G = torch.nn.DataParallel(G)
        D = torch.nn.DataParallel(D)
        E = torch.nn.DataParallel(E)

    # load checkpoint
    def load_from_checkpoint(_model, checkpoint_name):
        checkpoint = os.path.join(config.train.dir, checkpoint_name)
        last_epoch, step = dlcommon.utils.load_checkpoint(_model, None, checkpoint)

    load_from_checkpoint(G, config.checkpoint.g)
    load_from_checkpoint(D, config.checkpoint.d)
    load_from_checkpoint(E, config.checkpoint.e)

    # build datasets
    dataloaders = build_dataloaders(config)

    # calculation method for anomaly score
    from torch.nn import MSELoss
    class MSEInstances:
        def __init__(self):
            self.mse_elements = MSELoss(reduction='none')
        def __call__(self, input, target):
            loss_elements = self.mse_elements(input, target)
            loss_elements = torch.flatten(loss_elements, start_dim=1)
            loss_instances = loss_elements.mean(axis=1)
            return loss_instances
    score_fn = MSEInstances()  
        
    # train loop
    inference(config=config,
              G=G, D=D, E=E,
              dataloaders=dataloaders,
              hooks=hooks, score_fn=score_fn)
