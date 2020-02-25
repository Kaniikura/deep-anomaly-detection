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


def get_train_data_embedddings(config, model, split, dataloader, hooks):
    model.eval()

    batch_size = config.evaluation.batch_size
    total_size = len(dataloader.dataset)
    total_step = math.ceil(total_size / batch_size)

    aggregated_embs = []
    aggregated_labels = []
    tbar = tqdm.tqdm(enumerate(dataloader), total=total_step)
    for i, data in tbar:
        images = data['image'].cuda()
        labels = data['label'].cuda()
        

        with torch.no_grad():
            embs = hooks.forward_fn(model=model, images=images, labels=labels,
                                    data=data, split=split)
            embs = embs.cpu().numpy()
            aggregated_embs.append(embs)
            aggregated_labels.append(labels.cpu().numpy()) 

    # Putting all embeddings in shape (number of samples, length of one sample embeddings)
    aggregated_embs = torch.FloatTensor(np.concatenate(aggregated_embs)).cuda() # WIP: refactoring
    aggregated_labels = np.concatenate(aggregated_labels)

    return aggregated_embs, aggregated_labels

def inference_split(config, model, split, dataloader, hooks, train_embs, train_labels):
    model.eval()

    batch_size = config.inference.batch_size
    total_size = len(dataloader.dataset)
    total_step = math.ceil(total_size / batch_size)

    with torch.no_grad():
        aggregated_loss_dict = defaultdict(list)
        aggregated_outputs_dict = defaultdict(list)
        aggregated_outputs = []
        aggregated_distances = []
        aggregated_labels = []
        aggregated_indices = []

        tbar = tqdm.tqdm(enumerate(dataloader), total=total_step)
        for i, data in tbar:
            images = data['image'].cuda() #to(device)
            labels = data['is_anomaly']
            indices = data['index']

            embs = hooks.forward_fn(model=model, images=images, labels=labels,
                                       data=data, split=split)
            distances = hooks.distance_fn(train_embs=train_embs, test_embs=embs, split=split)
            distances = distances.cpu().numpy()
            aggregated_distances.append(distances)
            aggregated_labels.append(labels.numpy())
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
    aggregated_indices = concatenate(aggregated_indices)
    aggregated_distances = concatenate(aggregated_distances)
    metric_dict =  hooks.metric_fn(inputs=aggregated_distances, 
                                    labels=aggregated_labels, split=split, 
                                    train_labels=train_labels)

    return metric_dict, aggregated_indices

def inference(config, model, hooks, dataloaders):
    # get train data embeddings
    for dataloader in dataloaders:
        split = dataloader['split']
        dataset_mode = dataloader['mode']
        if split != 'get_embeddings':
            continue

        dataloader = dataloader['dataloader']
        embs, labels = get_train_data_embedddings(config, model, split, dataloader, hooks)
    # inference
    config.dataset.splits = [v for v in config.dataset.splits if v.split == config.inference.split]
    dataloaders = [dataloader for dataloader in dataloaders if dataloader['split'] == config.inference.split]
    assert len(dataloaders) == 1, f'len(dataloaders)({len(dataloaders)}) not 1'

    dataloader = dataloaders[0]
    split = dataloader['split']
    dataset_mode = dataloader['mode']
    dataloader = dataloader['dataloader']

    outputs_dict, indices = inference_split(config, model, split, dataloader, hooks, embs, labels)
    auc = outputs_dict['auc']
    print(f'[{split}] AUC: {auc}')

    
    hooks.write_result_fn(split, config.inference.output_path, outputs=outputs_dict, 
                         indices=indices, reference_csv_filename = config.inference.reference_csv_filename)


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

    # train loop
    inference(config=config,
              model=model,
              dataloaders=dataloaders,
              hooks=hooks)
