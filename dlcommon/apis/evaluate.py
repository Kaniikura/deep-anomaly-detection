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
        build_loss,
        build_optimizer,
        build_scheduler,
        build_dataloaders
)
import dlcommon.utils


def get_train_data_embedddings(config, model, split, dataloader, hooks, epoch):
    model.eval()

    batch_size = config.evaluation.batch_size
    total_size = len(dataloader.dataset)
    total_step = math.ceil(total_size / batch_size)

    aggregated_embs = []
    tbar = tqdm.tqdm(enumerate(dataloader), total=total_step)
    for i, data in tbar:
        images = data['image'].cuda()
        labels = data['label'].cuda()
        

        with torch.no_grad():
            embs = hooks.forward_fn(model=model, images=images, labels=labels,
                                    data=data, is_train=False)
            embs = embs.cpu().detach().numpy()
            aggregated_embs.append(embs) 

    # Putting all embeddings in shape (number of samples, length of one sample embeddings)
    aggregated_embs = np.concatenate(aggregated_embs) 

    return aggregated_embs

def evaluate_single_epoch(config, model, split, dataloader, hooks, epoch):
    model.eval()

    batch_size = config.evaluation.batch_size
    total_size = len(dataloader.dataset)
    total_step = math.ceil(total_size / batch_size)

    with torch.no_grad():
        losses = []
        aggregated_loss_dict = defaultdict(list)
        aggregated_outputs_dict = defaultdict(list)
        aggregated_outputs = []
        aggregated_distances = []
        aggregated_is_anomalies = []

        tbar = tqdm.tqdm(enumerate(dataloader), total=total_step)
        for i, data in tbar:
            images = data['image'].cuda() #to(device)
            labels = data['label'].cuda() #to(device)
            is_anomalies = data['is_anomaly'].numpy()
            aggregated_is_anomalies.append(is_anomalies)

            outputs = hooks.forward_fn(model=model, images=images, labels=labels,
                                       data=data, is_train=False)
            outputs = hooks.post_forward_fn(outputs=outputs, images=images, labels=labels,
                                            data=data, is_train=False)
            #embs = 

            #distances = 
            
            loss = hooks.loss_fn(outputs=outputs, labels=labels, data=data, is_train=False)
            if isinstance(loss, dict):
                loss_dict = loss
                loss = loss_dict['loss']
            else:
                loss_dict = {'loss': loss}
            losses.append(loss.item())

            f_epoch = epoch + i / total_step
            tbar.set_description(f'{split}, {f_epoch:.2f} epoch')

            for key, value in loss_dict.items():
                aggregated_loss_dict[key].append(value.item())

    def concatenate(v):
        # not a list or empty
        if not isinstance(v, list) or not v:
            return v

        # ndarray
        if isinstance(v[0], np.ndarray):
            return np.concatenate(v, axis=0)
        
        return v

    aggregated_is_anomalies = concatenate(aggregated_is_anomalies)
    log_dict = {key: sum(value)/len(value) for key, value in aggregated_loss_dict.items()}
    metric_dict =  hooks.metric_fn(outputs=distances, labels=labels, 
                                    data=data, is_train=True, split=split)
    log_dict.update(metric_dict)

    hooks.logger_fn(split=split,
                    outputs=aggregated_outputs,
                    labels=aggregated_is_anomalies,
                    log_dict=log_dict,
                    epoch=epoch)

    return metric_dict['score']

def evaluate_split(config, model, split, dataloader, hooks):
    model.eval()

    batch_size = config.evaluation.batch_size
    total_size = len(dataloader.dataset)
    total_step = math.ceil(total_size / batch_size)

    with torch.no_grad():
        losses = []
        aggregated_outputs_dict = defaultdict(list)
        aggregated_outputs = []
        aggregated_labels = []

        tbar = tqdm.tqdm(enumerate(dataloader), total=total_step)
        for i, data in tbar:
            images = data['image'].cuda()
            labels = data['label'].cuda()

            outputs = hooks.forward_fn(model=model, images=images, labels=labels,
                                       data=data, is_train=False)
            outputs = hooks.post_forward_fn(outputs=outputs, images=images, data=data, is_train=False)
            loss = hooks.loss_fn(outputs=outputs, labels=labels, data=data, is_train=False)
            losses.append(loss.item())

            # aggregate outputs
            if isinstance(outputs, dict):
                for key, value in outputs.items():
                    aggregated_outputs_dict[key].append(value.cpu().numpy())
            else:
                aggregated_outputs.append(outputs.cpu().numpy())

            aggregated_labels.append(labels.cpu().numpy())

    def concatenate(v):
        # not a list or empty
        if not isinstance(v, list) or not v:
            return v

        # ndarray
        if isinstance(v[0], np.ndarray):
            return np.concatenate(v, axis=0)
        
        return v

    aggregated_outputs_dict = {key:concatenate(value) for key, value in aggregated_outputs_dict.items()}
    aggregated_outputs = concatenate(aggregated_outputs)
    aggregated_labels = concatenate(aggregated_labels)

    # list & empty
    if isinstance(aggregated_outputs, list) and not aggregated_outputs:
        aggregated_outputs = aggregated_outputs_dict
    ##
    #WIP
    ##
    score = None 

    return score


def evaluate(config, model, hooks, dataloaders):
    # get train data embeddings
    for dataloader in dataloaders:
        split = dataloader['split']
        dataset_mode = dataloader['mode']
        if split != 'get_embeddings':
            continue

        dataloader = dataloader['dataloader']
        embs = get_train_data_embedddings(config, model, split, dataloader)
    # validation
    for dataloader in dataloaders:
        split = dataloader['split']
        dataset_mode = dataloader['mode']

        if dataset_mode != 'validation':
            continue

        dataloader = dataloader['dataloader']
        score = evaluate_split(config, model, split, dataloader, hooks)
        print(f'[{split}] score: {score}')


def run(config):
    # build hooks
    hooks = build_hooks(config)

    # build model
    model = build_model(config, hooks)

    # build loss
    loss = build_loss(config)
    loss_fn = hooks.loss_fn
    hooks.loss_fn = lambda **kwargs: loss_fn(loss_fn=loss, **kwargs)

    # load checkpoint
    checkpoint = config.checkpoint
    last_epoch, step = dlcommon.utils.load_checkpoint(model, None, checkpoint)

    # build datasets
    dataloaders = build_dataloaders(config)

    model = model.cuda()
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # train loop
    evaluate(config=config,
             model=model,
             dataloaders=dataloaders,
             hooks=hooks)

