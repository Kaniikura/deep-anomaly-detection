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
    aggregated_embs = np.concatenate(aggregated_embs) 
    aggregated_labels = np.concatenate(aggregated_labels)

    return aggregated_embs, aggregated_labels

def evaluate_split(config, model, split, dataloader, hooks, train_embs, train_labels):
    model.eval()

    total = 0
    correct = 0
    batch_size = config.evaluation.batch_size
    total_size = len(dataloader.dataset)
    total_step = math.ceil(total_size / batch_size)

    with torch.no_grad():
        losses = []
        aggregated_loss_dict = defaultdict(list)
        aggregated_outputs_dict = defaultdict(list)
        aggregated_outputs = []
        aggregated_distances = []
        aggregated_labels = []

        tbar = tqdm.tqdm(enumerate(dataloader), total=total_step)
        for i, data in tbar:
            images = data['image'].cuda() #to(device)
            labels = data['label'].cuda() #to(device)

            embs = hooks.forward_fn(model=model, images=images, labels=labels,
                                       data=data, split=split)
            embs = embs.cpu().numpy()
            distances = hooks.distance_fn(train_embs=train_embs, test_embs=embs, split=split)
            aggregated_distances.append(distances)
            aggregated_labels.append(labels.cpu().numpy())


    def concatenate(v):
        # not a list or empty
        if not isinstance(v, list) or not v:
            return v

        # ndarray
        if isinstance(v[0], np.ndarray):
            return np.concatenate(v, axis=0)
        
        return v

    aggregated_labels = concatenate(aggregated_labels)
    aggregated_distances = concatenate(aggregated_distances)
    metric_dict =  hooks.metric_fn(inputs=aggregated_distances, 
                                    labels=aggregated_labels, split=split, 
                                    train_labels=train_labels)

    return metric_dict['score']

def evaluate(config, model, hooks, dataloaders):
    # get train data embeddings
    for dataloader in dataloaders:
        split = dataloader['split']
        dataset_mode = dataloader['mode']
        if split != 'get_embeddings':
            continue

        dataloader = dataloader['dataloader']
        embs, labels = get_train_data_embedddings(config, model, split, dataloader, hooks)

        import umap
        from sklearn.datasets import load_digits
        from scipy.sparse.csgraph import connected_components
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        from sklearn.manifold import TSNE
        import time
        # silence NumbaPerformanceWarning
        import warnings
        from numba.errors import NumbaPerformanceWarning
        warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)

        def show_umap(embs, targets):
            digits = load_digits()
            targets = [float(targets[i]) for i in range(len(targets))]

            # UMAP
            embedding = umap.UMAP().fit_transform(embs)
            plt.scatter(embedding[:,0],embedding[:,1],c=targets,cmap=cm.tab10)
            plt.colorbar()
            plt.show()
        
        show_umap(embs, labels)

    # evaluation
    for dataloader in dataloaders:
        split = dataloader['split']
        dataset_mode = dataloader['mode']

        if split != 'evaluation':
            continue

        dataloader = dataloader['dataloader']
        score = evaluate_split(config, model, split, dataloader, hooks, embs, labels)
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

