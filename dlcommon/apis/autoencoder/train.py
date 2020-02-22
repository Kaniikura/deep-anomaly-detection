import os
import math
from collections import defaultdict

import tqdm

import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn import GroupNorm, Conv2d, Linear

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


def prepare_directories(config):
    os.makedirs(os.path.join(config.train.dir, 'checkpoint'), exist_ok=True)


def train_single_epoch(config, model, split, dataloader,
                       hooks, optimizer, scheduler, epoch):
    model.train()

    batch_size = config.train.batch_size
    total_size = len(dataloader.dataset)
    total_step = math.ceil(total_size / batch_size)

    tbar = tqdm.tqdm(enumerate(dataloader), total=total_step)
    for i, data in tbar:
        images = data['image'].cuda()
        outputs = hooks.forward_fn(model=model, images=images, labels=None,
                                   data=data, split=split)
        outputs = hooks.post_forward_fn(outputs=outputs, images=images, labels=None,
                                        data=data, split=split)
        loss = hooks.loss_fn(outputs=outputs, targets=images, data=data, split=split)
        
        if isinstance(loss, dict):
            loss_dict = loss
            loss = loss_dict['loss']
        else:
            loss_dict = {'loss': loss}

        loss.backward()
        
        optimizer.step()
        optimizer.zero_grad()
        
        if config.scheduler.name == 'OneCycleLR':
            scheduler.step()

        log_dict = {key:value.item() for key, value in loss_dict.items()}
        log_dict['lr'] = optimizer.param_groups[0]['lr']

        f_epoch = epoch + i / total_step
        tbar.set_description(f'{split}, {f_epoch:.2f} epoch')
        tbar.set_postfix(lr=optimizer.param_groups[-1]['lr'],
                         loss=loss.item())

        if i % config.train.image_log_step == 0:
            log_dict['images'] = images.cpu()
            log_dict['gen_images'] = outputs.detach().cpu()
        
        hooks.logger_fn(split=split, outputs=outputs, labels=None, log_dict=log_dict,
                        epoch=epoch, step=i, num_steps_in_epoch=total_step, is_normalized=False)
    

def validate_single_epoch(config, model, split, dataloader, hooks, epoch):
    model.eval()

    batch_size = config.evaluation.batch_size
    total_size = len(dataloader.dataset)
    total_step = math.ceil(total_size / batch_size)

    with torch.no_grad():
        losses = []
        aggregated_loss_dict = defaultdict(list)
        aggregated_outputs_dict = defaultdict(list)
        aggregated_outputs = []

        tbar = tqdm.tqdm(enumerate(dataloader), total=total_step)
        for i, data in tbar:
            images = data['image'].cuda() #to(device)
            outputs = hooks.forward_fn(model=model, images=images, labels=None,
                                       data=data, split=split)
            outputs = hooks.post_forward_fn(outputs=outputs, images=images, labels=None,
                                            data=data, split=split)
            loss = hooks.loss_fn(outputs=outputs, targets=images, data=data, split=split)
            if isinstance(loss, dict):
                loss_dict = loss
                loss = loss_dict['loss']
            else:
                loss_dict = {'loss': loss}
            losses.append(loss.item())

            f_epoch = epoch + i / total_step
            tbar.set_description(f'{split}, {f_epoch:.2f} epoch')
            tbar.set_postfix(loss=loss.item())

            for key, value in loss_dict.items():
                aggregated_loss_dict[key].append(value.item())
            log_dict = {}
            
            hooks.logger_fn(split=split, outputs=outputs, labels=None, log_dict=log_dict,
                        epoch=epoch, step=i, num_steps_in_epoch=total_step)

    log_dict = {key: sum(value)/len(value) for key, value in aggregated_loss_dict.items()}
            

    hooks.logger_fn(split=split,
                    outputs=aggregated_outputs,
                    labels=None,
                    log_dict=log_dict,
                    epoch=epoch)

    return -1*log_dict['loss']


def train(config, model, hooks, optimizer, scheduler, dataloaders, last_epoch):
    best_ckpt_score = -100000
    
    for epoch in range(last_epoch, config.train.num_epochs):
        # train 
        for dataloader in dataloaders:
            split = dataloader['split']
            dataset_mode = dataloader['mode']

            if dataset_mode != 'train':
                continue

            dataloader = dataloader['dataloader']
            train_single_epoch(config, model, split, dataloader, hooks,
                               optimizer, scheduler, epoch)

        score_dict = {}
        ckpt_score = None
        # validation
       
        for dataloader in dataloaders:
            split = dataloader['split']
            dataset_mode = dataloader['mode']

            if split != 'validation':
                continue

            dataloader = dataloader['dataloader']
            score = validate_single_epoch(config, model, split, dataloader, hooks,
                                          epoch)
            score_dict[split] = score
            # Use score of the first split
            if ckpt_score is None:
                ckpt_score = score

        # update learning rate
        if config.scheduler.name == 'ReduceLROnPlateau':
            scheduler.step(ckpt_score)
        elif config.scheduler.name == 'CosineAnnealingLR':
            param_epoch = (epoch + 1) % config.scheduler.params.T_max
            print('param_epoch:', param_epoch)
            scheduler.step(param_epoch+1)
        elif config.scheduler.name != 'OneCycleLR' and config.scheduler.name != 'ReduceLROnPlateau':
            scheduler.step()

        if config.scheduler.name == 'CosineAnnealingLR' and epoch % config.scheduler.params.T_max == config.scheduler.params.T_max - 1:
            snapshot_idx = epoch // config.scheduler.params.T_max
            print('save snapshot:', epoch, config.scheduler.params.T_max, snapshot_idx)
            dlcommon.utils.save_checkpoint(config, model, optimizer, epoch, keep=None,
                                      name=f'snapshot.{snapshot_idx}')
        if ckpt_score > best_ckpt_score:
            best_ckpt_score = ckpt_score
            dlcommon.utils.save_checkpoint(config, model, optimizer, epoch, keep=None,
                                      name='best.score')
            dlcommon.utils.copy_last_n_checkpoints(config, 5, 'best.score.{:04d}.pth')

        if epoch % config.train.save_checkpoint_epoch == 0:
            dlcommon.utils.save_checkpoint(config, model, optimizer,
                                        epoch, keep=config.train.num_keep_checkpoint)


def to_data_parallel(config, model, optimizer):
    if 'sync_bn' in config.train:
        print('sycn bn!!')
        from dlcommon.sync_batchnorm import SynchronizedBatchNorm1d, DataParallelWithCallback, convert_model
        model = convert_model(model)
        model = model.cuda()
        if torch.cuda.device_count() > 1:
            model = DataParallelWithCallback(model, list(range(torch.cuda.device_count())))
        return model, optimizer

    if torch.cuda.device_count() == 1:
        model = model.cuda()
        return model, optimizer

    model = model.cuda()
    return torch.nn.DataParallel(model), optimizer



def run(config):
    # prepare directories
    prepare_directories(config)

    # build hooks
    hooks = build_hooks(config)

    # build model
    model = build_model(config, hooks)
    # build loss
    loss = build_loss(config)
    loss_fn = hooks.loss_fn
    hooks.loss_fn = lambda **kwargs: loss_fn(loss_fn=loss, **kwargs)
    
    # build optimizer
    params = model.parameters()
    optimizer = build_optimizer(config, params=params)

    model = model.cuda()
    # load checkpoint
    checkpoint = dlcommon.utils.get_initial_checkpoint(config)
    if checkpoint is not None:
        last_epoch, step = dlcommon.utils.load_checkpoint(model, optimizer, checkpoint)
        print('epoch, step:', last_epoch, step)
    else:
        last_epoch, step = -1, -1

    model, optimizer = to_data_parallel(config, model, optimizer)

    # build scheduler
    scheduler = build_scheduler(config, optimizer=optimizer, 
                                last_epoch=last_epoch)

    # build datasets
    dataloaders = build_dataloaders(config)

    # build summary writer
    writer = SummaryWriter(logdir=config.train.dir)
    logger_fn = hooks.logger_fn
    hooks.logger_fn = lambda **kwargs: logger_fn(writer=writer, **kwargs)

    # train loop
    train(config=config,
          model=model,
          optimizer=optimizer,
          scheduler=scheduler,
          dataloaders=dataloaders,
          hooks=hooks,
          last_epoch=last_epoch+1)
