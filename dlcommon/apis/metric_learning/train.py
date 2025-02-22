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
        labels = data['label'].cuda()
        outputs = hooks.forward_fn(model=model, images=images, labels=labels,
                                   data=data, split=split)
        outputs = hooks.post_forward_fn(outputs=outputs, images=images, labels=labels,
                                        data=data, split=split)
        loss = hooks.loss_fn(outputs=outputs, targets=labels, data=data, split=split)
        
        if isinstance(loss, dict):
            loss_dict = loss
            loss = loss_dict['loss']
        else:
            loss_dict = {'loss': loss}

        loss.backward()
        
        if config.train.gradient_accumulation_step is None:
            optimizer.step()
            optimizer.zero_grad()
        elif (i+1) % config.train.gradient_accumulation_step == 0:
            optimizer.step()
            optimizer.zero_grad()
        
        if config.scheduler.name == 'OneCycleLR':
            scheduler.step()

        log_dict = {key:value.item() for key, value in loss_dict.items()}
        log_dict['lr'] = optimizer.param_groups[-1]['lr']

        f_epoch = epoch + i / total_step
        tbar.set_description(f'{split}, {f_epoch:.2f} epoch')
        tbar.set_postfix(lr=optimizer.param_groups[-1]['lr'],
                         loss=loss.item())

        if i % 10 == 0:
            log_dict['images'] = images.cpu()
        
        hooks.logger_fn(split=split, outputs=outputs, labels=labels, log_dict=log_dict,
                        epoch=epoch, step=i, num_steps_in_epoch=total_step)
    

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
        aggregated_labels = []

        correct = 0
        total = 0

        tbar = tqdm.tqdm(enumerate(dataloader), total=total_step)
        for i, data in tbar:
            images = data['image'].cuda() #to(device)
            labels = data['label'].cuda() #to(device)
            outputs = hooks.forward_fn(model=model, images=images, labels=labels,
                                       data=data, split=split)
            outputs = hooks.post_forward_fn(outputs=outputs, images=images, labels=labels,
                                            data=data, split=split)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            acc = float(correct)/total
            loss = hooks.loss_fn(outputs=outputs, targets=labels, data=data, split=split)
            if isinstance(loss, dict):
                loss_dict = loss
                loss = loss_dict['loss']
            else:
                loss_dict = {'loss': loss}
            losses.append(loss.item())

            f_epoch = epoch + i / total_step
            tbar.set_description(f'{split}, {f_epoch:.2f} epoch')
            tbar.set_postfix(loss=loss.item(), acc=acc)

            for key, value in loss_dict.items():
                aggregated_loss_dict[key].append(value.item())
            log_dict = {}
            if i % 10 == 0:
                log_dict.update({'images':images.cpu()})
            hooks.logger_fn(split=split, outputs=outputs, labels=labels, log_dict=log_dict,
                        epoch=epoch, step=i, num_steps_in_epoch=total_step)

    log_dict = {key: sum(value)/len(value) for key, value in aggregated_loss_dict.items()}
            

    hooks.logger_fn(split=split,
                    outputs=aggregated_outputs,
                    labels=aggregated_labels,
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


def group_weight(module):
    group_decay = []
    group_no_decay = []
    for m in module.modules():
        if isinstance(m, nn.Linear):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, Conv2d):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, _BatchNorm):
            if m.weight is not None:
                group_no_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, GroupNorm):
            if m.weight is not None:
                group_no_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
    assert len(list(module.parameters())) == len(group_decay) + len(group_no_decay)
    return group_decay, group_no_decay


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
    if 'no_bias_decay' in config.train and config.train.no_bias_decay:
        if 'encoder_lr_ratio' in config.train:
            encoder_lr_ratio = config.train.encoder_lr_ratio
            group_decay_encoder, group_no_decay_encoder = group_weight(model.encoder)
            base_lr = config.optimizer.params.lr
            params = [{'params': model.product.parameters(), 'lr': base_lr},
                      {'params': model.fc.parameters(), 'lr': base_lr},
                      {'params': group_decay_encoder, 'lr': base_lr * encoder_lr_ratio},
                      {'params': group_no_decay_encoder, 'lr': base_lr * encoder_lr_ratio, 'weight_decay': 0.0}]
        else:
            group_decay, group_no_decay = group_weight(model)
            params = [{'params': group_decay},
                      {'params': group_no_decay, 'weight_decay': 0.0}]
    elif 'encoder_lr_ratio' in config.train:
        denom = config.train.encoder_lr_ratio
        base_lr = config.optimizer.params.lr
        params = [{'params': model.encoder.parameters(), 'lr': base_lr * denom},
                  {'params': model.fc.parameters(), 'lr': base_lr},
                  {'params': model.product.parameters(), 'lr': base_lr}]
    else:
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
