def build_model(self):

    self.G = Generator(self.batch_size,self.imsize, self.z_dim, self.g_conv_dim).cuda()
    self.D = Discriminator(self.batch_size,self.imsize, self.d_conv_dim).cuda()
    if self.parallel:
        self.G = nn.DataParallel(self.G)
        self.D = nn.DataParallel(self.D)

    # Loss and optimizer
    # self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, [self.beta1, self.beta2])
    self.g_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.G.parameters()), self.g_lr, [self.beta1, self.beta2])
    self.d_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.D.parameters()), self.d_lr, [self.beta1, self.beta2])

    self.c_loss = torch.nn.CrossEntropyLoss()
    # print networks
    print(self.G)
    print(self.D)



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

def tensor2var(x, grad=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=grad)

def prepare_directories(config):
    os.makedirs(os.path.join(config.train.dir, 'checkpoint'), exist_ok=True)

def train_single_epoch(config, G, D, split, dataloader,
            hooks, optimizers, scheduler, epoch):
    batch_size = config.train.batch_size
    total_size = len(dataloader.dataset)
    total_step = math.ceil(total_size / batch_size)

    # Fixed input for debugging
    fixed_z = tensor2var(torch.randn(batch_size, config.train.z_dim))

    tbar = tqdm.tqdm(enumerate(dataloader), total=total_step)
    for i, data in tbar:
        # ================== Train D ================== #
        D.train()
        G.train()

        real_images, _ = data['image'].cuda()

        # Compute loss with real images
        # dr1, dr2, dr3, df1, df2, df3, gf1, gf2, gf3 are attention scores
        real_images = tensor2var(real_images)
        d_out_real,dr1,dr2, d3 = D(real_images)
        if config.train.adv_loss == 'wgan-gp':
            d_loss_real = - torch.mean(d_out_real)
        elif config.train.adv_loss == 'hinge':
            d_loss_real = torch.nn.ReLU()(1.0 - d_out_real).mean()

        # apply Gumbel Softmax
        z = tensor2var(torch.randn(real_images.size(0), config.train.z_dim))
        fake_images,gf1,gf2,gf3 = G(z)
        d_out_fake,df1,df2,df3 = D(fake_images)

        if config.train.adv_loss == 'wgan-gp':
            d_loss_fake = d_out_fake.mean()
        elif config.train.adv_loss == 'hinge':
            d_loss_fake = torch.nn.ReLU()(1.0 + d_out_fake).mean()


        # Backward + Optimize
        d_loss = d_loss_real + d_loss_fake
        optimizers['G'].zero_grad()
        optimizers['D'].zero_grad()
        d_loss.backward()
        optimizers['D'].step()


        if config.train.adv_loss == 'wgan-gp':
            # Compute gradient penalty
            alpha = torch.rand(real_images.size(0), 1, 1, 1).cuda().expand_as(real_images)
            interpolated = Variable(alpha * real_images.data + (1 - alpha) * fake_images.data, requires_grad=True)
            out,_,_,_ = D(interpolated)

            grad = torch.autograd.grad(outputs=out,
                                        inputs=interpolated,
                                        grad_outputs=torch.ones(out.size()).cuda(),
                                        retain_graph=True,
                                        create_graph=True,
                                        only_inputs=True)[0]

            grad = grad.view(grad.size(0), -1)
            grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
            d_loss_gp = torch.mean((grad_l2norm - 1) ** 2)

            # Backward + Optimize
            d_loss = self.lambda_gp * d_loss_gp

            self.reset_grad()
            d_loss.backward()
            self.d_optimizer.step()

        # ================== Train G and gumbel ================== #
        # Create random noise
        z = tensor2var(torch.randn(real_images.size(0), self.z_dim))
        fake_images,_,_ = self.G(z)

        # Compute loss with fake images
        g_out_fake,_,_ = self.D(fake_images)  # batch x n
        if self.adv_loss == 'wgan-gp':
            g_loss_fake = - g_out_fake.mean()
        elif self.adv_loss == 'hinge':
            g_loss_fake = - g_out_fake.mean()

        self.reset_grad()
        g_loss_fake.backward()
        self.g_optimizer.step()


        # Print out log info
        if (step + 1) % self.log_step == 0:
            elapsed = time.time() - start_time
            elapsed = str(datetime.timedelta(seconds=elapsed))
            print("Elapsed [{}], G_step [{}/{}], D_step[{}/{}], d_out_real: {:.4f}, "
                    " ave_gamma_l3: {:.4f}, ave_gamma_l4: {:.4f}".
                    format(elapsed, step + 1, self.total_step, (step + 1),
                            self.total_step , d_loss_real.data[0],
                            self.G.attn1.gamma.mean().data[0], self.G.attn2.gamma.mean().data[0] ))

        # Sample images
        if (step + 1) % self.sample_step == 0:
            fake_images,_,_= self.G(fixed_z)
            save_image(denorm(fake_images.data),
                        os.path.join(self.sample_path, '{}_fake.png'.format(step + 1)))

        if (step+1) % model_save_step==0:
            torch.save(self.G.state_dict(),
                        os.path.join(self.model_save_path, '{}_G.pth'.format(step + 1)))
            torch.save(self.D.state_dict(),
                        os.path.join(self.model_save_path, '{}_D.pth'.format(step + 1)))

   
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
