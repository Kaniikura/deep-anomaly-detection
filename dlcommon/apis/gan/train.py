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
from torch.autograd import Variable
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

def train_gan_single_epoch(config, G, D, g_optimizer, d_optimizer, split, dataloader, hooks, epoch):
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

        real_images = data['image'].cuda()

        # Compute loss with real images
        real_images = tensor2var(real_images)
        d_out_real = D(real_images)
        if config.train.adv_loss == 'wgan-gp':
            d_loss_real = - torch.mean(d_out_real)
        elif config.train.adv_loss == 'hinge':
            d_loss_real = torch.nn.ReLU()(1.0 - d_out_real).mean()

        # apply Gumbel Softmax
        z = tensor2var(torch.randn(real_images.size(0), config.train.z_dim))
        fake_images = G(z)
        d_out_fake = D(fake_images)

        if config.train.adv_loss == 'wgan-gp':
            d_loss_fake = d_out_fake.mean()
        elif config.train.adv_loss == 'hinge':
            d_loss_fake = torch.nn.ReLU()(1.0 + d_out_fake).mean()


        # Backward + Optimize
        d_loss = d_loss_real + d_loss_fake
        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()


        if config.train.adv_loss == 'wgan-gp':
            # Compute gradient penalty
            alpha = torch.rand(real_images.size(0), 1, 1, 1).cuda().expand_as(real_images)
            interpolated = Variable(alpha * real_images.data + (1 - alpha) * fake_images.data, requires_grad=True)
            out = D(interpolated)

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
            d_loss = config.train.lambda_gp * d_loss_gp

            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

        # ================== Train G and gumbel ================== #
        # Create random noise
        z = tensor2var(torch.randn(real_images.size(0), config.train.z_dim))
        fake_images = G(z)

        # Compute loss with fake images
        g_out_fake = D(fake_images)  # batch x n
        if config.train.adv_loss == 'wgan-gp':
            g_loss_fake = - g_out_fake.mean()
        elif config.train.adv_loss == 'hinge':
            g_loss_fake = - g_out_fake.mean()

        g_optimizer.zero_grad()
        g_loss_fake.backward()
        g_optimizer.step()


        # Print out log info
        f_epoch = epoch + i / total_step
        tbar.set_description(f'{split}, {f_epoch:.2f} epoch')
        tbar.set_postfix(d_out_real = f'{d_loss_real.data.item():.4f}')

        log_dict = dict()
        if i % config.train.image_log_step == 0:
            log_dict['images'] = real_images.cpu()
            log_dict['gen_images'] = fake_images.detach().cpu()
        
        if i % config.train.log_step == 0:
            log_dict.update({
                'd_loss_gp': d_loss_gp.item(),
                'gen_loss': g_loss_fake.item(),
            })
            hooks.logger_fn(split=split, outputs=None, labels=None, log_dict=log_dict,
                            epoch=epoch, step=i, num_steps_in_epoch=total_step, is_normalized=True)
    

def train_gan(config, G, D, g_optimizer, d_optimizer,
          dataloaders, hooks, last_epoch):
    best_ckpt_score = -100000
    
    for epoch in range(last_epoch, config.train.num_epochs):
        # train 
        for dataloader in dataloaders:
            split = dataloader['split']
            dataset_mode = dataloader['mode']

            if dataset_mode != 'train':
                continue

            dataloader = dataloader['dataloader']
            train_gan_single_epoch(config, G, D, g_optimizer, d_optimizer, split, 
                                dataloader, hooks, epoch)

        # save models
        if epoch % config.train.save_checkpoint_epoch == 0:
            dlcommon.utils.save_checkpoint(config, G, g_optimizer,
                                        epoch, keep=config.train.num_keep_checkpoint, member='g')
            dlcommon.utils.save_checkpoint(config, D, d_optimizer,
                                        epoch, keep=config.train.num_keep_checkpoint, member='d')


def train_enc_single_epoch(config, G, D, E, e_optimizer, split, dataloader, hooks, epoch):
    E.train()
    D.eval()
    G.eval()

    batch_size = config.train.batch_size
    total_size = len(dataloader.dataset)
    total_step = math.ceil(total_size / batch_size)

    # Fixed input for debugging
    fixed_z = tensor2var(torch.randn(batch_size, config.train.z_dim))

    MSE = torch.nn.MSELoss()

    tbar = tqdm.tqdm(enumerate(dataloader), total=total_step)
    for i, data in tbar:
        real_images = data['image'].cuda()
        # image encoding
        z = E(real_images)
        # reconstruct image from latent features
        recon_images = G(z)
        # get the last activation of dicriminator 
        recon_features = D(recon_images)
        image_features = D(real_images)

        loss_img = MSE(real_images, recon_images)
        loss_fts = MSE(recon_features, image_features)
        loss = loss_img + config.train.encoder.kappa*loss_fts

        # Backward + Optimize
        e_optimizer.zero_grad()
        loss.backward()
        e_optimizer.step()

        # Print out log info
        f_epoch = epoch + i / total_step
        tbar.set_description(f'{split}, {f_epoch:.2f} epoch')
        tbar.set_postfix(loss = f'{loss.item():.4f}')

        log_dict = dict()
        if i % config.train.image_log_step == 0:
            log_dict['images'] = real_images.cpu()
            log_dict['recon_images'] = recon_images.detach().cpu()
        
        if i % config.train.log_step == 0:
            log_dict.update({
                'loss': loss,
                'loss_img': loss_img,
                'loss_fts': loss_fts,
            })
            hooks.logger_fn(split=split, outputs=None, labels=None, log_dict=log_dict,
                            epoch=epoch, step=i, num_steps_in_epoch=total_step, is_normalized=True)

def train_encoder(config, G, D, E, e_optimizer, dataloaders, hooks, last_epoch):
    best_ckpt_score = -100000
    
    for epoch in range(last_epoch, config.train.encoder.num_epochs):
        # train 
        for dataloader in dataloaders:
            split = dataloader['split']
            dataset_mode = dataloader['mode']

            if dataset_mode != 'train':
                continue

            dataloader = dataloader['dataloader']
            train_enc_single_epoch(config, G, D, E, e_optimizer, split, 
                                dataloader, hooks, epoch)

        # save models
        if epoch % config.train.save_checkpoint_epoch == 0:
            dlcommon.utils.save_checkpoint(config, E, e_optimizer,
                                        epoch, keep=config.train.num_keep_checkpoint, member='e')
    

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

    # GAN training
    # build model
    model = build_model(config, hooks, member='gan')
    G = model.G
    D = model.D
    
    # build optimizer
    gen_params = G.parameters()
    g_optimizer = build_optimizer(config, member='gen', params=gen_params)
    critic_params = D.parameters()
    d_optimizer = build_optimizer(config, member='critic', params=critic_params)

    G = G.cuda()
    D = D.cuda()
    # load checkpoint
    d_checkpoint = dlcommon.utils.get_initial_checkpoint(config, member='d')
    g_checkpoint = dlcommon.utils.get_initial_checkpoint(config, member='g')
    if d_checkpoint is not None:
        assert g_checkpoint is not None
        last_epoch, step = dlcommon.utils.load_checkpoint(D, d_optimizer, d_checkpoint)
        _, _ =dlcommon.utils.load_checkpoint(G, g_optimizer, g_checkpoint)
        print('epoch, step:', last_epoch, step)
    else:
        last_epoch, step = -1, -1

    G, g_optimizer = to_data_parallel(config, G, g_optimizer)
    D, d_optimizer = to_data_parallel(config, D, d_optimizer)


    # build datasets
    dataloaders = build_dataloaders(config)

    # build summary writer
    writer = SummaryWriter(logdir=config.train.dir)
    logger_fn = hooks.logger_fn
    hooks.logger_fn = lambda **kwargs: logger_fn(writer=writer, **kwargs)

    # train loop
    train_gan(config=config,
            G=G, D=D,
            g_optimizer=g_optimizer,
            d_optimizer=d_optimizer,
            dataloaders=dataloaders,
            hooks=hooks,
            last_epoch=last_epoch+1)

    # Encoder training
    # build model
    E = build_model(config, hooks, member='encoder')

    # build optimizer
    enc_params = E.parameters()
    e_optimizer = build_optimizer(config, member='encoder', params=enc_params)

    E = E.cuda()

    e_checkpoint = dlcommon.utils.get_initial_checkpoint(config, member='e')
    if e_checkpoint is not None:
        last_epoch, step = dlcommon.utils.load_checkpoint(E, e_optimizer, e_checkpoint)
        print('epoch, step:', last_epoch, step)
    else:
        last_epoch, step = -1, -1

    E, e_optimizer = to_data_parallel(config, E, e_optimizer)

    train_encoder(config=config,
                G=G, D=D, E=E,
                e_optimizer=e_optimizer,
                dataloaders=dataloaders,
                hooks=hooks,
                last_epoch=last_epoch+1)
