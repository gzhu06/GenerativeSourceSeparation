import os
import json
import argparse
import math
import torch
import pickle
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
import torch.distributed as dist
from apex.parallel import DistributedDataParallel as DDP
from apex import amp
import numpy as np

from torch.optim import Adam, lr_scheduler
from data_utils import SpecLoader, SpecCollate
import models
import commons
import utils

global_step = 0

def main():
    """Assume Single Node Multi GPUs Training Only"""
    hps = utils.get_hparams()
    print('assign GPU...')
    os.environ["CUDA_VISIBLE_DEVICES"] = hps.train.gpus
    
    assert torch.cuda.is_available(), "CPU training is not allowed."
    n_gpus = torch.cuda.device_count()

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(np.random.randint(low=60000, high=70000))

    mp.spawn(train_and_eval, nprocs=n_gpus, args=(n_gpus, hps))

def train_and_eval(rank, n_gpus, hps):
    
    filelist_dir = os.path.join('./filelists/', hps.model_dir.split('/')[-1])

    global global_step
    if rank == 0:
        logger = utils.get_logger(hps.model_dir)
        logger.info(hps)
        writer = SummaryWriter(log_dir=hps.model_dir)
        writer_eval = SummaryWriter(log_dir=os.path.join(hps.model_dir, "eval"))

    dist.init_process_group(backend='nccl', init_method='env://', world_size=n_gpus, rank=rank)
    torch.manual_seed(hps.train.seed)
    torch.cuda.set_device(rank)
    
    train_list = os.path.join(filelist_dir, hps.data.training_files)
    train_dataset = SpecLoader(train_list, hps.data)
    collate_fn = SpecCollate(1)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,
                                                                    num_replicas=n_gpus,
                                                                    rank=rank,
                                                                    shuffle=True)
    
    train_loader = DataLoader(train_dataset, num_workers=8, shuffle=False,
                              batch_size=hps.train.batch_size, pin_memory=True,
                              drop_last=True, collate_fn=collate_fn, sampler=train_sampler)
    if rank == 0:
        validation_list = os.path.join(filelist_dir, hps.data.validation_files)
        val_dataset = SpecLoader(validation_list, hps.data)
        val_loader = DataLoader(val_dataset, num_workers=8, shuffle=False,
                                batch_size=hps.train.batch_size, pin_memory=True,
                                drop_last=True, collate_fn=collate_fn)

    generator = models.FlowGenerator(n_speakers=1, 
                                     out_channels=hps.data.n_ipt_channels,
                                     **hps.model).cuda(rank)
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=hps.train.learning_rate)
    scheduler = lr_scheduler.StepLR(optimizer_g, hps.train.decay_steps, 
                                    gamma=hps.train.decay_rate, last_epoch=-1)
    if hps.train.fp16_run:
        generator, optimizer_g = amp.initialize(generator, optimizer_g, opt_level="O1")
    generator = DDP(generator)
    epoch_str = 1
    global_step = 0
    
    try:
        _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "G_*.pth"), generator, optimizer_g)
        epoch_str += 1
        global_step = (epoch_str - 1) * len(train_loader)
        decay_step = int(global_step / hps.train.decay_steps)
        for opt_g in optimizer_g.param_groups:
            opt_g['lr'] = hps.train.learning_rate * np.power(hps.train.decay_rate, decay_step)
    except:
        print('No checkpoint found!')

    for epoch in range(epoch_str, hps.train.epochs + 1):
        if rank==0:
            train(rank, epoch, hps, generator, optimizer_g, scheduler, train_loader, logger, writer)
            evaluate(rank, epoch, hps, generator, optimizer_g, val_loader, logger, writer_eval)
            if epoch % hps.train.log_interval == 0:
                utils.save_checkpoint(generator, optimizer_g, scheduler.get_lr(), 
                                      epoch, os.path.join(hps.model_dir, "G_{}.pth".format(epoch)))
        else:
            train(rank, epoch, hps, generator, optimizer_g, scheduler, train_loader, None, None)

def train(rank, epoch, hps, generator, optimizer_g, scheduler, train_loader, logger, writer):
    train_loader.sampler.set_epoch(epoch)
    global global_step

    generator.train()
    for batch_idx, (y, y_lengths) in enumerate(train_loader):
        y, y_lengths = y.cuda(rank, non_blocking=True), y_lengths.cuda(rank, non_blocking=True)

        # Train Generator
        optimizer_g.zero_grad()
        
        z, logdet, z_mask = generator(y, y_lengths, gen=False)
        l_mle = commons.mle_loss(z, logdet, z_mask)

        loss_gs = [l_mle]
        loss_g = sum(loss_gs)

        with amp.scale_loss(loss_g, optimizer_g) as scaled_loss:
            scaled_loss.backward()
        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer_g), 4)
        optimizer_g.step()
        
        if rank==0:
            if batch_idx % hps.train.log_interval == 0:
                logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tTrainLoss: {:.6f}'.format(
                    epoch, batch_idx * len(y), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader),
                    loss_g.item()))
                logger.info([x.item() for x in loss_gs] + [global_step, scheduler.get_lr()])

        global_step += 1
        scheduler.step()
    
    if rank == 0:
        logger.info('====> Epoch: {}'.format(epoch))

def evaluate(rank, epoch, hps, generator, optimizer_g, val_loader, logger, writer_eval):
    if rank == 0:
        global global_step
        generator.eval()
        losses_tot = []
        with torch.no_grad():
            for batch_idx, (y, y_lengths) in enumerate(val_loader):
                y, y_lengths = y.cuda(rank, non_blocking=True),y_lengths.cuda(rank,non_blocking=True)

                z, logdet, z_mask = generator(y, y_lengths, gen=False)
                l_mle = commons.mle_loss(z, logdet, z_mask)

                loss_gs = [l_mle]
                loss_g = sum(loss_gs)

                if batch_idx == 0:
                    losses_tot = loss_gs
                else:
                    losses_tot = [x + y for (x, y) in zip(losses_tot, loss_gs)]

                if batch_idx % hps.train.log_interval == 0:
                    logger.info('Eval Epoch: {} [{}/{} ({:.0f}%)]\tEvalLoss: {:.6f}'.format(
                        epoch, batch_idx * len(y), len(val_loader.dataset),
                        100. * batch_idx / len(val_loader),
                        loss_g.item()))
                    logger.info([x.item() for x in loss_gs])
                     
        losses_tot = [x/len(val_loader) for x in losses_tot]
        loss_tot = sum(losses_tot)
        scalar_dict = {"loss/g/total": loss_tot}
        scalar_dict.update({"loss/g/{}".format(i): v for i, v in enumerate(losses_tot)})
        utils.summarize(writer=writer_eval,
                        global_step=global_step, 
                        scalars=scalar_dict)
        logger.info('====> Epoch: {}'.format(epoch))

if __name__ == "__main__":
    main()
