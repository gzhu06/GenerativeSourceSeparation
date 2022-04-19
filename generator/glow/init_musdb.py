import os
import sys
sys.path.append(os.getcwd())
from glob import glob
import json
import argparse
import math
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
import pickle
from data_utils import SpecLoader, SpecCollate
import models
import utils
import random

def filelist_gen(hps, tarInst):
    """
        Generate file list for glow model training

    Args:
        hps (string): configurations for initialization, including dataset etc.
        audio_folder (): [description]
    
    """
    audiofiles = []

    for audioFolder in hps.data.audio_folders:
        audiofiles += glob(audioFolder + '**/*.wav', recursive=True)
        
    random.shuffle(audiofiles)
    
    filelist_dir = os.path.join('./filelists/', hps.model_dir.split('/')[-1])
    if not os.path.exists(filelist_dir):
        os.makedirs(filelist_dir, exist_ok=True)
    
    for i, audiofile in enumerate(audiofiles):
        
        instType = audiofile.split('/')[-2].lower()
        if instType == tarInst:
            if i < int(0.02 * len(audiofiles)):
                with open(os.path.join(filelist_dir, hps.data.validation_files), 'a') as fv:
                    fv.write("{0}\n".format(audiofile))         
            else:
                with open(os.path.join(filelist_dir, hps.data.training_files), 'a') as ft:
                    ft.write("{0}\n".format(audiofile))
    
    filelistDir = os.path.join('./filelists/', hps.model_dir.split('/')[-1])
    print(filelistDir)

class FlowGenerator_DDI(models.FlowGenerator):
    """A helper for Data-dependent Initialization"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        for f in self.decoder.flows:
            if getattr(f, "set_ddi", False):
                f.set_ddi(True)

def main():
    hps = utils.get_hparams()
    os.environ["CUDA_VISIBLE_DEVICES"] = hps.train.gpus
    
    filelist_dir = os.path.join('./filelists/', hps.model_dir.split('/')[-1])
    if len(glob(filelist_dir + '/*.txt')) == 0:
        tarInst = params.log_dir.split('/')[-1]
        filelist_gen(hps, tarInst)
    logger = utils.get_logger(hps.model_dir)
    logger.info(hps)

    torch.manual_seed(hps.train.seed)
    
    train_list = os.path.join(filelist_dir, hps.data.training_files)
    train_dataset = SpecLoader(train_list, hps.data)
    collate_fn = SpecCollate(1)

    train_loader = DataLoader(train_dataset, num_workers=1, shuffle=True,
                              batch_size=hps.train.batch_size, pin_memory=True,
                              drop_last=True, collate_fn=collate_fn)

    generator = FlowGenerator_DDI(n_speakers=1, out_channels=hps.data.n_ipt_channels,
                                  **hps.model).cuda()
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=hps.train.learning_rate)
    generator.train()
    for batch_idx, (x, x_lengths) in enumerate(train_loader):
        x, x_lengths = x.cuda(), x_lengths.cuda()
        _ = generator(x, x_lengths, gen=False)
        break
    utils.save_checkpoint(generator, optimizer_g, hps.train.learning_rate, 0,
                          os.path.join(hps.model_dir, "ddi_G.pth"))
    
if __name__ == "__main__":
    main()
