import torch
from torch import optim
from tqdm import tqdm
import os, glob
from scipy.io.wavfile import read
import torch.nn.functional as F
import random
import numpy as np
import inverse_utils
import generator.glow.commons as commons
EPSILON = torch.finfo(torch.float32).eps

def music_sep_batch(mixtures, genList, stft, optSpace, 
                    lr, sigma, alpha1, alpha2, iteration, 
                    mask=False, wiener=False, scheduler_step=800, 
                    scheduler_gamma=0.2):

    # freeze generators weights
    numGen = len(genList)
    for genUnc in genList:
        for param in genUnc.parameters():
            param.requires_grad = False
    
    # compute spectrogram from mixture and cancel the log
    mixSpecs, mixPhases = inverse_utils.get_spec(mixtures, stft) # 513 * T
    mixSpecs = F.pad(mixSpecs.unsqueeze(0), (0, 0, 0, 1), "constant", 0) # 514 * T
    batch_size = mixSpecs.shape[0]
    segLen = int(mixSpecs.shape[-1] / 2) * 2 # glow model requires even dimension
    segLenTensor = torch.LongTensor([segLen]*batch_size).cuda() 
    mixSpecs = mixSpecs[:, :, :segLen].cuda().requires_grad_(False)
    mixPhases = mixPhases[:, :, :segLen].requires_grad_(False)
    
    # zCol of shape (batch_size, num_sources, *spec_shape...)
    zCol = torch.randn((batch_size, numGen, mixSpecs.shape[-2], segLen), 
                       dtype=torch.float, device='cuda')
    zCol = (sigma * zCol).requires_grad_(True)
    
    optimizer = optim.Adam([zCol], lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

    # define loss and initialize data variables
    for i in tqdm(range(iteration)):
        xCol = []
        mixSynSpecs = 0
        logdets = []
        z_masks = []
        for j in range(numGen):

            gen = genList[j]
            xTemp, logdet, z_mask = gen(zCol[:, j, :, :], segLenTensor, gen=True)
            logdets.append(-logdet) # logdet in reverse gives log|dx/dz|, we want log|dz/dx|
            z_masks.append(z_mask)
            
            if mask:
                if i > 5:
                    maskTemp = torch.div(torch.sum(xTemp, dim=1), 
                                         torch.max(torch.sum(xTemp, dim=1), 
                                                   torch.sum(mixSpecs, dim=1))+1e-8).unsqueeze(1)
                else:
                    maskTemp = torch.ones((batch_size, 1, segLen), dtype=torch.float, device='cuda')
                mixSynSpecs += xTemp * maskTemp
                xCol.append(xTemp * maskTemp)
            else:
                mixSynSpecs += xTemp
                xCol.append(xTemp)

        mixSpecs = torch.abs(mixSpecs)  + 1e-8
        mixSynSpecs = torch.abs(mixSynSpecs)  + 1e-8
        
        loss_rec = (mixSpecs * torch.log(mixSpecs/mixSynSpecs + 1e-8) - mixSpecs + mixSynSpecs).mean()

        # regularization
        loss_r = 0.0
        for j in range(numGen):
            if optSpace == 'z':
                lss = 0.5 * torch.sum(zCol[:, j, :, :] ** 2) # neg normal likelihood w/o the constant term
                l_mle = lss / torch.sum(torch.ones_like(zCol[:, j, :, :]) * z_masks[j]) # averaging across batch, channel and time axes
            elif optSpace == 'x':
                l_mle = commons.mle_loss(zCol[:, j, :, :], logdets[j], z_masks[j]) # logdets and z_masks are first indexed by source_num
            loss_gs = [l_mle]
            loss_r += sum(loss_gs)
            
        loss = alpha1 * loss_rec + alpha2 * loss_r #+ 0.1 * loss_coh

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
    
    xCol_wiener = []
    if wiener:
        for i in range(len(xCol)): 
            xCol_wiener.append(torch.mul(torch.div(xCol[i], mixSynSpecs), mixSpecs))
    
        return torch.stack(xCol_wiener), mixPhases
    else:
        return torch.stack(xCol), mixPhases