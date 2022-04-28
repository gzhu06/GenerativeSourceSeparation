import torch
from torch import optim
from tqdm import tqdm
import os
import json
import librosa
import torch.nn.functional as F
import random
import numpy as np
import pickle
import scipy

import generator.glow.utils as glowutils
import generator.glow.models as glowmodels
import generator.glow.commons as commons

MAX_WAV_VALUE = 32768.0
        
def get_spec(audio, stft):
    audio = torch.FloatTensor(audio.astype(np.float32))
    audio_norm = audio / MAX_WAV_VALUE

    mag, pha = stft.stft_fn.transform(audio_norm)
    spec = mag.data
    spec = torch.squeeze(spec, 0)
    return spec, pha

def load_glow(modelName="musdb"):
    
    glowFolder = './generator/glow/logs/'
    modelDir = os.path.join(glowFolder, modelName)
    hps = glowutils.get_hparams_from_dir(modelDir)
    checkpointPath = glowutils.latest_checkpoint_path(modelDir)

    generator = glowmodels.FlowGenerator(n_speakers=numObj, out_channels=hps.data.n_ipt_channels,
                                         **hps.model).cuda()
    glowutils.load_checkpoint(checkpointPath, generator)
    generator.eval()
    
    # import stft operator
    hparams = hps.data
    stft = commons.TacotronSTFT(hparams.filter_length, hparams.hop_length, 
                                hparams.win_length, hparams.n_mel_channels, 
                                hparams.sampling_rate, hparams.mel_fmin,
                                hparams.mel_fmax)
    return generator, stft

def resynthesize_from_spec(specIPT, mixWav, stft):

    # extract phase from original input
    mixTensor = torch.FloatTensor(mixWav.astype(np.float32))
    _, mixPhase = stft.stft_fn.transform(mixTensor.unsqueeze(0))
    
    # get STFT
    xEst = stft.stft_fn.inverse(specIPT.unsqueeze(0), mixPhase).cpu().numpy()[0]
    return xEst

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    load_glow()
    
if __name__ == "__main__":
    main()
    