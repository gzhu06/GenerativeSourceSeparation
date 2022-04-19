import os, glob
import torch
import numpy as np
from tqdm import tqdm
import pickle
import generator.glow.commons as commons
import generator.glow.utils as glowutils
import museval

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
modelDir = './generator/unglow/logs/bass_spec'
# resultFolder = '/storage/ge/musdb18/musdb18_wav/pieces/model_test/test_glow/exp1/music_high_xmap/'
resultFolder = '/storage/ge/slakh/model_test/glow_high_sigma/'
hps = glowutils.get_hparams_from_dir(modelDir)
hparams = hps.data
stft = commons.TacotronSTFT(hparams.filter_length, hparams.hop_length,
                            hparams.win_length, hparams.n_mel_channels, 
                            hparams.sampling_rate, hparams.mel_fmin,
                            hparams.mel_fmax)

MAX_WAV_VALUE = 32768.0
ENR_THRESHOLD = 20.0
FREQ_BIN = 513
numTracks = 4
gtList = glob.glob(resultFolder + '**/gt.pkl', recursive=True)

def sdr_eval(inputMag, angArray, refSrc, frameSize):
    xEst = stft.stft_fn.inverse(inputMag.cpu(), angArray).cpu().numpy()[0]
    xEst = xEst / max(1.01 * np.max(np.abs(xEst)), 1)
    
    srcref = np.expand_dims(refSrc, (0, 2))
    
    xest = np.expand_dims((xEst*MAX_WAV_VALUE).astype(np.int16), (0, 2))
    result = museval.evaluate(srcref, xest, win=frameSize, hop=frameSize, 
                              mode='v4', padding=True)
    return result[0][0], xEst

def eval_musdb(gtList, estFolder):
    ## musdb
    total_sdr = [[] for i in range(numTracks)]
    energies = [{} for i in range(numTracks)]
    
    for i, gtPkl in enumerate(tqdm(gtList[:])):
        gt = pickle.load(open(gtPkl, "rb" ))
        estFolder, _ = os.path.split(gtPkl)
        estPath = os.path.join(estFolder, 'est.pkl')
        est = pickle.load(open(estPath, "rb" ))

        mix_ang = gt['mix_ang']
        for j in range(numTracks):

            if str(j) not in gt:
                continue

            energies[j][i] = []
            src = gt[str(j)][0]
            frame_size = len(src)
            steps = int((len(src)/frame_size))
            for idx in range(steps):
                unitWav = np.zeros(frame_size)
                unitWav = src[idx*frame_size: (idx+1)*frame_size]
                unitEnergy = np.inner(unitWav/MAX_WAV_VALUE, unitWav/MAX_WAV_VALUE)
                energies[j][i].append(unitEnergy)

            estSrc = est[j]
            sdrs, xEst = sdr_eval(estSrc[:, :FREQ_BIN], mix_ang, src, frame_size)
    #         ipd.display(ipd.Audio(src, rate=22050))
    #         ipd.display(ipd.Audio(xEst, rate=22050))
            assert len(energies[j][i]) == len(sdrs)
            total_sdr[j].append(sdrs)
            
    return total_sdr, energies

def eval_slakh(gtList, estFolder):
    
    ## slakh
    TRACKNAME = ['Bass', 'Drums', 'Guitar', 'Piano']
    # TRACKNAME = ['Drums', 'Piano', 'Bass']
    total_sdr = {}
    energies = {}
    for track in TRACKNAME:
        total_sdr[track] = []
        energies[track] = []
    for i, gtPkl in enumerate(tqdm(gtList[:])):
        gt = pickle.load(open(gtPkl, "rb" ))
        estFolder, _ = os.path.split(gtPkl)
        estPath = os.path.join(estFolder, 'est.pkl')
        est = pickle.load(open(estPath, "rb" ))

        mix_ang = gt['mix_ang']
        for j, track in enumerate(TRACKNAME):

            if track not in gt:
                continue

            src = gt[track]
            frame_size = len(src)
            steps = int((len(src)/frame_size))
            for idx in range(steps):
                unitWav = np.zeros(frame_size)
                unitWav = src[idx*frame_size: (idx+1)*frame_size]
                unitEnergy = np.inner(unitWav/MAX_WAV_VALUE, unitWav/MAX_WAV_VALUE)
                energies[track].append(unitEnergy)

            estSrc = est[j]
            sdrs, xEst = sdr_eval(estSrc[:, :FREQ_BIN], mix_ang, src, frame_size)
            total_sdr[track].append(sdrs)

    return total_sdr, energies

if __name__ == "__main__":
    
    gtList = glob.glob(resultFolder + '**/gt.pkl', recursive=True)

    if 'musdb' in resultFolder:
        total_sdr, energies = eval_musdb(gtList[:], resultFolder)
        total_inst = [i for i in range(numTracks)]
    elif 'slakh' in resultFolder:
        total_sdr, energies = eval_slakh(gtList, resultFolder)
        total_inst = ['Bass', 'Drums', 'Guitar', 'Piano']

    for inst in total_inst:
        sdr_list = []
        enrg_list = []
        if 'musdb' in resultFolder:
            for i, sdr_seg in enumerate(total_sdr[inst]):
                for j, sdr in enumerate(sdr_seg):
                    if not np.isnan(sdr) and energies[inst][i][j] > 20:
                        sdr_list.append(sdr)
                        enrg_list.append(energies[inst][i][j])
        elif 'slakh' in resultFolder:
            for i, sdr_seg in enumerate(total_sdr[inst]):
                sdr = sdr_seg[0]
                if not np.isnan(sdr) and energies[inst][i] > 20:
                    sdr_list.append(sdr)
                    enrg_list.append(energies[inst][i])
        
        print(inst)
        print(np.median(sdr_list))
        print(len(sdr_list))
        