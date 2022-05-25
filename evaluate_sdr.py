import os, glob
import torch
import numpy as np
from tqdm import tqdm
import pickle
import generator.glow.commons as commons
import generator.glow.utils as glowutils
import museval

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
modelDir = './generator/glow/logs/bass_lr'
resultFolder = '/storage/ge/musdb18/musdb18_wav/pieces/model_test/test_glow/exp2/music_2000_zmle_150'
hps = glowutils.get_hparams_from_dir(modelDir)
hparams = hps.data
stft = commons.TacotronSTFT(hparams.filter_length, 
                            hparams.hop_length,
                            hparams.win_length, 
                            hparams.sampling_rate)

MAX_WAV_VALUE = 32768.0
ENR_THRESHOLD = 20.0
FREQ_BIN = 513
numTracks = 4
gtList = glob.glob(os.path.join(resultFolder, '**/gt.pkl'), recursive=True)

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

if __name__ == "__main__":
    
    gtList = glob.glob(os.path.join(resultFolder, '**/gt.pkl'), recursive=True)

    total_sdr, energies = eval_musdb(gtList[:], resultFolder)
    total_inst = [i for i in range(numTracks)]

    for inst in total_inst:
        sdr_list = []
        enrg_list = []
        for i, sdr_seg in enumerate(total_sdr[inst]):
            for j, sdr in enumerate(sdr_seg):
                if not np.isnan(sdr) and energies[inst][i][j] > 20:
                    sdr_list.append(sdr)
                    enrg_list.append(energies[inst][i][j])
        
        print(inst)
        print(np.median(sdr_list))
        print(len(sdr_list))
        