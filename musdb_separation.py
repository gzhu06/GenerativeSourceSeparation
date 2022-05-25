import torch
from torch import optim
from tqdm import tqdm
import os, glob
from scipy.io.wavfile import read
import torch.nn.functional as F
import random
import numpy as np
import inverse_utils
from source_separation import music_sep_batch
import pickle

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
EPSILON = torch.finfo(torch.float32).eps
HPS = {}
optiObj = 'mle'
HPS['lr'] = 0.01
HPS['alpha1'] = 1.0
HPS['iteration'] = 150
HPS['optSpace'] = 'z'
HPS['sigma'] = 0.0
HPS['alpha2'] = 0.0 # 0.0 for z
if optiObj == 'map':
    HPS['optSpace'] = 'x'
    HPS['sigma'] = 0.1
    HPS['alpha2'] = 0.001 # 0.0 for z

TASK = {'sv':['vocals_torch', 'accompaniment_torch'],
        'music':['vocals_lr', 'bass_lr', 'drums_lr', 'other_lr']}

musdbTBRoot = '/storage/ge/musdb18/musdb18_wav/'
mixData = 'test_sv_separation'
epoch = 800
modelList = 'sv'
expName = modelList+'_'+str(epoch)+'_'+HPS['optSpace']+optiObj+'_'+str(HPS['iteration'])
glowRoot = os.path.join(musdbTBRoot, 'pieces', 'model_test', 'test_glow', 'exp2', expName + '_torch')
musdb18List = glob.glob(os.path.join(musdbTBRoot, 'pieces', mixData, '*/mixture*.wav'))
    
def predict_source(genList, stft, musdbMixture, sources, tarFolder):

    # read mixture
    sr, mixSample = read(musdbMixture)
    y_mix = np.expand_dims(mixSample, axis=0)
    
    # read sources
    srcs = []
    for src in sources:
        _, srcSample = read(src) 
        y_src = np.expand_dims(srcSample, axis=0)
        srcs.append(y_src)
    
    # step 1: source separation
    EstSrc, mixPhase = music_sep_batch(y_mix, genList, stft, **HPS)

    # step 2: resynthesizing from linear spectrogram and save GTs
    with open(os.path.join(tarFolder, 'est.pkl'), 'wb') as handle:
        pickle.dump(EstSrc, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    gt_dict = {}
    gt_dict['mix'] = y_mix
    gt_dict['mix_ang'] = mixPhase.cpu()
    for i, src in enumerate(srcs):
        gt_dict[str(i)] = src
    with open(os.path.join(tarFolder, 'gt.pkl'), 'wb') as handle:
        pickle.dump(gt_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":

    # load generators for source separation
    print('Load Glow Generators:...')
    genList = []
    labels = []
    for modelName in TASK[modelList]:
        genModel, STFTfunc = inverse_utils.load_glow(modelName=modelName, epoch=epoch)
        genList.append(genModel)
    
    random.shuffle(musdb18List)
    for musdbMixture in tqdm(musdb18List[:]):
        
        idx = musdbMixture.split('/')[-1].split('-')[-1].split('.wav')[0]
        folderName, _ = os.path.split(musdbMixture)
        tarFolder = os.path.join(glowRoot, musdbMixture.split('/')[-2], idx)
        os.makedirs(tarFolder, exist_ok=True)
        
        if os.path.exists(os.path.join(tarFolder, 'gt.pkl')):
            continue
        
        sources = []
        sourcePool =  TASK[modelList]
        
        for modelName in sourcePool:
            inst = modelName.split('_')[0]
            src_path = os.path.join(folderName, inst + '-' + idx + '.wav')
            sources.append(src_path)
        predict_source(genList, STFTfunc, musdbMixture, sources, tarFolder)
        