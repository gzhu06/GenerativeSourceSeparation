import glob, os, shutil
import subprocess
import random
import argparse
from scipy.io.wavfile import read
import numpy as np
SR = 22050
MAX_WAV_VALUE = 32768.0
THRLD = 20.0
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("-resample", type=bool, default=False, help="reformat step")
parser.add_argument("-trim", type=bool, default=False, help="trim step")
parser.add_argument("-split", type=bool, default=True, help="split step")
args = parser.parse_args()

def resample_file(fileList, outFolder):
    """
    downsample musdb18 stem files into separate tracks 

    Args:
        fileList (list): a list of mp4 stem files
        outFolder (string): target folder to store separate audiofile tracks
    """

    # first downsampled to SR
    for inputAudiofile in fileList:
        
        outputAudiofolder = os.path.join(outFolder, inputAudiofile.split('/')[-3], inputAudiofile.split('/')[-2])
        os.makedirs(outputAudiofolder, exist_ok=True)

        outputAudiofile = os.path.join(outputAudiofolder, inputAudiofile.split('/')[-1])

        cmd = ['ffmpeg', '-i', inputAudiofile, '-ac', '1', '-af', 'aresample=resampler=soxr', '-ar', str(SR), outputAudiofile]
        completed_process = subprocess.run(cmd)
        
        # confrim process completed successfully
        assert completed_process.returncode == 0

#         # confirm new file has desired sample rate
#         assert soundfile.info(outputAudiofile).samplerate == SR
        
def trim_files(folder, saveFolder, instType):
    """
    trim silence from musdb18 wav files

    Args:
        folder (string): target folder to store separate audiofile tracks
        saveFolder (string):
        instType (string): instrument type
    """
    
    wav_files = glob.glob(folder + '/*/*/' + instType + '.wav')
    tarFolder = os.path.join(saveFolder, instType)
    os.makedirs(tarFolder, exist_ok=True)
    
    for audiofile in wav_files:
        audio_name = audiofile.split('/')[-2].strip().replace(" ", "") + '-' + instType
        
        trim_name = os.path.join(tarFolder, audio_name + '_trim.wav')
        trim_cmd = ['ffmpeg', '-i', audiofile, '-af',
                    'silenceremove=stop_periods=-1:stop_duration=1:stop_threshold=-50dB',
                    trim_name]
            
        completed_process = subprocess.run(trim_cmd)

def split_files(folder, saveFolder, instType):
    """
    split trimmed files into pieces

    Args:
        folder (string): target folder to store separate audiofile tracks
        saveFolder (string):
        instType (string): instrument type
    """
    import librosa
    wav_files = glob.glob(os.path.join(folder, instType) + '/*.wav')
    tarFolder = os.path.join(saveFolder, instType)
    os.makedirs(tarFolder, exist_ok=True)
    pieceDurations = []
    for audiofile in wav_files:
        audio_name = audiofile.split('/')[-1].split('.wav')[0] + '-' + instType
        totalDuration = librosa.get_duration(filename=audiofile, sr=SR)
        start_time = 0
        i = 0
        
        while True:

            duration = random.uniform(7.0, 10.0)
            end_time = start_time + duration
            
            if end_time > totalDuration:
                start_time = totalDuration - duration - 1.0
                end_time = totalDuration
            
            piece_name = os.path.join(tarFolder, audio_name + '-' + str(i) + '.wav')
            cut_cmd = ['ffmpeg', '-i', audiofile, '-ss', str(start_time), 
                       '-to', str(end_time), piece_name]
            completed_process = subprocess.run(cut_cmd)
            
            pieceDuration = librosa.get_duration(filename=piece_name, sr=SR)
            pieceDurations.append(pieceDuration)
            if pieceDuration < 3.0:
                os.remove(piece_name)
                
            start_time = end_time
            i += 1
            
            if start_time >= totalDuration:
                break
    print(pieceDurations)
    
def split_test_files(iptMixfolder, optFolder, sources=['drums.wav']):
    import librosa
    iptMixfiles = glob.glob(iptMixfolder + '/*/mixture.wav')
    segmentLens = []
    for iptMixfile in iptMixfiles:
        folderName, _ = os.path.split(iptMixfile)
        tarFolder = os.path.join(optFolder, folderName.split('/')[-1])
        os.makedirs(tarFolder, exist_ok=True)
    
        fsr = librosa.get_samplerate(iptMixfile)
        totalDuration = librosa.get_duration(filename=iptMixfile, sr=fsr)
        
        start_time = 0
        duration = 60.0
        i = 0

        while True:
            
            if start_time >= totalDuration:
                break
            
            # set time stamps for cutting
            end_time = start_time + duration
            if end_time + duration > totalDuration:
                end_time = totalDuration
                
            segmentLens.append(end_time-start_time)
            
            # cut segment
            i += 1
            if keep:
                mix_piece = os.path.join(tarFolder, 'mixture' + '-' + str(i) + '.wav')
                cut_cmd = ['ffmpeg', '-i', iptMixfile, '-ac', '1', '-ss', str(start_time), 
                           '-to', str(end_time), mix_piece]
                completed_process = subprocess.run(cut_cmd)

                for src in sources:
                    src_piece = os.path.join(tarFolder, src + '-' + str(i) + '.wav')
                    iptSrc = os.path.join(folderName, src + '.wav')
                    cut_cmd = ['ffmpeg', '-i', iptSrc,  '-ac', '1', '-ss', str(start_time), 
                               '-to', str(end_time), src_piece]
                    completed_process = subprocess.run(cut_cmd)
                start_time = end_time
            else:
                start_time = end_time
                continue
    print(segmentLens)

def main():

    musdbRoot = '/storage/ge/musdb18/musdb18_wav/'
    dataTypes = ['test']
    instsOfInterest = ['vocals', 'accompaniment']
    
    if args.resample:
        os.makedirs(outputFolder, exist_ok=True)
        resample_file(stemFiles, outputFolder)
    
    elif 'test' in dataTypes:
        musdbTestFolder = os.path.join(musdbRoot, 'raw/test')
        baselineTestFolder = '/storage/ge/musdb18/musdb18_wav/pieces/test_sv_separation_44k'
        split_test_files(musdbTestFolder, baselineTestFolder, instsOfInterest)
    else:
        for dataType in dataTypes:
            stemFiles = glob.glob(musdbRoot + dataType + '/*/*.wav')
            outputFolder = os.path.join(musdbRoot, 'processed', dataType)
            trimFolder = os.path.join(musdbRoot, 'trimmed', dataType)
            pieceFolder = os.path.join(musdbRoot, 'pieces', 'train_wavgan')
            if args.resample:
                os.makedirs(outputFolder, exist_ok=True)
                resample_file(stemFiles, outputFolder)

            if args.trim:
                os.makedirs(trimFolder, exist_ok=True)
                for inst in instsOfInterest:
                    trim_files(outputFolder, trimFolder, inst)

            if args.split:
                os.makedirs(pieceFolder, exist_ok=True)
                for inst in instsOfInterest:
                    split_files(trimFolder, pieceFolder, inst)

if __name__ == "__main__":
    main()