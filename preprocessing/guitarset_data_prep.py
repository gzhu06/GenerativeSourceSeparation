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

def resample_file(fileList, outputAudioFolder):
    """
    downsample musdb18 stem files into separate tracks 

    Args:
        fileList (list): a list of mp4 stem files
        outFolder (string): target folder to store separate audiofile tracks
    """

    # first downsampled to SR
    for inputAudiofile in fileList:
        
        os.makedirs(outputAudioFolder, exist_ok=True)
        outputAudiofile = os.path.join(outputAudioFolder, inputAudiofile.split('/')[-1])

        cmd = ['ffmpeg', '-i', inputAudiofile, '-ac', '1', '-af', 'aresample=resampler=soxr', '-ar', str(SR), outputAudiofile]
        completed_process = subprocess.run(cmd)
        
        # confrim process completed successfully
        assert completed_process.returncode == 0

#         # confirm new file has desired sample rate
#         assert soundfile.info(outputAudiofile).samplerate == SR
        
def trim_files(folder, tarFolder):
    """
    trim silence from musdb18 wav files

    Args:
        folder (string): target folder to store separate audiofile tracks
        saveFolder (string):
        instType (string): instrument type
    """
    
    wav_files = glob.glob(os.path.join(folder, '*.wav'))
    os.makedirs(tarFolder, exist_ok=True)
    
    for audiofile in wav_files:
        audio_name = audiofile.split('/')[-1]
        trim_name = os.path.join(tarFolder, audio_name + '_trim.wav')
        trim_cmd = ['ffmpeg', '-i', audiofile, '-af',
                    'silenceremove=stop_periods=-1:stop_duration=1:stop_threshold=-50dB',
                    trim_name]
            
        completed_process = subprocess.run(trim_cmd)

def split_files(folder, tarFolder):
    """
    split trimmed files into pieces

    Args:
        folder (string): target folder to store separate audiofile tracks
        saveFolder (string):
    """
    import librosa
    wav_files = glob.glob(os.path.join(folder, '*.wav'))
    os.makedirs(tarFolder, exist_ok=True)
    pieceDurations = []
    for audiofile in wav_files:
        audio_name = audiofile.split('/')[-1].split('.wav')[0]
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
            if pieceDuration < 2.0:
                os.remove(piece_name)
                
            start_time = end_time
            i += 1
            
            if start_time >= totalDuration:
                break
    print(pieceDurations)

def main():

    datasetRoot = '/storage/ge/guitarset/'
    outputFolder = os.path.join(datasetRoot, 'mono_wav')
    trimFolder = os.path.join(datasetRoot, 'trimmed')
    pieceFolder = os.path.join(datasetRoot, 'pieces')
    stemFiles = glob.glob(os.path.join(datasetRoot, 'audio_mono-mic', '*.wav'), recursive=True)
    
    if args.resample:
        os.makedirs(outputFolder, exist_ok=True)
        resample_file(stemFiles, outputFolder)

    if args.trim:
        
        os.makedirs(trimFolder, exist_ok=True)
        trim_files(outputFolder, trimFolder)

    if args.split:
        os.makedirs(pieceFolder, exist_ok=True)
        split_files(trimFolder, pieceFolder)

if __name__ == "__main__":
    main()