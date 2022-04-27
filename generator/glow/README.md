
# Glow
Unconditional audio generator Glow 

## Introduction
We provide uncontional generator training using MUSDB18 for modeling source priors. In Glow generator training, we use apex from NVIDIA.

## Usage
### Dataset:
MUSDB: https://sigsep.github.io/datasets/musdb.html 

### Model training

In data preparation, we use `musdb18_data_prep.py` to define instruments to train source priors and train/test split. Also prepare audio data segments with resampling, trimming and splitting the original MUSDB18 audio files.

Before training, create json file similar to given template file in ./configs folder first</br>
Run `init_musdb.py` first to create list for training and test: </br>
```
python init_musdb.py -m {musdb} -c ./configs/{musdb}.json
```
Then run train:</br>
```
python train_musdb.py -m {musdb} -c ./configs/{musdb}.json
```

Replace {musdb} with one of instruments previously defined in `musdb18_data_prep.py`.

## Pretrained models:

### model
Download this file, unzip and move every folder to `./logs`
([Pretrained model folder](https://drive.google.com/file/d/16_L8-f1mYZ7oHnoxDpVTjAEpDHeBEb2y/view?usp=sharing))

## References

[GlowTTS](https://github.com/jaywalnut310/glow-tts)
