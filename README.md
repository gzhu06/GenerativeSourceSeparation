
# Generative Source Separation using Glow
source separation as an inverse problem This model is trained using spectrogram features. 

## requirements
pytorch>=1.5.0\
tqdm\
librosa\
jupyter\
museval\
tqdm\
pandas

## Usage

### Model training

We provide uncontional generator training, conditional version is easy to implement based on unconditional version.

#### unconditional glow model
Go to corresponding generator folder, create json file similar to given template file in ./configs folder first</br>
Run init first to create list for training and test, then run: </br>
```
python init_musdb.py -m musdb -c ./configs/musdb.json
```
Then run train:</br>
```
python train_musdb.py -m musdb -c ./configs/musdb.json
```


## Pretrained models:
### dataset:

#### Music
MUSDB: https://sigsep.github.io/datasets/musdb.html 

### model


([Pretrained model folder](https://drive.google.com))

## References



