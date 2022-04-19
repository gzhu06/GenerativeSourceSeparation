
# Glow
Unconditional audio generator Glow 

## Introduction
We provide uncontional generator training for modeling source priors, conditional version is easy to implement based on unconditional version. In Glow generator training, we use apex from NVIDIA.

## Usage

### Model training

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
Pretrained models and configurations will be released soon.
([Pretrained model folder](https://drive.google.com))


## References



