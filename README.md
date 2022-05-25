
# Generative Source Separation using Glow
Source separation as an inverse problem. Open source code for the paper 'Music Source Separation with Generative Flow' [arxiv](https://arxiv.org/abs/2204.09079). Demo page is [here](https://airlabur.github.io/gss/).

## Introduction
Music source separation with both paired mixed signals and source signals has obtained substantial progress over the years. However, this setting highly relies on large amounts of paired data. Source-only supervision decouples the process of learning a mapping from a mixture to particular sources into a two stage paradigm: source modeling and separation. In this project, we leverage flow-based implicit generators to train music source priors and likelihood based objective to separate music mixtures.

<p align="center"><img align="center" src="./diagram.png", width=900></p>

## requirements
pytorch>=1.10.0\
tqdm\
librosa\
jupyter\
museval\
tqdm\
pandas

## Usage
Download our pretrained [checkpoints](https://drive.google.com/file/d/16_L8-f1mYZ7oHnoxDpVTjAEpDHeBEb2y/view?usp=sharing), then run inference on any audio files.
    
### Inference
There are two examples in `inference_demo.ipynb`, you can also preview these samples in the `Bonus tracks` from the [demo](https://airlabur.github.io/gss/) page. You can also try your own music mixture wav files. In our framework, it's able to process relatively long audio segments (even over 1 minute).

### Reproducing experimental results in the paper
First prepare test portion of MUSDB18 using `musdb18_data_prep.py` in `preprocessing` folder. Then run `musdb_spearation.py` and `evaluate_sdr.py` with defined model checkpoint path and parameters.

## Experimental Results on MusDB
| Method     |Backbone   |  Vocals  | Bass     |Drums     | Other    |
|------------|-----------|----------|----------|----------|----------|
| Demucs(v2) | U-Network |7.14      |5.50      |6.74      |4.16      |
| Conv-TasNet|TCN        |7.00      |4.19      |5.25      |3.94      |
| Open Unmix |BiLSTM     |6.86      |4.88      |6.35      |3.86      |
| Wave-U-Net  | U-Network |5.06      |2.63      |3.74      |1.95      |
| InstGlow   |Glow       |3.92      |2.58      |3.85      |2.37      |

## References
[GlowTTS](https://github.com/jaywalnut310/glow-tts)

