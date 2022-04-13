
# Generative Source Separation using Glow
Source separation as an inverse problem.  

## Introduction
Music source separation with both paired mixed signals and source signals has obtained substantial progress over the years. However, this setting highly relies on large amounts of paired data. Source-only supervision decouples the process of learning a mapping from a mixture to particular sources into a two stage paradigm: source modeling and separation. In this project, we leverage flow-based implicit generators to train music source priors and likelihood based objective to separate music mixtures.

<p align="center"><img align="center" src="./diagram.png", width=900></p>

## requirements
pytorch>=1.5.0\
tqdm\
librosa\
jupyter\
museval\
tqdm\
pandas

## Usage
    
### Inference
Run `inference_demo.py` for further details.

## Experimental Results on MusDB
| Method     |Backbone   |  Vocals  | Bass     |Drums     | Other    |
|------------|-----------|----------|----------|----------|----------|
| Demucs(v2) | Vanilla   |2.26/0.256|2.37/0.279|4.14/0.408|6.79/0.553|
| Conv-TasNet|ECAPA-C512 |1.74/0.167|1.76/0.186|3.01/0.272|6.61/0.461|
| Unmix      |ECAPA-C1024|1.38/0.146|1.51/0.152|2.7/0.254 |6.60/0.427|
| Wav-U-Net  | U-Net     |1.99/0.266|2.26/0.253|3.93/0.385|7.40/0.633|
| InstGlow   |Glow       |1.6 /0.154|1.66/0.173|2.86 /0.26|7.95/0.582|

## References



