import random
import numpy as np
import torch
import torch.utils.data
import torch.nn.functional as F
import commons 
from utils import load_wav_to_torch, load_filepaths

EPSILON = torch.finfo(torch.float32).eps
class SpecLoader(torch.utils.data.Dataset):
    """
        1) loads audio, speaker_id, text pairs
        2) normalizes text and converts them to sequences of one-hot vectors
        3) computes linear-spectrograms from audio files.
    """
    def __init__(self, audiopaths, hparams):
        self.audiopaths = load_filepaths(audiopaths)
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.add_noise = hparams.add_noise
        self.stft = commons.TacotronSTFT(hparams.filter_length, hparams.hop_length, 
                                         hparams.win_length, hparams.n_mel_channels, 
                                         hparams.sampling_rate, hparams.mel_fmin,
                                         hparams.mel_fmax)

        random.seed(1234)
        random.shuffle(self.audiopaths)

    def get_spec(self, audiopath_):
        # separate filename, speaker_id and text
        # 513 * T
        audiopath = audiopath_[0]
        spec = self.compute_spec(audiopath)
        spec = F.pad(spec, (0, 0, 0, 1), "constant", 0.0)
        return spec

    def compute_spec(self, filename):
        audio, sampling_rate = load_wav_to_torch(filename)
        if sampling_rate != self.stft.sampling_rate:
            raise ValueError("{} {} SR doesn't match target {} SR".format(
                sampling_rate, self.stft.sampling_rate))
        if self.add_noise:
            audio = audio + torch.rand_like(audio)
        audio_norm = audio / self.max_wav_value
        audio_norm = audio_norm.unsqueeze(0)
        spec, _ = self.stft.stft_fn.transform(audio_norm)
        spec = torch.squeeze(spec, 0)
        return spec

    def __getitem__(self, index):
        return self.get_spec(self.audiopaths[index])

    def __len__(self):
        return len(self.audiopaths)

# +
class SpecCollate():
    """ Zero-pads model inputs and targets based on number of frames per step
    """
    def __init__(self, n_frames_per_step=1):
        self.n_frames_per_step = n_frames_per_step

    def __call__(self, batch):
        """Collate's training batch from linear-spectrogram
        PARAMS
        ------
        batch: [text_normalized, spec_normalized]
        """
        # Right zero-pad linear-spec
        spec_lengths, ids_sorted_decreasing = torch.sort(torch.LongTensor([x[0].shape[-1] for x in batch]), dim=0, descending=True)
        num_specs = batch[0].size(0)
        max_target_len = spec_lengths[0]
        if max_target_len % self.n_frames_per_step != 0:
            max_target_len += self.n_frames_per_step - max_target_len % self.n_frames_per_step
            assert max_target_len % self.n_frames_per_step == 0

        # include linear padded & sid
        spec_padded = torch.FloatTensor(len(batch), num_specs, max_target_len)
        spec_padded.zero_()
        output_lengths = torch.LongTensor(len(batch))
        for i in range(len(ids_sorted_decreasing)):
            spec = batch[ids_sorted_decreasing[i]]
            spec_padded[i, :, :spec.size(1)] = spec
            output_lengths[i] = spec.size(1)

        return spec_padded, output_lengths