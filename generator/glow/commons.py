import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from librosa.filters import mel as librosa_mel_fn
from audio_processing import dynamic_range_compression
from audio_processing import dynamic_range_decompression
from stft import STFT

def intersperse(lst, item):
  result = [item] * (len(lst) * 2 + 1)
  result[1::2] = lst
  return result

def mle_loss_full(z, m, logs, logdet, mask):
  l = torch.sum(logs) + 0.5 * torch.sum(torch.exp(-2 * logs) * ((z - m)**2)) # neg normal likelihood w/o the constant term
  l = l - torch.sum(logdet) # log jacobian determinant
  l = l / torch.sum(torch.ones_like(z) * mask) # averaging across batch, channel and time axes
  l = l + 0.5 * math.log(2 * math.pi) # add the remaining constant term
  return l

# +
def mle_loss(z, logdet, mask):
  l = 0.5 * torch.sum(z**2) # neg normal likelihood w/o the constant term
  l = l - torch.sum(logdet) # log jacobian determinant
  l = l / torch.sum(torch.ones_like(z) * mask) # averaging across batch, channel and time axes
  l = l + 0.5 * math.log(2 * math.pi) # add the remaining constant term
  return l

def split_feature(tensor, type="split"):
    """
    type = ["split", "cross"]
    """
    C = tensor.size(1)
    if type == "split":
      return tensor[:, : C // 2, ...], tensor[:, C // 2 :, ...]
    elif type == "cross":
      return tensor[:, 0::2, ...], tensor[:, 1::2, ...]

# -

# @torch.jit.script
def fused_add_tanh_sigmoid_multiply(input_a, input_b, n_channels):
  n_channels_int = n_channels[0]
  in_act = input_a + input_b
  t_act = torch.tanh(in_act[:, :n_channels_int, :])
  s_act = torch.sigmoid(in_act[:, n_channels_int:, :])
  acts = t_act * s_act
  return acts


def convert_pad_shape(pad_shape):
  l = pad_shape[::-1]
  pad_shape = [item for sublist in l for item in sublist]
  return pad_shape

def shift_1d(x):
  x = F.pad(x, convert_pad_shape([[0, 0], [0, 0], [1, 0]]))[:, :, :-1]
  return x

def sequence_mask(length, max_length=None):
  if max_length is None:
    max_length = length.max()
  x = torch.arange(max_length, dtype=length.dtype, device=length.device)
  return x.unsqueeze(0) < length.unsqueeze(1)

class TacotronSTFT(nn.Module):
  def __init__(self, filter_length=1024, hop_length=256, win_length=1024,
                 n_mel_channels=80, sampling_rate=22050, mel_fmin=0.0,
                 mel_fmax=8000.0):
    super(TacotronSTFT, self).__init__()
    self.n_mel_channels = n_mel_channels
    self.sampling_rate = sampling_rate
    self.stft_fn = STFT(filter_length, hop_length, win_length)
    mel_basis = librosa_mel_fn(
        sampling_rate, filter_length, n_mel_channels, mel_fmin, mel_fmax)
    mel_basis = torch.from_numpy(mel_basis).float()
    self.register_buffer('mel_basis', mel_basis)

  def spectral_normalize(self, magnitudes):
    output = dynamic_range_compression(magnitudes)
    return output

  def spectral_de_normalize(self, magnitudes):
    output = dynamic_range_decompression(magnitudes)
    return output

  def mel_spectrogram(self, y):
    """Computes mel-spectrograms from a batch of waves
    PARAMS
    ------
    y: Variable(torch.FloatTensor) with shape (B, T) in range [-1, 1]

    RETURNS
    -------
    mel_output: torch.FloatTensor of shape (B, n_mel_channels, T)
    """
    assert(torch.min(y.data) >= -1)
    assert(torch.max(y.data) <= 1)

    magnitudes, phases = self.stft_fn.transform(y)
    magnitudes = magnitudes.data
    mel_output = torch.matmul(self.mel_basis, magnitudes)
    mel_output = self.spectral_normalize(mel_output)
    return mel_output

def squeeze(x, x_mask=None, n_sqz=2):
  b, c, t = x.size()

  t = (t // n_sqz) * n_sqz
  x = x[:,:,:t]
  x_sqz = x.view(b, c, t//n_sqz, n_sqz)
  x_sqz = x_sqz.permute(0, 3, 1, 2).contiguous().view(b, c*n_sqz, t//n_sqz)
  
  if x_mask is not None:
    x_mask = x_mask[:,:,n_sqz-1::n_sqz]
  else:
    x_mask = torch.ones(b, 1, t//n_sqz).to(device=x.device, dtype=x.dtype)
  return x_sqz * x_mask, x_mask


def unsqueeze(x, x_mask=None, n_sqz=2):
  b, c, t = x.size()

  x_unsqz = x.view(b, n_sqz, c//n_sqz, t)
  x_unsqz = x_unsqz.permute(0, 2, 3, 1).contiguous().view(b, c//n_sqz, t*n_sqz)

  if x_mask is not None:
    x_mask = x_mask.unsqueeze(-1).repeat(1,1,1,n_sqz).view(b, 1, t*n_sqz)
  else:
    x_mask = torch.ones(b, 1, t*n_sqz).to(device=x.device, dtype=x.dtype)
  return x_unsqz * x_mask, x_mask

