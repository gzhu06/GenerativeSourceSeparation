import math
import torch
from torch import nn
from torch.nn import functional as F

import commons
import modules
import attentions

class FlowSpecDecoder(nn.Module):
  def __init__(self, 
      in_channels, 
      hidden_channels, 
      kernel_size, 
      dilation_rate, 
      n_blocks, 
      n_layers, 
      p_dropout=0., 
      n_split=4,
      n_sqz=2,
      sigmoid_scale=False,
      gin_channels=0):
    super().__init__()

    self.in_channels = in_channels
    self.hidden_channels = hidden_channels
    self.kernel_size = kernel_size
    self.dilation_rate = dilation_rate
    self.n_blocks = n_blocks
    self.n_layers = n_layers
    self.p_dropout = p_dropout
    self.n_split = n_split
    self.n_sqz = n_sqz
    self.sigmoid_scale = sigmoid_scale
    self.gin_channels = gin_channels

    self.flows = nn.ModuleList()
    for b in range(n_blocks):
      self.flows.append(modules.ActNorm(channels=in_channels * n_sqz))
      self.flows.append(modules.InvConvNear(channels=in_channels * n_sqz, n_split=n_split))
      self.flows.append(attentions.CouplingBlock(in_channels * n_sqz,
                                                 hidden_channels,
                                                 kernel_size=kernel_size, 
                                                 dilation_rate=dilation_rate,
                                                 n_layers=n_layers,
                                                 gin_channels=gin_channels,
                                                 p_dropout=p_dropout,
                                                 sigmoid_scale=sigmoid_scale))

  def forward(self, x, x_mask, g=None, reverse=False):
    
    if not reverse:
      flows = self.flows
    else:
      flows = reversed(self.flows)
        
    logdet_tot = 0

    if self.n_sqz > 1:
      x, x_mask = commons.squeeze(x, x_mask, self.n_sqz)
    for f in flows:
        x, logdet = f(x, x_mask, g=g, reverse=reverse)
        logdet_tot += logdet
    if self.n_sqz > 1:
      x, x_mask = commons.unsqueeze(x, x_mask, self.n_sqz)
    return x, logdet_tot

  def store_inverse(self):
    for f in self.flows:
      f.store_inverse()

class FlowGenerator(nn.Module):
  def __init__(self, 
      hidden_channels, 
      out_channels,
      n_blocks_dec=12, 
      kernel_size_dec=5, 
      dilation_rate=5, 
      n_block_layers=4,
      p_dropout_dec=0., 
      n_speakers=0, 
      gin_channels=0, 
      n_split=4,
      n_sqz=1,
      sigmoid_scale=False,
      block_length=None,
      hidden_channels_dec=None,
      **kwargs):

    super().__init__()
    self.hidden_channels = hidden_channels
    self.out_channels = out_channels
    self.n_blocks_dec = n_blocks_dec
    self.kernel_size_dec = kernel_size_dec
    self.dilation_rate = dilation_rate
    self.n_block_layers = n_block_layers
    self.p_dropout_dec = p_dropout_dec
    self.n_speakers = n_speakers
    self.gin_channels = gin_channels
    self.n_split = n_split
    self.n_sqz = n_sqz
    self.sigmoid_scale = sigmoid_scale
    self.hidden_channels_dec = hidden_channels_dec

    self.decoder = FlowSpecDecoder(
        out_channels, 
        hidden_channels_dec or hidden_channels, 
        kernel_size_dec, 
        dilation_rate, 
        n_blocks_dec, 
        n_block_layers, 
        p_dropout=p_dropout_dec, 
        n_split=n_split,
        n_sqz=n_sqz,
        sigmoid_scale=sigmoid_scale,
        gin_channels=gin_channels)

    if n_speakers > 1:
      self.emb_g = nn.Embedding(n_speakers, gin_channels)
      nn.init.uniform_(self.emb_g.weight, -0.1, 0.1)

  def forward(self, y, y_lengths, g=None, gen=False, noise_scale=1., length_scale=1.):
    if g is not None:
      g = F.normalize(self.emb_g(g)).unsqueeze(-1) # [b, h]

    y_max_length = y.size(2)
    y, y_lengths, y_max_length = self.preprocess(y, y_lengths, y_max_length)
    z_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, y_max_length), 1).to(torch.float32)
    if gen:
      mel_gen, logdet = self.decoder(y, z_mask, g=g, reverse=True)
      return mel_gen, logdet, z_mask # added logdet zmask
    else:
      z, logdet = self.decoder(y, z_mask, g=g, reverse=False)
      return z, logdet, z_mask

  def preprocess(self, y, y_lengths, y_max_length):
    if y_max_length is not None:
      y_max_length = (y_max_length // self.n_sqz) * self.n_sqz
      y = y[:,:,:y_max_length]
    y_lengths = (y_lengths // self.n_sqz) * self.n_sqz
    return y, y_lengths, y_max_length

  def store_inverse(self):
    self.decoder.store_inverse()
