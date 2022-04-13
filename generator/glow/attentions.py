import torch
from torch import nn
from torch.nn import functional as F
import modules

class CouplingBlock(nn.Module):
  def __init__(self, in_channels, hidden_channels, 
               kernel_size, dilation_rate, n_layers, 
               gin_channels=0, p_dropout=0, sigmoid_scale=False):
    super().__init__()
    self.in_channels = in_channels
    self.hidden_channels = hidden_channels
    self.kernel_size = kernel_size
    self.dilation_rate = dilation_rate
    self.n_layers = n_layers
    self.gin_channels = gin_channels
    self.p_dropout = p_dropout
    self.sigmoid_scale = sigmoid_scale

    start = torch.nn.Conv1d(in_channels//2, hidden_channels, 1)
    start = torch.nn.utils.weight_norm(start)
    self.start = start
    # Initializing last layer to 0 makes the affine coupling layers
    # do nothing at first.  It helps to stabilze training.
    end = torch.nn.Conv1d(hidden_channels, in_channels, 1)
    end.weight.data.zero_()
    end.bias.data.zero_()
    self.end = end

    self.wn = modules.WN(in_channels, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels, p_dropout)

  def forward(self, x, x_mask=None, reverse=False, g=None, **kwargs):
    b, c, t = x.size()
    if x_mask is None:
      x_mask = 1
    x_0, x_1 = x[:,:self.in_channels//2], x[:,self.in_channels//2:]

    x = self.start(x_0) * x_mask
    x = self.wn(x, x_mask, g)
    out = self.end(x)

    z_0 = x_0
    m = out[:, :self.in_channels//2, :]
    logs = out[:, self.in_channels//2:, :]
    if self.sigmoid_scale:
      logs = torch.log(1e-6 + torch.sigmoid(logs + 2))

    if reverse:
      z_1 = (x_1 - m) * torch.exp(-logs) * x_mask
      logdet = -torch.sum(logs * x_mask, [1, 2])
    else:
      z_1 = (m + torch.exp(logs) * x_1) * x_mask
      logdet = torch.sum(logs * x_mask, [1, 2])

    z = torch.cat([z_0, z_1], 1)
    return z, logdet

  def store_inverse(self):
    self.wn.remove_weight_norm()