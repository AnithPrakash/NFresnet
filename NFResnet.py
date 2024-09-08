import torch
from torch import nn

from function.base import WSCon2d, ScaledStdConv2d


activation_fn = {
    'identity': lambda x, *args, **kwargs: nn.Identity(*args, **kwargs)(x) * _nonlin_gamma['identity'],
    'celu': lambda x, *args, **kwargs: nn.CELU(*args, **kwargs)(x) * _nonlin_gamma['celu'],
    'elu': lambda x, *args, **kwargs: nn.ELU(*args, **kwargs)(x) * _nonlin_gamma['elu'],
    'gelu': lambda x, *args, **kwargs: nn.GELU(*args, **kwargs)(x) * _nonlin_gamma['gelu'],
    'leaky_relu': lambda x, *args, **kwargs: nn.LeakyReLU(*args, **kwargs)(x) * _nonlin_gamma['leaky_relu'],
    'log_sigmoid': lambda x, *args, **kwargs: nn.LogSigmoid(*args, **kwargs)(x) * _nonlin_gamma['log_sigmoid'],
    'log_softmax': lambda x, *args, **kwargs: nn.LogSoftmax(*args, **kwargs)(x) * _nonlin_gamma['log_softmax'],
    'relu': lambda x, *args, **kwargs: nn.ReLU(*args, **kwargs)(x) * _nonlin_gamma['relu'],
    'relu6': lambda x, *args, **kwargs: nn.ReLU6(*args, **kwargs)(x) * _nonlin_gamma['relu6'],
    'selu': lambda x, *args, **kwargs: nn.SELU(*args, **kwargs)(x) * _nonlin_gamma['selu'],
    'sigmoid': lambda x, *args, **kwargs: nn.Sigmoid(*args, **kwargs)(x) * _nonlin_gamma['sigmoid'],
    'silu': lambda x, *args, **kwargs: nn.SiLU(*args, **kwargs)(x) * _nonlin_gamma['silu'],
    'softplus': lambda x, *args, **kwargs: nn.Softplus(*args, **kwargs)(x) * _nonlin_gamma['softplus'],
    'tanh': lambda x, *args, **kwargs: nn.Tanh(*args, **kwargs)(x) * _nonlin_gamma['tanh'],
}



class SqueezeExcite(nn.Module):
  
  def __init__(self, in_channels, out_channels, se_ratio=0.5, hidden_channels=None, activation='relu'):
    assert (se_ratio != None) or ((se_ratio is None) and (hidden_channels is not None))
    
    if se_ratio is None:
      hidden_channels = hidden_channels
    else:
      hidden_channels = max(1, se_ratio * in_channels)
      
    self.fc0 = nn.Linear(in_channels, hidden_channels)
    self.fc1 = nn.Linear(hidden_channels, out_channels)
    
    self.activation  = activation_fn[activation]
    super(SqueezeExcite, self).__init__()
    
  def forward(self, x):
    h = torch.mean(x, [2,3])
    h = self.fc0(h)
    h = self.fc1(self.activation(h))
    
    return h.expand_as(x)
  
  
class NFBlock(nn.Module):
  
  def __init__(self, in_channels, out_channels, expansion=0.5, se_ratio=0.5, kernel_shape=3, group_size=128, stride=1, beta=1.0, alpha=0.2, conv=ScaledStdConv2d, activation='gelu'):
    
    width = int(self.out_channels * expansion)
    self.groups = width // group_size
    self.width = group_size * self.groups
    
    self.conv0 = conv(in_channels, self.width, 1)
    
    self.conv1 = conv(self.width, self.width, 3, groups=self.groups)
    
    self.alpha = alpha
    self.beta = beta