import torch 
from torch import nn
from torch.functional import F
from torch import Tensor

from typing import Optional, List, Tuple

# this code defines a custom 1d conv layer class and help to normalize the weights of the convolutional layers


class WSConv1d(nn.Conv1d):
    r"""applies a 1D convolutional over an input signal composed of several input
    planes.
    This module supports
    * :attr:`stride` controls the stride for the cross-correlation, a single
      number or a one-element tuple.
    * :attr:`padding` controls the amount of implicit zero-paddings on both sides
      for :attr:`padding` number of points.
    * :attr:`dilation` controls the spacing between the kernel points; also
      known as the à trous algorithm. It is harder to describe, but this `link`_
      has a nice visualization of what :attr:`dilation` does.
    * :attr:`groups` controls the connections between inputs and outputs.
      :attr:`in_channels` and :attr:`out_channels` must both be divisible by
      :attr:`groups`. For example,
        * At groups=1, all inputs are convolved to all outputs.
        * At groups=2, the operation becomes equivalent to having two conv
          layers side by side, each seeing half the input channels,
          and producing half the output channels, and both subsequently
          concatenated.
        * At groups= :attr:`in_channels`, each input channel is convolved with
          its own set of filters,
    """
    def __init__(self,in_channels, out_channels, kernal_size, stride=1, padding=0, dilation=1, groups=1, bais=True, padding_mode='zeros'):
        super().__init__(in_channels,out_channels, kernal_size,stride=stride, padding=padding,
                         dilation=dilation, groups=groups, bais=bais, padding_mode=padding_mode)
        nn.init.kaiming_normal_(self.weight)
        self.gain = nn.Parameter(torch.ones(
            self.weight.size()[0], requires_grad=True
        ))

    # the function help to calculate the mean and variance of the weight 
    def standardize_weights(self,eps):
        mean=torch.mean(self.weight, dim=(1,2), keepdim=True)
        var= torch.std(self.weight, dim=(1,2), keepdim=True, unbiased=False)**2
        fan_in= torch.prod(torch.tensor(self.weight.shape))
 
        #fan in : cal the product of the weight dimensions 
        scale= torch.rsqrt(torch.max(
            var * fan_in,torch.tensor(eps).to(var.device))) * self.gain.view_as(var).to(var.device)
        #scale and shift : compute the scaling factor and shift for weights standardization
        shift = mean * scale 
        return self.weight * scale - shift 
    
    def forward(self, input, eps=1e-4):
        weight=self.standardize_weights(eps)
        return F.conv1d(input,weight,self.bias,self.stride,self.padding,self.dilation,self.groups)
    
class WSCon2d(nn.Conv2d):
    """Applies a 2D convo over an input signal composed 
    of several input planes after wieght normalization"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                         dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode)

        nn.init.kaiming_normal_(self.weight)
        self.gain = nn.Parameter(torch.ones(
            self.weight.size(0), requires_grad=True))
        
    def standardize_weights(self, eps):
        mean= torch.mean(self.weight, dim=(1, 2, 3), keepdim=True)
        var= torch.std(self.weight, dim=(1,2), keepdim=True, unbiased=False) ** 2
        fan_in=torch.prod(torch.tensor(self.weight.shape))

        scale=torch.rsqrt(torch.max(
            var * fan_in, torch.tensor(eps).to(var.device))) * self.gain.view_as(var).to(var.device)
        shift=mean * scale 
        return self.weight * scale - shift
    
    def forward(self, input, eps=1e-4):
        weight= self.standardize_weights(eps)
        return F.conv1d(input, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    

class WSConvTranspose2d(nn.ConvTranspose2d):
    """Applies a 2D transposed convolution operator over an input image
    composed of several input planes after weight normalization/standardization."""
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size,
                 stride=1,
                 padding=0,
                 output_padding=0,
                 groups: int = 1,
                 bias: bool = True,
                 dilation: int = 1,
                 padding_mode: str = 'zeros'):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                         output_padding=output_padding, groups=groups, bias=bias, dilation=dilation, padding_mode=padding_mode)

        nn.init.kaiming_normal_(self.weight)
        self.gain = nn.Parameter(torch.ones(
            self.weight.size(0), requires_grad=True))
        
    def standardize_weights(self, eps):
        mean = torch.mean(self.weight, dims=(1,2,3), keepdim=True)
        var= torch.std(self.weight, dim=(1,2,3), keepdim=True) ** 2
        fan_in= torch.prod(torch.tensor(self.weight.shape[1:]))

        scale = torch.rsqrt(torch.max(
            var * fan_in, torch.tensor(eps).to(var.device))) * self.gain.view_as(var).to(var.device)
        shift = mean * scale
        return self.weight * scale - shift

    def forward(self, input: Tensor, output_size: Optional[List[int]] = None, eps: float = 1e-4) -> Tensor:
        weight = self.standardize_weight(eps)
        return F.conv_transpose2d(input, weight, self.bias, self.stride, self.padding, self.output_padding, self.groups, self.dilation)
    
    
class ScaledStdConv2d(nn.Conv2d):
    """Conv2d layer with scaled weight standardization"""
    def __init__(self, in_channels, out_channels, kernel_size, stride= 1, padding = 0, dilation = 1, groups = 1, bias= True, gain=True,gamma=1.0,eps=1e-5,use_layernorm=False) -> None:
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.gain = nn.Parameter(torch.ones(
            self.out_channels,1,1,1
        )) if gain else None
        self.scale = gamma * self.weight[0].numel() ** -0.5
        self.eps= eps ** 2 if use_layernorm else eps
        self.use_layernorm = use_layernorm

    def get_weight(self):
        if self.use_layernorm:
            weight = self.scale * \
                F.layer_norm(self.weight, self.weight.shape[1:], eps=self.eps)
        else:
            mean = torch.mean(
                self.weight, dim=[1, 2, 3], keepdim=True)
            std = torch.std(
                self.weight, dim=[1, 2, 3], keepdim=True, unbiased=False)
            weight = self.scale * (self.weight - mean) / (std + self.eps)
        if self.gain is not None:
            weight = weight * self.gain
        return weight

    def forward(self, x):
        return F.conv2d(x, self.get_weight(), self.bias, self.stride, self.padding, self.dilation, self.groups)