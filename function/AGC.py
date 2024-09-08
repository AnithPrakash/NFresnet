import torch 
from torch import nn, optim 

from function.utils import unitwise_norm
from collections.abc import Iterable

#creating Adaptive gradient clipping 

class AGC(optim.Optimizer):
    """Generic implmentation of the Adaptive Gradient Clipping"""

    def __init__(self, params, optim: optim.Optimizer, clipping: float = 1e-2, eps: float = 1e-3, model=None, ignore_agc=["fc"]):
        if clipping < 0.0:
            raise ValueError("Invalid clipping value: {}".format(clipping))
        if eps < 0.0:
            raise ValueError("Invalid eps value: {}".format(eps))
        
        self.optim=optim

        defaults= dict(clipping=clipping, eps=eps)
        defaults ={**defaults, **optim.defaults}

        if not isinstance(ignore_agc, Iterable):
            ignore_agc = [ignore_agc]

        if model is not None:
            assert ignore_agc not in [
                None, []], "You must specify ignore_agc for AGC to ignore fc-like(or other) layers"
            names = [name for name, module in model.named_modules()]

            for module_name in ignore_agc:
                if module_name not in names:
                    raise ModuleNotFoundError(
                        "Module name {} not found in the model".format(module_name))
            params = [{"params": list(module.parameters())} for name,
                          module in model.named_modules() if name not in ignore_agc]
        
        else:
            params = [{"params": params}]

        self.agc_params = params
        self.eps = eps
        self.clipping = clipping
        
        self.param_groups = optim.param_groups
        self.state = optim.state

        #super(AGC, self).__init__([], defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.agc_params:
            for p in group['params']:
                if p.grad is None:
                    continue

                param_norm = torch.max(unitwise_norm(
                    p.detach()), torch.tensor(self.eps).to(p.device))
                grad_norm = unitwise_norm(p.grad.detach())
                max_norm = param_norm * self.clipping

                trigger = grad_norm > max_norm

                clipped_grad = p.grad * \
                    (max_norm / torch.max(grad_norm,
                                          torch.tensor(1e-6).to(grad_norm.device)))
                p.grad.detach().data.copy_(torch.where(trigger, clipped_grad, p.grad))

        return self.optim.step(closure)

    def zero_grad(self, set_to_none: bool = False):
        r"""Sets the gradients of all optimized :class:`torch.Tensor` s to zero..
        """
        for group in self.agc_params:
            for p in group['params']:
                if p.grad is not None:
                    if set_to_none:
                        p.grad = None
                    else:
                        if p.grad.grad_fn is not None:
                            p.grad.detach_()
                        else:
                            p.grad.requires_grad_(False)
                        p.grad.zero_()