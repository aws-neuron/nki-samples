import torch 
from torch import nn
from torch import autograd
from neuronxcc.nki.kernels.vision import select_and_scatter_kernel

import torch_xla.core.xla_model as xm
from torch_neuronx.xla_impl.ops import nki_jit

select_and_scatter_func = nki_jit()(select_and_scatter_kernel)

class ModuleWrapper(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x): 
        return self.func(x)


class NeuronMaxPool2d(autograd.function.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return nn.MaxPool2d(kernel_size=3, stride=2, padding=(1, 1))(input)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        output = torch.zeros_like(input)
        select_and_scatter_func(input, grad_output, output)
        return output


def replace_maxpool(model):
    model.maxpool = ModuleWrapper(NeuronMaxPool2d.apply)
    return model