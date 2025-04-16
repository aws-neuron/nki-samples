import torch
import torch.nn as nn
import torch.nn.functional as F

from softmax_nki_kernels import nki_softmax_kernel

class NaiveSoftmax(nn.Module):
    def __init__(self):
        super(NaiveSoftmax, self).__init__()

    def forward(self, x):

        numerator = torch.exp(x)
        denominator = torch.sum(numerator, dim=-1, keepdim=True)
        sm = numerator / denominator 
        return sm

def naive_softmax(logits: torch.tensor) -> torch.tensor :
    softmax = NaiveSoftmax()
    probs = softmax(logits)
    return probs

from torch_xla.core import xla_model as xm
device = xm.xla_device()

logits = torch.tensor([[1.0,2.0,3.0,4.0,5.0], [5.0,4.0,3.0,2.0,1.0]]).to(device)

sm_naive = naive_softmax(logits)
sm_nki = nki_softmax_kernel(logits)

assert torch.allclose(sm_naive, sm_nki, rtol=1e-5, atol=1e-5)