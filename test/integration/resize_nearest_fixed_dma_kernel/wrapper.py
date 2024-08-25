import torch
import torch.nn as nn
from diffusers.models.unet_2d_condition import UNet2DConditionOutput

"""
Define a double-wrapper for the UNet because of the special UNet2DConditionOutput output type and a wrapper for the text encoder. 
These wrappers enable torch_neuronx.trace to trace the wrapped models for compilation with the Neuron compiler

Details
https://github.com/aws-neuron/aws-neuron-samples/blob/master/torch-neuronx/inference/hf_pretrained_sd2_inpainting_936_624_inference.ipynb
"""

class UNetWrap(nn.Module):
    def __init__(self, unet):
        super().__init__()
        self.unet = unet

    def forward(self, sample, timestep, encoder_hidden_states, cross_attention_kwargs=None):
        out_tuple = self.unet(sample, timestep, encoder_hidden_states, return_dict=False)
        return out_tuple

class NeuronUNet(nn.Module):
    def __init__(self, unetwrap):
        super().__init__()
        self.unetwrap = unetwrap
        self.config = unetwrap.unet.config
        self.in_channels = unetwrap.unet.in_channels
        self.device = unetwrap.unet.device

    def forward(self, sample, timestep, encoder_hidden_states, timestep_cond=None, added_cond_kwargs=None, cross_attention_kwargs=None, return_dict=False):
        sample = self.unetwrap(sample.bfloat16(), timestep.bfloat16().expand((sample.shape[0],)), encoder_hidden_states.bfloat16())[0]
        return UNet2DConditionOutput(sample=sample)

class NeuronTextEncoder(nn.Module):
    def __init__(self, text_encoder):
        super().__init__()
        self.neuron_text_encoder = text_encoder
        self.config = text_encoder.config
        self.dtype = text_encoder.dtype
        self.device = text_encoder.device

    def forward(self, emb, attention_mask = None):
        return [self.neuron_text_encoder(emb)['last_hidden_state']]