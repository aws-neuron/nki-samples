import os
from typing import Optional

import torch
import torch.nn as nn
import torch_neuronx
import numpy as np

from matplotlib import pyplot as plt
from matplotlib import image as mpimg
import time
import copy
from IPython.display import clear_output

from diffusers import StableDiffusionInpaintPipeline
from diffusers.models.unet_2d_condition import UNet2DConditionOutput
# Compatibility for diffusers<0.18.0
from packaging import version
import diffusers
diffusers_version = version.parse(diffusers.__version__)
use_new_diffusers = diffusers_version >= version.parse('0.18.0')
if use_new_diffusers:
    from diffusers.models.attention_processor import Attention
else:
    from diffusers.models.cross_attention import CrossAttention

from wrapper import UNetWrap, NeuronUNet, NeuronTextEncoder

# Define datatype for UNet
DTYPE = torch.bfloat16

clear_output(wait=False)

"""
In the following section, we will compile parts of the Stable Diffusion pipeline for execution on Neuron. 
Note that this only needs to be done once: After you have compiled and saved the model by running the following section of code, 
you can reuse it any number of times without having to recompile. In particular, we will compile:

The CLIP text encoder;
The VAE encoder;
The VAE decoder;
The UNet, and
The VAE_post_quant_conv These blocks are chosen because they represent the bulk of the compute in the pipeline, 
and performance benchmarking has shown that running them on Neuron yields significant performance benefit.

The UNet contains Upsample2D layers.
In this example, we will replace Upsample2D layer with our Custom_Upsample2D layer which is using NKI kernel (resize_nearest_fixed_dma_kernel). (Line 179)
In this custom layer, when it finds the upscaling factor is not an integer, it applies the kernel. (Line 142)
Example shape : (1, 1280, 30, 20) -> (1, 1280, 59, 38)
Please note that this is a NKI example, showing how we apply the kernel. The non-integer Upsample2D is already supported in our compiler.

Additional Notes
- We use the optimized get_attention_scores utility function, 
 to replace the original get_attention_scores function in the attention_processor.Attention class
- In order to save RAM before tracing each model, 
 we make a deepcopy of the part of the pipeline and delete the pipeline object from the memory,  
 This trick allows the compile to succeed on instance types with a smaller amount of RAM.


Code details :
https://github.com/aws-neuron/aws-neuron-samples/blob/master/torch-neuronx/inference/hf_pretrained_sd2_inpainting_936_624_inference.ipynb

"""


from neuronxcc.nki.kernels.vision import resize_nearest_fixed_dma_kernel
from torch_neuronx import nki_jit
from torch_xla.core import xla_model as xm
from torch import Tensor
import torch.nn.functional as F
from diffusers.utils import USE_PEFT_BACKEND
from diffusers.models.lora import LoRACompatibleConv
from diffusers.models.resnet import Upsample2D

class Custom_Upsample2D(nn.Module):
    """A 2D upsampling layer with an optional convolution.
    Copied from diffusers.models.resnet.Upsample2D
    """

    def __init__(
        self,
        channels: int,
        conv,
        conv2d_0,
        use_conv: bool = False,
        use_conv_transpose: bool = False,
        out_channels: Optional[int] = None,
        name: str = "conv",
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_conv_transpose = use_conv_transpose
        self.name = name
        self.conv = conv
        self.Conv2d_0 = conv2d_0

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        output_size: Optional[int] = None,
        scale: float = 1.0,
    ) -> torch.FloatTensor:
        assert hidden_states.shape[1] == self.channels

        if self.use_conv_transpose:
            return self.conv(hidden_states)

        # Cast to float32 to as 'upsample_nearest2d_out_frame' op does not support bfloat16
        dtype = hidden_states.dtype
        if dtype == torch.bfloat16:
            hidden_states = hidden_states.to(torch.float32)

        if hidden_states.shape[0] >= 64:
            hidden_states = hidden_states.contiguous()

        if output_size is None:
            hidden_states = F.interpolate(hidden_states, scale_factor=2.0, mode="nearest")
        else:
            # # When the upscailng factor is not an integer, compile using resize_nearest_fixed_dma_kernel
            # # Otherwise, use the default interpolate function.
            if (output_size[0] % hidden_states.shape[2] != 0 or output_size[1] % hidden_states.shape[3] != 0):
                from neuronxcc.nki.kernels.vision import resize_nearest_fixed_dma_kernel
                from torch_neuronx import nki_jit
                from torch_xla.core import xla_model as xm
                # Compile NKI kernel
                device = xm.xla_device()
                hidden_states = hidden_states.to(device=device)
                nki_func = nki_jit(resize_nearest_fixed_dma_kernel)
                # Apply NKI kernel
                hidden_states = torch.permute(hidden_states, (0, 2, 3, 1))
                result = torch.empty((hidden_states.shape[0], output_size[0], output_size[1], hidden_states.shape[3]) , dtype=torch.float32, device=device)
                nki_func[hidden_states.shape[0]](hidden_states, result)
                hidden_states = torch.permute(result, (0, 3, 1, 2))
            else :
                hidden_states = F.interpolate(hidden_states, size=output_size, mode="nearest")
        
        # If the input is bfloat16, we cast back to bfloat16
        if dtype == torch.bfloat16:
            hidden_states = hidden_states.to(torch.bfloat16)

        if self.use_conv:
            if self.name == "conv":
                if isinstance(self.conv, LoRACompatibleConv) and not USE_PEFT_BACKEND:
                    hidden_states = self.conv(hidden_states, scale)
                else:
                    hidden_states = self.conv(hidden_states)
            else:
                if isinstance(self.Conv2d_0, LoRACompatibleConv) and not USE_PEFT_BACKEND:
                    hidden_states = self.Conv2d_0(hidden_states, scale)
                else:
                    hidden_states = self.Conv2d_0(hidden_states)

        return hidden_states

# Replace diffusers.models.resnet.Upsample2D with Custom_Upsample2D
def replace_upsampling(module):
    for name, child in module.named_children():
        if isinstance(child, Upsample2D):
            if child.name == "conv":
                setattr(module, name, Custom_Upsample2D(channels=child.channels, conv=child.conv, conv2d_0=None, use_conv=child.use_conv, use_conv_transpose=child.use_conv_transpose, out_channels=child.out_channels, name=child.name))
            else:
                setattr(module, name, Custom_Upsample2D(channels=child.channels, conv=None, conv2d_0=child.Conv2d_0, use_conv=child.use_conv, use_conv_transpose=child.use_conv_transpose, out_channels=child.out_channels, name=child.name))
            
        else:
            replace_upsampling(child)


# optimized attention scores
def get_attention_scores(self, query, key, attn_mask):       
    dtype = query.dtype

    if self.upcast_attention:
        query = query.float()
        key = key.float()

    # Check for square matmuls
    if(query.size() == key.size()):
        attention_scores = custom_badbmm(
            key,
            query.transpose(-1, -2)
        )

        if self.upcast_softmax:
            attention_scores = attention_scores.float()

        attention_probs = torch.nn.functional.softmax(attention_scores, dim=1).permute(0,2,1)
        attention_probs = attention_probs.to(dtype)

    else:
        attention_scores = custom_badbmm(
            query,
            key.transpose(-1, -2)
        )

        if self.upcast_softmax:
            attention_scores = attention_scores.float()

        attention_probs = torch.nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = attention_probs.to(dtype)
        
    return attention_probs

def custom_badbmm(a, b):
    bmm = torch.bmm(a, b)
    scaled = bmm * 0.125
    return scaled


COMPILER_WORKDIR_ROOT = "sd2_inpainting_neuron"

def trace_text_encoder(model_id):
    pipe = StableDiffusionInpaintPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
    text_encoder = copy.deepcopy(pipe.text_encoder)
    del pipe
    # Apply the wrapper to deal with custom return type
    text_encoder = NeuronTextEncoder(text_encoder)

    # Compile text encoder
    # This is used for indexing a lookup table in torch.nn.Embedding,
    # so using random numbers may give errors (out of range).
    emb = torch.tensor([[49406, 18376,   525,  7496, 49407,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0]])
    text_encoder_neuron = torch_neuronx.trace(
            text_encoder.neuron_text_encoder, 
            emb, 
            compiler_workdir=os.path.join(COMPILER_WORKDIR_ROOT, 'text_encoder'),
            )
    
    # Enable asynchronous loading to speed up model load
    torch_neuronx.async_load(text_encoder_neuron)   
    
    # Save the compiled text encoder
    text_encoder_filename = os.path.join(COMPILER_WORKDIR_ROOT, 'text_encoder/model.pt')
    torch.jit.save(text_encoder_neuron, text_encoder_filename)

    # delete unused objects
    del text_encoder
    del text_encoder_neuron

def trace_vae_encoder(model_id, height, width):
    # Only keep the model being compiled in RAM to minimze memory pressure
    pipe = StableDiffusionInpaintPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
    vae_encoder = copy.deepcopy(pipe.vae.encoder)
    del pipe

    sample_input = torch.randn([1, 3, height, width])
    vae_encoder_neuron = torch_neuronx.trace(
            vae_encoder, 
            sample_input, 
            compiler_workdir=os.path.join(COMPILER_WORKDIR_ROOT, 'vae_encoder'),
            )
    
    # Enable asynchronous loading to speed up model load
    torch_neuronx.async_load(vae_encoder_neuron)    

    # Save the compiled text encoder
    vae_encoder_filename = os.path.join(COMPILER_WORKDIR_ROOT, 'vae_encoder/model.pt')
    torch.jit.save(vae_encoder_neuron, vae_encoder_filename)

    # delete unused objects
    del vae_encoder
    del vae_encoder_neuron


def trace_vae_decoder(model_id, height, width):
    # Only keep the model being compiled in RAM to minimze memory pressure
    pipe = StableDiffusionInpaintPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
    decoder = copy.deepcopy(pipe.vae.decoder)
    del pipe

    # Compile vae decoder
    decoder_in = torch.randn([1, 4, height // 8, width // 8])
    decoder_neuron = torch_neuronx.trace(
        decoder, 
        decoder_in, 
        compiler_workdir=os.path.join(COMPILER_WORKDIR_ROOT, 'vae_decoder'),
        compiler_args=["--verbose", "info"]
    )

    # Enable asynchronous loading to speed up model load
    torch_neuronx.async_load(decoder_neuron)

    # Save the compiled vae decoder
    decoder_filename = os.path.join(COMPILER_WORKDIR_ROOT, 'vae_decoder/model.pt')
    torch.jit.save(decoder_neuron, decoder_filename)

    # delete unused objects
    del decoder
    del decoder_neuron

def trace_unet(model_id, height, width):
    # --- Compile UNet and save ---
    DTYPE = torch.bfloat16
    pipe = StableDiffusionInpaintPipeline.from_pretrained(model_id, torch_dtype=DTYPE)

    # Replace original cross-attention module with custom cross-attention module for better performance
    Attention.get_attention_scores = get_attention_scores

    # Apply double wrapper to deal with custom return type
    pipe.unet = NeuronUNet(UNetWrap(pipe.unet))

    # Only keep the model being compiled in RAM to minimze memory pressure
    unet = copy.deepcopy(pipe.unet.unetwrap)

    # replace Upsample2D
    replace_upsampling(unet)

    del pipe

    sample_1b = torch.randn([1, 9, height // 8, width // 8], dtype=DTYPE)
    timestep_1b = torch.tensor(999, dtype=DTYPE).expand((1,))
    encoder_hidden_states_1b = torch.randn([1, 77, 1024], dtype=DTYPE)
    example_inputs = sample_1b, timestep_1b, encoder_hidden_states_1b

    unet_neuron = torch_neuronx.trace(
        unet,
        example_inputs,
        compiler_workdir=os.path.join(COMPILER_WORKDIR_ROOT, 'unet'),
        compiler_args=["--model-type=unet-inference", "--verbose=info"],
    )

    # Enable asynchronous and lazy loading to speed up model load
    torch_neuronx.async_load(unet_neuron)
    torch_neuronx.lazy_load(unet_neuron)

    # save compiled unet
    unet_filename = os.path.join(COMPILER_WORKDIR_ROOT, 'unet/model.pt')
    torch.jit.save(unet_neuron, unet_filename)

    # delete unused objects
    del unet
    del unet_neuron
    
def trace_post_quant_conv(model_id, height, width):
    # Only keep the model being compiled in RAM to minimze memory pressure
    pipe = StableDiffusionInpaintPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
    post_quant_conv = copy.deepcopy(pipe.vae.post_quant_conv)
    del pipe

    # Compile vae post_quant_conv
    post_quant_conv_in = torch.randn([1, 4, height // 8 , width // 8])
    post_quant_conv_neuron = torch_neuronx.trace(
        post_quant_conv, 
        post_quant_conv_in,
        compiler_workdir=os.path.join(COMPILER_WORKDIR_ROOT, 'vae_post_quant_conv'),
        compiler_args=["--verbose", "info"]
    )

    # Enable asynchronous loading to speed up model load
    torch_neuronx.async_load(post_quant_conv_neuron)

    # Save the compiled vae post_quant_conv
    post_quant_conv_filename = os.path.join(COMPILER_WORKDIR_ROOT, 'vae_post_quant_conv/model.pt')
    torch.jit.save(post_quant_conv_neuron, post_quant_conv_filename)

    # delete unused objects
    del post_quant_conv
    del post_quant_conv_neuron

model_id = "stabilityai/stable-diffusion-2-inpainting"
height, width = 624, 936

# trace the parts of the pipeline
trace_text_encoder(model_id)
trace_vae_decoder(model_id, height, width)
trace_vae_encoder(model_id, height, width)
trace_unet(model_id, height, width)
trace_post_quant_conv(model_id, height, width)
