import os

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

parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir)
from perf_utils.LatencyCollector import benchmark

import sys
if len(sys.argv) != 2:
    print("Usage: python sd2_inpainting_936_624_benchmark.py <metric_path>")
    exit(1)
metric_path = os.path.abspath(sys.argv[1])

from wrapper import UNetWrap, NeuronUNet, NeuronTextEncoder

# Define datatype for UNet
DTYPE = torch.bfloat16

clear_output(wait=False)

"""
Sample image is taken from: https://huggingface.co/stabilityai/stable-diffusion-2-inpainting
"""

# --- Load all compiled models and run pipeline ---
COMPILER_WORKDIR_ROOT = "sd2_inpainting_neuron"
model_id = "stabilityai/stable-diffusion-2-inpainting"

pipe = StableDiffusionInpaintPipeline.from_pretrained(model_id, torch_dtype=torch.float32)

text_encoder_filename = os.path.join(COMPILER_WORKDIR_ROOT, 'text_encoder/model.pt')
unet_filename = os.path.join(COMPILER_WORKDIR_ROOT, 'unet/model.pt')
vae_encoder_filename = os.path.join(COMPILER_WORKDIR_ROOT, 'vae_encoder/model.pt')
decoder_filename = os.path.join(COMPILER_WORKDIR_ROOT, 'vae_decoder/model.pt')
post_quant_conv_filename = os.path.join(COMPILER_WORKDIR_ROOT, 'vae_post_quant_conv/model.pt')

# Load the compiled UNet onto two neuron cores.
pipe.unet = NeuronUNet(UNetWrap(pipe.unet))
device_ids = [0,1]
pipe.unet.unetwrap = torch_neuronx.DataParallel(torch.jit.load(unet_filename), device_ids, set_dynamic_batching=False)

# Load other compiled models onto a single neuron core.
pipe.text_encoder = NeuronTextEncoder(pipe.text_encoder)
pipe.text_encoder.neuron_text_encoder = torch.jit.load(text_encoder_filename)
pipe.vae.encoder = torch.jit.load(vae_encoder_filename)
pipe.vae.decoder = torch.jit.load(decoder_filename)
pipe.vae.post_quant_conv = torch.jit.load(post_quant_conv_filename)

height, width = 624, 936

import PIL
base_image = PIL.Image.open('inpainting_photo.png')
mask = PIL.Image.open('inpainting_mask.png')

prompt = 'Face of a yellow cat, high resolution, sitting on a park bench'
n_runs = 10

bench_result = benchmark(n_runs, "stable_diffusion_2_inpainting_936_624", pipe, {'prompt':prompt, 'image':base_image, 'mask_image':mask, 'height':height, 'width':width}, metric_path)