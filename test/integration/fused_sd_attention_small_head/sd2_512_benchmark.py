import os
os.environ["NEURON_FUSE_SOFTMAX"] = "1"

import torch
import torch.nn as nn
import torch_neuronx

from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.models.unet_2d_condition import UNet2DConditionOutput

parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir)
from perf_utils.LatencyCollector import benchmark

import sys
if len(sys.argv) != 2:
    print("Usage: python sd2_512_benchmark.py <metric_path>")
    exit(1)
metric_path = os.path.abspath(sys.argv[1])

# Define datatype
DTYPE = torch.bfloat16

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

    def forward(self, sample, timestep, encoder_hidden_states, cross_attention_kwargs=None, return_dict=False):
        sample = self.unetwrap(sample, timestep.to(dtype=DTYPE).expand((sample.shape[0],)), encoder_hidden_states)[0]
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
    
def decode_latents(self, latents):
    latents = latents.to(torch.float)
    latents = 1 / self.vae.config.scaling_factor * latents
    image = self.vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).float().numpy()
    return image

StableDiffusionPipeline.decode_latents = decode_latents

# --- Load all compiled models and benchmark pipeline ---
COMPILER_WORKDIR_ROOT = 'sd2_compile_dir_512'
model_id = "stabilityai/stable-diffusion-2-1-base"
text_encoder_filename = os.path.join(COMPILER_WORKDIR_ROOT, 'text_encoder/model.pt')
decoder_filename = os.path.join(COMPILER_WORKDIR_ROOT, 'vae_decoder/model.pt')
unet_filename = os.path.join(COMPILER_WORKDIR_ROOT, 'unet/model.pt')
post_quant_conv_filename = os.path.join(COMPILER_WORKDIR_ROOT, 'vae_post_quant_conv/model.pt')

pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=DTYPE)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

# Load the compiled UNet onto two neuron cores.
pipe.unet = NeuronUNet(UNetWrap(pipe.unet))
#TODO: FIXME, dual core execution is not working on release pipeline at the moment.
device_ids = [0]
pipe.unet.unetwrap = torch_neuronx.DataParallel(torch.jit.load(unet_filename), device_ids, set_dynamic_batching=True)

class NeuronTypeConversionWrapper(nn.Module):
    def __init__(self, network):
        super().__init__()
        self.network = network

    def forward(self, x):
        return self.network(x.float())

# Load other compiled models onto a single neuron core.
pipe.text_encoder = NeuronTextEncoder(pipe.text_encoder)
pipe.text_encoder.neuron_text_encoder = torch.jit.load(text_encoder_filename)
pipe.vae.decoder = NeuronTypeConversionWrapper(torch.jit.load(decoder_filename))
pipe.vae.post_quant_conv = NeuronTypeConversionWrapper(torch.jit.load(post_quant_conv_filename))

prompt = "a photo of an astronaut riding a horse on mars"
n_runs = 10
bench_result = benchmark(n_runs, "stable_diffusion_512", pipe, prompt, metric_path)
