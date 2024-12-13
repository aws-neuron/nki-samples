# Copyright (c) 2023, Amazon.com. All Rights Reserved

"""
Package containing public kernels for Neuron Kernel Interface (NKI).

Kernels here are the same to the ones available in the 
NKI Github Sample Repo.

https://github.com/aws-neuron/nki-samples
"""
from neuronxcc.nki.kernels.attention import fused_self_attn_for_SD_small_head_size, flash_attn_bwd, flash_fwd
from neuronxcc.nki.kernels.vision import resize_nearest_fixed_dma_kernel, select_and_scatter_kernel
from neuronxcc.nki.kernels.tutorial import add_kernel_nx8x128x512
from neuronxcc.nki.kernels.allocated_attention import allocated_fused_self_attn_for_SD_small_head_size
from neuronxcc.nki.kernels.allocated_fused_linear import allocated_fused_rms_norm_qkv

from neuronxcc.nki._private_kernels.legacy.attention import \
  (fused_self_attn_for_SD_small_head_size as _fused_self_attn_for_SD_small_head_size,
   flash_attn_bwd as _flash_attn_bwd, flash_fwd as _flash_fwd)
from neuronxcc.nki._private_kernels.legacy.vision import (
  resize_nearest_fixed_dma_kernel as _resize_nearest_fixed_dma_kernel,
  select_and_scatter_kernel as _select_and_scatter_kernel)
from neuronxcc.nki._private_kernels.legacy.tutorial import add_kernel_nx8x128x512 as _add_kernel_nx8x128x512
from neuronxcc.nki._private_kernels.legacy.allocated_fused_linear import _allocated_fused_rms_norm_qkv

fused_self_attn_for_SD_small_head_size._legacy_func = _fused_self_attn_for_SD_small_head_size
flash_attn_bwd._legacy_func = _flash_attn_bwd
flash_fwd._legacy_func = _flash_fwd
resize_nearest_fixed_dma_kernel._legacy_func = _resize_nearest_fixed_dma_kernel
select_and_scatter_kernel._legacy_func = _select_and_scatter_kernel
add_kernel_nx8x128x512._legacy_func = _add_kernel_nx8x128x512
allocated_fused_rms_norm_qkv._legacy_func = _allocated_fused_rms_norm_qkv
