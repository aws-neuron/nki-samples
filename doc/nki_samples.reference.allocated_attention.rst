Allocated Attention
=======================

.. currentmodule:: nki_samples.reference.allocated_attention

This file hosts the high-performance reference implementation for
the attention blocks that are used
in `Stable Diffusion <https://huggingface.co/spaces/stabilityai/stable-diffusion>`_ models.
This implementation uses 
the `direct allocation API <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/nki_direct_allocation_guide.html>` to achieve better performance.

.. autosummary::
    :toctree: generated
    
    allocated_fused_self_attn_for_SD_small_head_size
    