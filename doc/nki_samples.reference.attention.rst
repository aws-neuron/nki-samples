Attention
=======================

.. currentmodule:: nki_samples.reference.attention

This file hosts the high-performance reference implementation for
`FlashAttention <https://arxiv.org/abs/2205.14135>`_ (forward & backward), and attention blocks that are used
in `Stable Diffusion <https://huggingface.co/spaces/stabilityai/stable-diffusion>`_ models.

.. autosummary::
    :toctree: generated
    
    flash_fwd
    flash_attn_bwd
    fused_self_attn_for_SD_small_head_size
    