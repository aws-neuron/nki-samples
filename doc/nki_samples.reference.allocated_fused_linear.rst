Allocated Fused Linear
=======================

.. currentmodule:: nki_samples.reference.allocated_fused_linear

This file hosts the high-performance kernel that computes `RMSNorm(hidden) @ wQKV`.
This implementation uses 
the `direct allocation API <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/nki_direct_allocation_guide.html>` to achieve better performance.

.. autosummary::
    :toctree: generated
    
    allocated_fused_rms_norm_qkv
    