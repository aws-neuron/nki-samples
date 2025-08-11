"""
Copyright (c) 2025, Amazon.com. All Rights Reserved

Kernels written by the AWS Neuron Team.

WARNING: These kernels:
   - Are tested only against internal nightly compiler builds
   - May rely on internal compiler feature/flags and not be compatible with public NeuronSDK
   - Have not been extensively tested across all input configurations
   - Carry no compatibility guarantees
   - The behavior of these kernels may be modified without prior notice

Flash Paged Attention kernels with variable-length sequence inputs.

"""

import neuronxcc.nki.language as nl

B_P_SIZE = nl.tile_size.pmax
B_FMAX_SIZE = nl.tile_size.gemm_moving_fmax

# Magic number to replace -inf similar to what Tensorizer uses
NEG_INF = -9984.0
