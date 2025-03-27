"""
Copyright (c) 2025, Amazon.com. All Rights Reserved

Kernels written by the AWS Neuron.

WARNING: These kernels:
   - Are tested only against internal nightly builds
   - May not be compatible with public NeuronSDK releases
   - Have not been extensively tested across all input configurations
   - Carry no compatibility guarantees
   - The behavior of these kernels may be modified without prior notice
"""

from neuronxcc import nki
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
from math import ceil

@nki.jit(mode='trace')
def rmsnorm_group(
    sequence_tile,
    squared_sequence_group,
    inverse_rms_scale,
    ln_w,
    broadcasted_ln_w,
    eps_bias,
    *, 
    group_s_idx, 
    group_s_sz
):
  """Performs RMSNorm on a group of sequences within a tile where 
      RMS(x) = sqrt(mean((x + eps)^2) + eps) by taking the mean over the hidden dimension (H).
      RMSNorm(x) = x / RMS(x).

  Args:
      sequence_tile: Tile of shape [NUM_GROUP_S, GROUP_S_SZ, H]. This tile is made up of NUM_GROUP_S many groups
        of shape [GROUP_S_SZ, H] except possibly the last group which has shape [last_group_s_sz, H].  
      squared_sequence_group: Buffer of shape [GROUP_S_SZ, H] used to store squared values during RMS computation.
      inverse_rms_scale: Tensor of shape [NUM_GROUP_S, GROUP_S_SZ, 1] that stores 1 / RMS(x) values.
      ln_w: The gamma parameter tensor of shape [1, H].  
      broadcasted_ln_w: The broadcasted values for the group being processed of shape [GROUP_S_SZ, GROUP_H_SZ].
      eps_bias: Tensor of shape [GROUP_S_SZ, 1] containing eps, a small positive float, for numerical stability.
      group_s_idx: A group index to index the NUM_GROUP_S groups. Must be in the range [0, NUM_GROUP_S).
      group_s_sz: Size of S dimension of the group to be processed (i.e.,  sequence_tile[group_s]).
  """
  NUM_GROUP_S, GROUP_S_SZ, H = sequence_tile.shape

  # Index and size sanity check.
  assert 0 <= group_s_sz <= GROUP_S_SZ, f"size of the group being processed must be between 0 and {GROUP_S_SZ} (inclusive)"
  assert group_s_idx == NUM_GROUP_S - 1 or group_s_sz == GROUP_S_SZ, \
    f"if group_s_idx {group_s_idx} is not NUM_GROUP_S - 1 ({NUM_GROUP_S - 1}), group_s_sz ({group_s_sz}) must be GROUP_S_SZ ({GROUP_S_SZ})"

  # Compute 1 / RMS(group).
  squared_sequence_group[:group_s_sz, :] = nisa.activation_reduce(op=nl.square, data=sequence_tile[group_s_idx, :group_s_sz, :],
                                                                  reduce_op=nl.add, reduce_res=inverse_rms_scale[group_s_idx, :group_s_sz, :],
                                                                  bias=eps_bias[:group_s_sz], scale=1.0)
  inverse_rms_scale[group_s_idx, :group_s_sz, :] = nisa.activation(op=nl.rsqrt, data=inverse_rms_scale[group_s_idx, :group_s_sz, :],
                                                                   bias=eps_bias[:group_s_sz], scale=1 / H)
  
  GROUP_H_SZ = nl.tile_size.gemm_moving_fmax  # 512
  NUM_GROUP_H = ceil(H / GROUP_H_SZ)

  zero_bias = nl.zeros((group_s_sz, 1), dtype=nl.float32, buffer=nl.sbuf)

  for group_h in range(NUM_GROUP_H):
    if group_h == NUM_GROUP_H - 1:
      group_h_sz = H % GROUP_H_SZ if H % GROUP_H_SZ != 0 else GROUP_H_SZ
      group_h_base = group_h * GROUP_H_SZ

      # Apply the inverse RMS scale such that sequence_tile stores x / RMS(x) for a [group_s_sz, group_h_sz] group.
      sequence_tile[group_s_idx, :group_s_sz, nl.ds(group_h_base, group_h_sz)] = nisa.activation(
        op=nl.copy, data=sequence_tile[group_s_idx, :group_s_sz, nl.ds(group_h_base, group_h_sz)],
        bias=zero_bias, scale=inverse_rms_scale[group_s_idx, :group_s_sz, :])
      
      # Load [1, group_h_sz] of ln_w for the group H into the first row of broadcasted_ln_w.
      broadcasted_ln_w[:1, :group_h_sz] = nl.load(ln_w[:1, nl.ds(group_h_base, group_h_sz)])

      # Initialize a ones vector of shape [1, 128].
      ones = nl.ones((1, 128), dtype=nl.bfloat16, buffer=nl.sbuf)

      # Matrix multiply ones.T @ broadcasted_ln_w[0, :group_h_sz] to broadcast the ln_w gamma scale
      # to shape [128, group_h_sz].
      broadcasted_ln_w_psum = nisa.nc_matmul(ones, broadcasted_ln_w[:1, :group_h_sz])

      # Copy broadcasted gamma scale from PSUM to SBUF.
      broadcasted_ln_w[:group_s_sz, :group_h_sz] = nisa.activation(op=nl.copy, data=broadcasted_ln_w_psum[:group_s_sz, :group_h_sz],
                                                                   bias=zero_bias[:group_s_sz], scale=1.0)

      # Apply gamma.
      sequence_tile[group_s_idx, :group_s_sz, nl.ds(group_h_base, group_h_sz)] = nisa.tensor_tensor(
        data1=sequence_tile[group_s_idx, :group_s_sz, nl.ds(group_h_base, group_h_sz)],
        data2=broadcasted_ln_w[:group_s_sz, :group_h_sz],
        op=nl.multiply
      )
    else:
      group_h_sz = GROUP_H_SZ
      group_h_base = group_h * GROUP_H_SZ

      # Apply the inverse RMS scale such that sequence_tile stores x / RMS(x) for a [group_s_sz, group_h_sz] group.
      sequence_tile[group_s_idx, :group_s_sz, nl.ds(group_h_base, group_h_sz)] = nisa.activation(
        op=nl.copy, data=sequence_tile[group_s_idx, :group_s_sz, nl.ds(group_h_base, group_h_sz)],
        bias=zero_bias, scale=inverse_rms_scale[group_s_idx, :group_s_sz, :])
      
      # Load [1, group_h_sz] of ln_w for the group H into the first row of broadcasted_ln_w.
      broadcasted_ln_w[:1, :group_h_sz] = nl.load(ln_w[:1, nl.ds(group_h_base, group_h_sz)])

      # Initialize a ones vector of shape [1, 128].
      ones = nl.ones((1, 128), dtype=nl.bfloat16, buffer=nl.sbuf)

      # Matrix multiply ones.T @ broadcasted_ln_w[0, :group_h_sz] to broadcast the ln_w gamma scale
      # to shape [128, group_h_sz].
      broadcasted_ln_w_psum = nisa.nc_matmul(ones, broadcasted_ln_w[:1, :group_h_sz])

      # Copy broadcasted gamma scale from PSUM to SBUF.
      broadcasted_ln_w[:group_s_sz, :group_h_sz] = nisa.activation(op=nl.copy, data=broadcasted_ln_w_psum[:group_s_sz, :group_h_sz],
                                                                   bias=zero_bias[:group_s_sz], scale=1.0)

      # Apply gamma.
      sequence_tile[group_s_idx, :group_s_sz, nl.ds(group_h_base, group_h_sz)] = nisa.tensor_tensor(
        data1=sequence_tile[group_s_idx, :group_s_sz, nl.ds(group_h_base, group_h_sz)],
        data2=broadcasted_ln_w[:group_s_sz, :group_h_sz],
        op=nl.multiply
      )

@nki.jit(mode='trace')
def quantize_group(
    sequence_tile,
    quantized_sequence_tile,
    tile_dequant_scale, 
    tile_quant_scale, 
    *, 
    group_s_idx, 
    group_s_sz, 
    lower_bound, 
    min_val):
  """NKI kernel to perform fp8 quantization on a group along the hidden dimension.

  Args:
      sequence_tile: Tile of shape [NUM_GROUP_S, GROUP_S_SZ, H]. This tile is made up of NUM_GROUP_S many groups
        of shape [GROUP_S_SZ, H] except possibly the last group which has shape [last_group_s_sz, H].
      quantized_sequence_tile: Tile of shape [NUM_GROUP_S, GROUP_S_SZ, H + 4] of dtype float8_e4m3. This tile
        stores the quantized sequence_tile in quantized_sequence_tile[:, :, :H] and stores the fp32 quantization
        scales as 4 fp8 in quantized_sequence_tile[:, :, H:].
      abs_group: Buffer of shape [GROUP_S_SZ, H] used for computing absolute maximum of a group.
      tile_dequant_scale: Temporary buffer of shape [NUM_GROUP_S, GROUP_S_SZ, 1] used for storing dequantization
        scales.
      tile_quant_scale: Temporary buffer of shape [NUM_GROUP_S, GROUP_S_SZ, 1] used for storing quantization 
        scales.
      group_s_idx: A group index to index the NUM_GROUP_S groups. Must be in the range [0, NUM_GROUP_S).
      group_s_sz: Size of S dimension of the group to be processed (i.e.,  sequence_tile[group_s]).
      lower_bound: Optional non-negative float (default 0) used for clipping input values and scale. By default,
        lower_bound is ignored and clipping occurs only when a lower_bound > 0 is specified as an argument.
      min_val: Optional positive float (default 1e-6) used as the minimum threshold for dequantization scales to 
        ensure numerical stability (since tile_quant_scale is computed as 1 / tile_dequant_scale).
  
  Returns:
      None
  """
  FP8_RANGE = 240

  # sequence_tile and quantized_sequence_tile shape check.
  NUM_GROUP_S, GROUP_S_SZ, H = sequence_tile.shape
  NUM_GROUP_S_, GROUP_S_SZ_, H_ = quantized_sequence_tile.shape
  assert NUM_GROUP_S == NUM_GROUP_S_ and GROUP_S_SZ == GROUP_S_SZ_ and H_ == H + 4, "shapes of quantized and unquantized sequence should match"

  # Quantization and dequantization scale check.
  NUM_GROUP_S_, GROUP_S_SZ_, _ = tile_dequant_scale.shape
  assert NUM_GROUP_S == NUM_GROUP_S_ and GROUP_S_SZ == GROUP_S_SZ_, "first 2 dimensions of group and dequantization scale should match"
  assert tile_dequant_scale.shape == tile_quant_scale.shape, "shapes of dequantization and quantization scale should match"

  # Numerical accuracy constants check.
  assert min_val > 0, f"min_val ({min_val}) must be positive"

  abs_group = nl.ndarray((GROUP_S_SZ, H), dtype=nl.bfloat16, buffer=nl.sbuf)

  # Index and size sanity check.
  assert 0 <= group_s_sz <= GROUP_S_SZ, f"size of the group being processed must be between 0 and {GROUP_S_SZ} (inclusive)"
  assert group_s_idx == NUM_GROUP_S - 1 or group_s_sz == GROUP_S_SZ, \
    f"if group_s_idx {group_s_idx} is not NUM_GROUP_S - 1 ({NUM_GROUP_S - 1}), group_s_sz ({group_s_sz}) must be GROUP_S_SZ ({GROUP_S_SZ})"

  # abs_group stores the absolute value of the group being processed.
  # tile_dequant_scale[group_s, 0:group_s_sz, 0] stores abs_group reduced over abs_group's last dimension (i.e. dimension H) 
  # to get absolute max (absMax).
  abs_group[0:group_s_sz, :] = nisa.tensor_scalar_reduce(data=sequence_tile[group_s_idx, 0:group_s_sz, :], op0=nl.abs, operand0=0.0,
                                                         reduce_op=nl.max,
                                                         reduce_res=tile_dequant_scale[group_s_idx, 0:group_s_sz, 0])
  
  if lower_bound > 0:
    # Clip tile_dequant_scale in range [0, lower_bound].
    tile_dequant_scale[group_s_idx, 0:group_s_sz, 0] = nisa.tensor_scalar(data=tile_dequant_scale[group_s_idx, 0:group_s_sz, 0],
                                                                          op0=nl.minimum, operand0=lower_bound)
    
    # Clip the group being processed in range [-lower_bound, lower_bound].
    sequence_tile[group_s_idx, 0:group_s_sz, :] = nisa.tensor_scalar(data=sequence_tile[group_s_idx, 0:group_s_sz, :],
                                                                     op0=nl.minimum, operand0=lower_bound)
    sequence_tile[group_s_idx, 0:group_s_sz, :] = nisa.tensor_scalar(data=sequence_tile[group_s_idx, 0:group_s_sz, :],
                                                                     op0=nl.maximum, operand0=-lower_bound)
  
  # Compute absMax / 240 to get the dequantization scale.
  zero_bias = nl.zeros((group_s_sz, 1), dtype=nl.float32, buffer=nl.sbuf)
  tile_dequant_scale[group_s_idx, 0:group_s_sz, 0] = nisa.activation(op=nl.copy, data=tile_dequant_scale[group_s_idx, 0:group_s_sz, 0],
                                                                     scale=1 / FP8_RANGE, bias=zero_bias) 
  
  # Take tile_dequant_scale = max(tile_dequant_scale, min_val) for numerical stability reasons.
  tile_dequant_scale[group_s_idx, 0:group_s_sz, 0] = nisa.tensor_scalar(data=tile_dequant_scale[group_s_idx, 0:group_s_sz, 0],
                                                                        op0=nl.maximum, operand0=min_val)
  
  # Compute tile_quant_scale by taking reciprocal of tile_dequant_scale.
  tile_quant_scale[group_s_idx, 0:group_s_sz, 0] = nisa.reciprocal(tile_dequant_scale[group_s_idx, 0:group_s_sz, 0])

  # Apply quantization scale.
  quantized_sequence_tile[group_s_idx, 0:group_s_sz, :H] = nisa.tensor_scalar(data=sequence_tile[group_s_idx, 0:group_s_sz, :],
                                                                              op0=nl.multiply, operand0=tile_quant_scale[group_s_idx, 0:group_s_sz, 0])

@nki.compiler.skip_middle_end_transformations
@nki.jit(kernel_return=False)
def rmsnorm_quant(hidden, ln_w, lower_bound, out, kernel_name, eps=1e-6):
  """See docstring of compatible_rmsnorm_quant. 
  
  The key difference is this kernel takes the output tensor `out` and writes to it by reference
  (and does not return any result) while compatible_rmsnorm_quant returns the output tensor.
  """
  do_rmsnorm = kernel_name == "RMSNormQuant"

  B, S, H = hidden.shape

  ln_w = ln_w.reshape((1, H))

  GROUPS_PER_TILE_S = 1
  GROUP_S_SZ = nl.tile_size.pmax  # 128

  TILE_S_SZ = GROUPS_PER_TILE_S * GROUP_S_SZ
  NUM_TILE_S = ceil(S / TILE_S_SZ)

  GROUP_H_SZ = nl.tile_size.gemm_moving_fmax  # 512

  # MIN_VAL is a small constant used to upper bound dequantization scale for numerical stability
  # since quantization scale is computed as 1 / dequantization scale.
  MIN_VAL = 1e-6

  eps_bias = nisa.memset((GROUP_S_SZ, 1), value=eps, dtype=nl.bfloat16)

  # FIXME: tile on B * S instead of just S
  for batch in range(B):
    for tile_s in range(NUM_TILE_S):
      if do_rmsnorm:
        squared_sequence_group = nl.ndarray((GROUP_S_SZ, H), dtype=nl.float32, buffer=nl.sbuf)

      if tile_s == NUM_TILE_S - 1:
        tile_s_sz = S % TILE_S_SZ if S % TILE_S_SZ != 0 else TILE_S_SZ
        groups_per_tile_s = ceil(tile_s_sz / GROUP_S_SZ)

        if do_rmsnorm:
          broadcasted_ln_w = nl.ndarray((GROUP_S_SZ, GROUP_H_SZ), dtype=nl.bfloat16, buffer=nl.sbuf)
          inverse_rms_scale = nl.ndarray((groups_per_tile_s, nl.par_dim(GROUP_S_SZ), 1), dtype=nl.float32, buffer=nl.sbuf)

        quantized_sequence_tile = nl.ndarray((groups_per_tile_s, nl.par_dim(GROUP_S_SZ), H + 4), dtype=nl.float8_e4m3, buffer=nl.sbuf)
        sequence_tile = nl.ndarray((groups_per_tile_s, nl.par_dim(GROUP_S_SZ), H), dtype=nl.bfloat16, buffer=nl.sbuf)

        tile_dequant_scale = nl.ndarray((groups_per_tile_s, nl.par_dim(GROUP_S_SZ), 1), dtype=nl.float32, buffer=nl.sbuf)
        tile_quant_scale = nl.ndarray((groups_per_tile_s, nl.par_dim(GROUP_S_SZ), 1), dtype=nl.float32, buffer=nl.sbuf)
        
        for group_s in range(groups_per_tile_s):
          if group_s == groups_per_tile_s - 1:
            group_s_sz = tile_s_sz % GROUP_S_SZ if tile_s_sz % GROUP_S_SZ != 0 else GROUP_S_SZ
            group_s_base = tile_s * TILE_S_SZ + group_s * GROUP_S_SZ
            sequence_tile[group_s, :group_s_sz, :] = nl.load(hidden[batch, nl.ds(group_s_base, group_s_sz), :])
            
            if do_rmsnorm:
              rmsnorm_group(sequence_tile, squared_sequence_group, inverse_rms_scale, ln_w, broadcasted_ln_w, eps_bias, group_s_idx=group_s, group_s_sz=group_s_sz)
            quantize_group(sequence_tile, quantized_sequence_tile, tile_dequant_scale, tile_quant_scale, group_s_idx=group_s, group_s_sz=group_s_sz, lower_bound=lower_bound, min_val=MIN_VAL)
            
            nl.store(out[batch, nl.ds(group_s_base, group_s_sz), :H], value=quantized_sequence_tile[group_s, :group_s_sz, :H])
            nl.store(out[batch, nl.ds(group_s_base, group_s_sz), H:], value=tile_dequant_scale.view(nl.float8_e4m3)[group_s, :group_s_sz, :4])
          else:
            group_s_sz = GROUP_S_SZ
            group_s_base = tile_s * TILE_S_SZ + group_s * GROUP_S_SZ
            sequence_tile[group_s, :group_s_sz, :] = nl.load(hidden[batch, nl.ds(group_s_base, group_s_sz), :])

            if do_rmsnorm:
              rmsnorm_group(sequence_tile, squared_sequence_group, inverse_rms_scale, ln_w, broadcasted_ln_w, eps_bias, group_s_idx=group_s, group_s_sz=group_s_sz)
            quantize_group(sequence_tile, quantized_sequence_tile, tile_dequant_scale, tile_quant_scale, group_s_idx=group_s, group_s_sz=group_s_sz, lower_bound=lower_bound, min_val=MIN_VAL)

            nl.store(out[batch, nl.ds(group_s_base, group_s_sz), :H], value=quantized_sequence_tile[group_s, :group_s_sz, :H])
            nl.store(out[batch, nl.ds(group_s_base, group_s_sz), H:], value=tile_dequant_scale.view(nl.float8_e4m3)[group_s, :group_s_sz, :4])
      else:
        tile_s_sz = TILE_S_SZ
        groups_per_tile_s = ceil(tile_s_sz / GROUP_S_SZ)

        if do_rmsnorm:
          broadcasted_ln_w = nl.ndarray((GROUP_S_SZ, GROUP_H_SZ), dtype=nl.bfloat16, buffer=nl.sbuf)
          inverse_rms_scale = nl.ndarray((groups_per_tile_s, nl.par_dim(GROUP_S_SZ), 1), dtype=nl.float32, buffer=nl.sbuf)

        quantized_sequence_tile = nl.ndarray((groups_per_tile_s, nl.par_dim(GROUP_S_SZ), H + 4), dtype=nl.float8_e4m3, buffer=nl.sbuf)
        sequence_tile = nl.ndarray((groups_per_tile_s, nl.par_dim(GROUP_S_SZ), H), dtype=nl.bfloat16, buffer=nl.sbuf)
  
        tile_dequant_scale = nl.ndarray((groups_per_tile_s, nl.par_dim(GROUP_S_SZ), 1), dtype=nl.float32, buffer=nl.sbuf)
        tile_quant_scale = nl.ndarray((groups_per_tile_s, nl.par_dim(GROUP_S_SZ), 1), dtype=nl.float32, buffer=nl.sbuf)
        
        for group_s in range(groups_per_tile_s):
          if group_s == groups_per_tile_s - 1:
            group_s_sz = tile_s_sz % GROUP_S_SZ if tile_s_sz % GROUP_S_SZ != 0 else GROUP_S_SZ
            group_s_base = tile_s * TILE_S_SZ + group_s * GROUP_S_SZ
            sequence_tile[group_s, :group_s_sz, :] = nl.load(hidden[batch, nl.ds(group_s_base, group_s_sz), :])

            if do_rmsnorm:
              rmsnorm_group(sequence_tile, squared_sequence_group, inverse_rms_scale, ln_w, broadcasted_ln_w, eps_bias, group_s_idx=group_s, group_s_sz=group_s_sz)
            quantize_group(sequence_tile, quantized_sequence_tile, tile_dequant_scale, tile_quant_scale, group_s_idx=group_s, group_s_sz=group_s_sz, lower_bound=lower_bound, min_val=MIN_VAL)

            nl.store(out[batch, nl.ds(group_s_base, group_s_sz), :H], value=quantized_sequence_tile[group_s, :group_s_sz, :H])
            nl.store(out[batch, nl.ds(group_s_base, group_s_sz), H:], value=tile_dequant_scale.view(nl.float8_e4m3)[group_s, :group_s_sz, :4])
          else:
            group_s_sz = GROUP_S_SZ
            group_s_base = tile_s * TILE_S_SZ + group_s * GROUP_S_SZ
            sequence_tile[group_s, :group_s_sz, :] = nl.load(hidden[batch, nl.ds(group_s_base, group_s_sz), :])

            if do_rmsnorm:
              rmsnorm_group(sequence_tile, squared_sequence_group, inverse_rms_scale, ln_w, broadcasted_ln_w, eps_bias, group_s_idx=group_s, group_s_sz=group_s_sz)
            quantize_group(sequence_tile, quantized_sequence_tile, tile_dequant_scale, tile_quant_scale, group_s_idx=group_s, group_s_sz=group_s_sz, lower_bound=lower_bound, min_val=MIN_VAL)

            nl.store(out[batch, nl.ds(group_s_base, group_s_sz), :H], value=quantized_sequence_tile[group_s, :group_s_sz, :H])
            nl.store(out[batch, nl.ds(group_s_base, group_s_sz), H:], value=tile_dequant_scale.view(nl.float8_e4m3)[group_s, :group_s_sz, :4])

# FIXME: Ideally, shard_rmsnorm_quant should create slices of hidden, ln_w, and out and call 
#   rmsnorm_quant to reduce code duplication.
@nki.compiler.skip_middle_end_transformations
@nki.jit(kernel_return=False)
def shard_rmsnorm_quant(hidden, ln_w, lower_bound, out, kernel_name, eps=1e-6):
  """See docstring of compatible_rmsnorm_quant. 
  
  The key difference is this kernel takes the output tensor `out` and writes to it by reference
  (and does not return any result) while compatible_rmsnorm_quant returns the output tensor.

  The compute of (potentially) RMSNorm and quantization on hidden of shape [B, S, H] along
  the S dimension.
  """
  do_rmsnorm = kernel_name == "RMSNormQuant"

  B, S, H = hidden.shape

  ln_w = ln_w.reshape((1, H))

  GROUPS_PER_TILE_S = 1
  GROUP_S_SZ = nl.tile_size.pmax  # 128

  num_shards = nl.num_programs(axes=0)
  assert S % num_shards == 0, f"VNC-sharded RMSNormQuant kernel requires sequence length ({S}) to be divisible by {num_shards} shards"
  shard_s_sz = S // num_shards

  shard_id = nl.program_id(axis=0)
  shard_s_offset = shard_id * shard_s_sz
  
  TILE_S_SZ = GROUPS_PER_TILE_S * GROUP_S_SZ
  NUM_TILE_S = ceil(shard_s_sz / TILE_S_SZ)

  GROUP_H_SZ = nl.tile_size.gemm_moving_fmax  # 512

  # MIN_VAL is a small constant used to upper bound dequantization scale for numerical stability
  # since quantization scale is computed as 1 / dequantization scale.
  MIN_VAL = 1e-6

  eps_bias = nisa.memset((GROUP_S_SZ, 1), value=eps, dtype=nl.bfloat16)

  # FIXME: tile on B * S instead of just S
  for batch in range(B):
    for tile_s in range(NUM_TILE_S):
      if do_rmsnorm:
        squared_sequence_group = nl.ndarray((GROUP_S_SZ, H), dtype=nl.float32, buffer=nl.sbuf)

      if tile_s == NUM_TILE_S - 1:
        tile_s_sz = shard_s_sz % TILE_S_SZ if shard_s_sz % TILE_S_SZ != 0 else TILE_S_SZ
        groups_per_tile_s = ceil(tile_s_sz / GROUP_S_SZ)

        if do_rmsnorm:
          broadcasted_ln_w = nl.ndarray((GROUP_S_SZ, GROUP_H_SZ), dtype=nl.bfloat16, buffer=nl.sbuf)
          inverse_rms_scale = nl.ndarray((groups_per_tile_s, nl.par_dim(GROUP_S_SZ), 1), dtype=nl.float32, buffer=nl.sbuf)

        quantized_sequence_tile = nl.ndarray((groups_per_tile_s, nl.par_dim(GROUP_S_SZ), H + 4), dtype=nl.float8_e4m3, buffer=nl.sbuf)
        sequence_tile = nl.ndarray((groups_per_tile_s, nl.par_dim(GROUP_S_SZ), H), dtype=nl.bfloat16, buffer=nl.sbuf)

        tile_dequant_scale = nl.ndarray((groups_per_tile_s, nl.par_dim(GROUP_S_SZ), 1), dtype=nl.float32, buffer=nl.sbuf)
        tile_quant_scale = nl.ndarray((groups_per_tile_s, nl.par_dim(GROUP_S_SZ), 1), dtype=nl.float32, buffer=nl.sbuf)
        
        for group_s in range(groups_per_tile_s):
          if group_s == groups_per_tile_s - 1:
            group_s_sz = tile_s_sz % GROUP_S_SZ if tile_s_sz % GROUP_S_SZ != 0 else GROUP_S_SZ
            group_s_base = shard_s_offset + tile_s * TILE_S_SZ + group_s * GROUP_S_SZ
            sequence_tile[group_s, :group_s_sz, :] = nl.load(hidden[batch, nl.ds(group_s_base, group_s_sz), :])
            
            if do_rmsnorm:
              rmsnorm_group(sequence_tile, squared_sequence_group, inverse_rms_scale, ln_w, broadcasted_ln_w, eps_bias, group_s_idx=group_s, group_s_sz=group_s_sz)
            quantize_group(sequence_tile, quantized_sequence_tile, tile_dequant_scale, tile_quant_scale, group_s_idx=group_s, group_s_sz=group_s_sz, lower_bound=lower_bound, min_val=MIN_VAL)
            
            nl.store(out[batch, nl.ds(group_s_base, group_s_sz), :H], value=quantized_sequence_tile[group_s, :group_s_sz, :H])
            nl.store(out[batch, nl.ds(group_s_base, group_s_sz), H:], value=tile_dequant_scale.view(nl.float8_e4m3)[group_s, :group_s_sz, :4])
          else:
            group_s_sz = GROUP_S_SZ
            group_s_base = shard_s_offset + tile_s * TILE_S_SZ + group_s * GROUP_S_SZ
            sequence_tile[group_s, :group_s_sz, :] = nl.load(hidden[batch, nl.ds(group_s_base, group_s_sz), :])

            if do_rmsnorm:
              rmsnorm_group(sequence_tile, squared_sequence_group, inverse_rms_scale, ln_w, broadcasted_ln_w, eps_bias, group_s_idx=group_s, group_s_sz=group_s_sz)
            quantize_group(sequence_tile, quantized_sequence_tile, tile_dequant_scale, tile_quant_scale, group_s_idx=group_s, group_s_sz=group_s_sz, lower_bound=lower_bound, min_val=MIN_VAL)

            nl.store(out[batch, nl.ds(group_s_base, group_s_sz), :H], value=quantized_sequence_tile[group_s, :group_s_sz, :H])
            nl.store(out[batch, nl.ds(group_s_base, group_s_sz), H:], value=tile_dequant_scale.view(nl.float8_e4m3)[group_s, :group_s_sz, :4])
      else:
        tile_s_sz = TILE_S_SZ
        groups_per_tile_s = ceil(tile_s_sz / GROUP_S_SZ)

        if do_rmsnorm:
          broadcasted_ln_w = nl.ndarray((GROUP_S_SZ, GROUP_H_SZ), dtype=nl.bfloat16, buffer=nl.sbuf)
          inverse_rms_scale = nl.ndarray((groups_per_tile_s, nl.par_dim(GROUP_S_SZ), 1), dtype=nl.float32, buffer=nl.sbuf)

        quantized_sequence_tile = nl.ndarray((groups_per_tile_s, nl.par_dim(GROUP_S_SZ), H + 4), dtype=nl.float8_e4m3, buffer=nl.sbuf)
        sequence_tile = nl.ndarray((groups_per_tile_s, nl.par_dim(GROUP_S_SZ), H), dtype=nl.bfloat16, buffer=nl.sbuf)
  
        tile_dequant_scale = nl.ndarray((groups_per_tile_s, nl.par_dim(GROUP_S_SZ), 1), dtype=nl.float32, buffer=nl.sbuf)
        tile_quant_scale = nl.ndarray((groups_per_tile_s, nl.par_dim(GROUP_S_SZ), 1), dtype=nl.float32, buffer=nl.sbuf)
        
        for group_s in range(groups_per_tile_s):
          if group_s == groups_per_tile_s - 1:
            group_s_sz = tile_s_sz % GROUP_S_SZ if tile_s_sz % GROUP_S_SZ != 0 else GROUP_S_SZ
            group_s_base = shard_s_offset + tile_s * TILE_S_SZ + group_s * GROUP_S_SZ
            sequence_tile[group_s, :group_s_sz, :] = nl.load(hidden[batch, nl.ds(group_s_base, group_s_sz), :])

            if do_rmsnorm:
              rmsnorm_group(sequence_tile, squared_sequence_group, inverse_rms_scale, ln_w, broadcasted_ln_w, eps_bias, group_s_idx=group_s, group_s_sz=group_s_sz)
            quantize_group(sequence_tile, quantized_sequence_tile, tile_dequant_scale, tile_quant_scale, group_s_idx=group_s, group_s_sz=group_s_sz, lower_bound=lower_bound, min_val=MIN_VAL)

            nl.store(out[batch, nl.ds(group_s_base, group_s_sz), :H], value=quantized_sequence_tile[group_s, :group_s_sz, :H])
            nl.store(out[batch, nl.ds(group_s_base, group_s_sz), H:], value=tile_dequant_scale.view(nl.float8_e4m3)[group_s, :group_s_sz, :4])
          else:
            group_s_sz = GROUP_S_SZ
            group_s_base = shard_s_offset + tile_s * TILE_S_SZ + group_s * GROUP_S_SZ
            sequence_tile[group_s, :group_s_sz, :] = nl.load(hidden[batch, nl.ds(group_s_base, group_s_sz), :])

            if do_rmsnorm:
              rmsnorm_group(sequence_tile, squared_sequence_group, inverse_rms_scale, ln_w, broadcasted_ln_w, eps_bias, group_s_idx=group_s, group_s_sz=group_s_sz)
            quantize_group(sequence_tile, quantized_sequence_tile, tile_dequant_scale, tile_quant_scale, group_s_idx=group_s, group_s_sz=group_s_sz, lower_bound=lower_bound, min_val=MIN_VAL)

            nl.store(out[batch, nl.ds(group_s_base, group_s_sz), :H], value=quantized_sequence_tile[group_s, :group_s_sz, :H])
            nl.store(out[batch, nl.ds(group_s_base, group_s_sz), H:], value=tile_dequant_scale.view(nl.float8_e4m3)[group_s, :group_s_sz, :4])

# FIXME: add an optional argument to respect autocast options passed in by the user
@nki.compiler.enable_stack_allocator
@nki.compiler.skip_middle_end_transformations
@nki.jit
def compatible_rmsnorm_quant(hidden, ln_w, lower_bound, kernel_name, eps=1e-6):
  """NKI kernel to either 
      (1) perform RMSNorm and quantize the normalized hidden over the hidden dimension (H).
      (2) quantize hidden over dimension H.
    
    The kernel supports no specialization, or specialization along 1 dimension.

  Args:
      hidden: Input tensor of shape [B, S, H] on HBM to be possibly normalized, and quantized.
      ln_w: Gamma weights of shape [H] or [1, H] used in RMSNorm.
      lower_bound: Non-negative float used for clipping dequantization scales and hidden (if no RMSNorm is performed)
        or normalized hidden (if RMSNorm is performed). Clipping is performed iff lower_bound > 0. That is, clipping
        is disabled if lower_bound == 0.
      kernel_name: A string that is one of "QuantOnly" or "RMSNormQuant". If kernel_name is "QuantOnly", only 
        quantization is performed. If kernel_name is "RMSNormQuant", both RMSNorm and quantization are performed.
      eps: A positive float (default 1e-6) to ensure numerical stability when computing RMSNorm.

  Returns:
      out: Output tensor of shape [B, S, H + 4] on HBM. out[:, :, :H] of shape [B, S, H] stores the possibly
        normalized, and quantized tensor. out[:, :, H:] of shape [B, S, 4] stores 4 fp8 floats (for each unique
        batch and sequence length index) which can be reinterpreted as a fp32 dequantization scale.

  NOTE:
      The autocast argument may NOT be respected properly.
  """
  grid_ndim = nl.program_ndim()
  assert grid_ndim == 0 or grid_ndim == 1, f"RMSNorm quantization kernel only supports no specialization, or specialization along one axis."

  assert kernel_name in ["QuantOnly", "RMSNormQuant"] 

  assert lower_bound >= 0, f"lower_bound ({lower_bound}) must be non-negative"
  assert eps > 0, f"eps ({eps}) must be positive"

  B, S, H = hidden.shape
  assert ln_w.shape == (1, H) or ln_w.shape == (H, ), f"ln_w.shape={ln_w.shape} should be equal to one of (1, H)={(1, H)}, or (H, )={(H, )}"

  out = nl.ndarray((B, S, H + 4), dtype=nl.float8_e4m3, buffer=nl.shared_hbm)

  perform_sharding = grid_ndim == 1 and nl.num_programs(axes=0) > 1

  if perform_sharding:
    shard_rmsnorm_quant(hidden, ln_w, lower_bound, out, kernel_name, eps)
  else:
    rmsnorm_quant(hidden, ln_w, lower_bound, out, kernel_name, eps)
  
  return out