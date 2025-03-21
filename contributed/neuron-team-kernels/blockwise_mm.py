"""
Copyright (c) 2025, Amazon.com. All Rights Reserved

Kernels written by the AWS Neuron.

WARNING: These kernels:
   - Are tested only against internal nightly builds
   - May not be compatible with public NeuronSDK releases
   - Have not been extensively tested across all input configurations
   - Carry no compatibility guarantees
   - The behavior of these kernels may be modified without prior notice

Blockwise matmul kernels

"""
import logging
from collections import namedtuple
from enum import IntEnum

import numpy as np

import neuronxcc.nki as nki
import neuronxcc.nki.compiler as ncc
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
import neuronxcc.nki.typing as nt
from neuronxcc.nki._private.private_api import do_while
from neuronxcc.nki.isa.constants import oob_mode
from common import div_ceil

SkipMode = namedtuple('SkipMode', ['skip_token', 'skip_weight'])

class ExpertAffinityScaleMode(IntEnum):
  NO_SCALE = 0
  POST_SCALE = 1
  PRE_SCALE = 2

# TILE_SIZE is fixed to 128
TILE_SIZE = 128
PSUM_SIZE = 512
N_PSUM_BANKS = 8
DVE_CHANNELS_PER_BANK = 32
TOTAL_PSUM_SIZE = PSUM_SIZE * N_PSUM_BANKS



def check_blockwise_mm_kernel_compatibility(hidden_size,
                                            block_size,
                                            intermediate_size_tp,
                                            ):
  available_block_sizes = [128, 256, 512, 1024]
  assert block_size in available_block_sizes, f"Only support block_size in {available_block_sizes}, found {block_size}"
  assert 4096 <= hidden_size <= 8192, f"Only support hidden dim size in range [4096, 8192], found {hidden_size}"
  assert hidden_size % PSUM_SIZE == 0, f"Hidden dim size must be multiples of {PSUM_SIZE}, found {hidden_size} "

  # FIXME: reenable after kwargs support
  # if ('quant' in kwargs and kwargs['quant'] and
  #         'gate_up_proj_weight_dtype' in kwargs and 'down_proj_weight_dtype' in kwargs):
  #   assert (kwargs['gate_up_proj_weight_dtype'] == np.int8 and
  #           kwargs['down_proj_weight_dtype'] == np.int8), "Only support weights of type int8 for quantization"


def blockwise_mm(
    hidden_states: nt.tensor,
    expert_affinities_masked: nt.tensor,
    gate_up_proj_weight: nt.tensor,
    down_proj_weight: nt.tensor,
    block_size: int,
    token_position_to_id: nt.tensor,
    block_to_expert: nt.tensor,
    output: nt.tensor,
    gate_up_proj_scale: nt.tensor=None,
    down_proj_scale: nt.tensor=None,
    gate_up_activations_T: nt.tensor=None,
    down_activations: nt.tensor=None,
    skip_dma: SkipMode = SkipMode(False, False),
    compute_type=nl.bfloat16,
    is_tensor_update_accumulating: bool=True,
    expert_affinities_scaling_mode: ExpertAffinityScaleMode=ExpertAffinityScaleMode.POST_SCALE,
    lnc: int=1,
  ):

  """
  The blockwise matrix multiplication (matmul) kernel implements a Mixture of Experts (MoE) 
  layer at a block granularity, offering an alternative to token dropping approaches. 
  This method assumes that tokens have already been assigned to blocks, as specified 
  by the user through the token_position_to_id parameter.

  Dimensions:
      H: Hidden dimension size
      T: Total number of input tokens (after linearizing across the batch dimension)
      B: Number of tokens per block
      N: Total number of blocks
      E: Number of experts
      I_TP: Intermediate size / tp degree

  Args:
      hidden_states (nt.tensor): Tensor of input hidden states on HBM of size (T+1, H). The reason it is T+1 is because padding token position is set to T.
                                      TODO: with skip_dma, id will be set to -1, so this shape can be (T, H). Similarly for expert_affinities_masked, output
      expert_affinities_masked (nt.tensor): Tensor of expert affinities corresponding to each token of size ((T+1) * E, 1). 
                                      TODO: cannot refactor to (T+1, E) as we currently don't support dynamic slice on both axis.
      gate_up_proj_weight (nt.tensor): Tensor of concatenated gate and up projection weights on HBM (E, H, 2, I_TP)
      down_proj_weight (nt.tensor): Tensor of down projection weights on HBM (E, I_TP, H)
      block_size (int): Number of tokens per block
      token_position_to_id (nt.tensor): Tensor of block index of the corresponding tokens on HBM (N * B,)
                                        Note that we include tokens included for padding purposes and N * B >= T.
                                        For padding token, id is set to T. TODO: with skip_dma, id will be set to -1.
      block_to_expert (nt.tensor): Tensor of expert indices of corresponding blocks on HBM (N, 1)
      skip_dma (bool): Whether to skip DMA operations (default: False)

  Returns:
      output (nt.tensor): Tensor of output hidden states on HBM of size (T+1, H).
      gate_up_activations_T (nt.tensor): Tensor of gate and up projection activations on HBM (2E, I_TP)
                                        If specified, store the intermediate activations of gate and up projection for training
      down_activations (nt.tensor): Tensor of down projection activations on HBM (E, I_TP)
                                    If specified, store the intermediate activations of down projection for training

  Notes:
      - All input/output tensors must have the same floating point dtype
      - token_position_to_id and block_to_expert must be np.int32 tensors

  Pseudo code:
      output = np.zeros()
      for n in range(num_blocks):
          local_token_position_to_id = token_position_to_id[n]
          block_expert_idx = block_to_expert[n]

          # block_hidden_states: (1, B, H) ; hidden_states: (T, H)
          block_hidden_states = hidden_states[local_token_position_to_id].unsqueeze(0)
          # block_mlp_output: (B, H)
          block_mlp_output = self.mlp_op(block_hidden_states, expert_indices=block_expert_idx).squeeze(0)
          # block_output: (B, H)
          block_output = block_mlp_output * expert_affinities_masked[local_token_position_to_id, block_expert_idx].unsqueeze(1)
          # Update the tokens computed by the block
          output[local_token_position_to_id] += block_output
  """
  # Infer Config from input shape
  T, H = hidden_states.shape
  B = block_size
  E, I_TP, _ = down_proj_weight.shape
  assert not (lnc != 2 and expert_affinities_scaling_mode == ExpertAffinityScaleMode.PRE_SCALE), "Only support prescale for blockwise_mm_baseline_shard_hidden"

  check_blockwise_mm_kernel_compatibility(H, B, I_TP)
  if lnc == 2:
    blockwise_mm_baseline_shard_hidden(
      hidden_states=hidden_states,
      expert_affinities_masked=expert_affinities_masked,
      gate_up_proj_weight=gate_up_proj_weight,
      down_proj_weight=down_proj_weight,
      block_size=block_size,
      token_position_to_id=token_position_to_id,
      block_to_expert=block_to_expert,
      output=output,
      gate_up_activations_T=gate_up_activations_T,
      down_activations=down_activations,
      skip_dma=skip_dma,
      compute_dtype=compute_type,
      is_tensor_update_accumulating=is_tensor_update_accumulating,
      expert_affinities_scaling_mode=expert_affinities_scaling_mode,
    )
  elif gate_up_proj_scale is not None and down_proj_scale is not None:
    # Quantize check
    blockwise_mm_baseline_quant(
      hidden_states=hidden_states,
      expert_affinities_masked=expert_affinities_masked,
      gate_up_proj_weight=gate_up_proj_weight,
      down_proj_weight=down_proj_weight,
      block_size=block_size,
      token_position_to_id=token_position_to_id,
      block_to_expert=block_to_expert,
      gate_up_proj_scale=gate_up_proj_scale,
      down_proj_scale=down_proj_scale,
      output=output,
      skip_dma=skip_dma,
      compute_dtype=compute_type,
      is_tensor_update_accumulating=is_tensor_update_accumulating,
      expert_affinities_scaling_mode=expert_affinities_scaling_mode,
    )
  elif H == 6144 and T == 4224 and E ==16 and B == 512 and I_TP == 336 and compute_type == nl.bfloat16 and gate_up_activations_T is None and down_activations is None and not skip_dma:
    # DBRX only, inference only, only trn1, no quantization, no skipping
    blockwise_mm_tuned_schedule_0(
      hidden_states=hidden_states,
      expert_affinities_masked=expert_affinities_masked,
      gate_up_proj_weight=gate_up_proj_weight,
      down_proj_weight=down_proj_weight,
      block_size=block_size,
      token_position_to_id=token_position_to_id,
      block_to_expert=block_to_expert,
      output=output,
      skip_dma=skip_dma,
      compute_dtype=compute_type,
      is_tensor_update_accumulating=is_tensor_update_accumulating,
      expert_affinities_scaling_mode=expert_affinities_scaling_mode,
    )
  else:
    blockwise_mm_baseline(
      hidden_states=hidden_states,
      expert_affinities_masked=expert_affinities_masked,
      gate_up_proj_weight=gate_up_proj_weight,
      down_proj_weight=down_proj_weight,
      block_size=block_size,
      token_position_to_id=token_position_to_id,
      block_to_expert=block_to_expert,
      output=output,
      skip_dma=skip_dma,
      compute_dtype=compute_type,
      is_tensor_update_accumulating=is_tensor_update_accumulating,
      expert_affinities_scaling_mode=expert_affinities_scaling_mode,
    )

def output_initialization(output):
  T, H = output.shape
  for tile_idx in nl.affine_range(int(np.ceil(T/TILE_SIZE))):
    zeros = nl.zeros((TILE_SIZE, H), dtype=output.dtype)
    store_p, store_f = nl.mgrid[0:TILE_SIZE, 0:H]
    nl.store(output[tile_idx * TILE_SIZE + store_p, store_f], 
             value=zeros[store_p, store_f],
             mask=tile_idx*TILE_SIZE + store_p < T)

def output_initialization_shard(output, num_shards, shard_id):
  T, H = output.shape
  H_per_shard = H // num_shards
  h_offset = H_per_shard * shard_id

  for tile_idx in nl.affine_range(int(np.ceil(T/TILE_SIZE))):
    zeros = nl.zeros((TILE_SIZE, H_per_shard), dtype=output.dtype)
    store_p, store_f = nl.mgrid[0:TILE_SIZE, 0:H_per_shard]
    nl.store(output[tile_idx * TILE_SIZE + store_p, store_f + h_offset], 
             value=zeros[store_p, store_f],
             mask=tile_idx*TILE_SIZE + store_p < T)
    
def load_gate_up_proj_weights(gate_up_proj_weight, block_expert, compute_dtype, 
                              skip_dma: SkipMode = SkipMode(False, False), load_dst=None, mask=None):
  assert len(gate_up_proj_weight.shape) == 4, "Unsupported gate_up_proj_weight layout, should be [E, H, 2, I_TP]"

  _, H, _, I_TP = gate_up_proj_weight.shape
  h_outer_tripcount = int(np.ceil(H/PSUM_SIZE))
  h_inner_tripcount = PSUM_SIZE // TILE_SIZE
  if load_dst is None:
    load_dst = nl.ndarray((h_outer_tripcount, h_inner_tripcount, nl.par_dim(TILE_SIZE), 2, I_TP), dtype=gate_up_proj_weight.dtype, buffer=nl.sbuf)
  if gate_up_proj_weight.dtype != compute_dtype:
    gup_weights = nl.ndarray((h_outer_tripcount, h_inner_tripcount, nl.par_dim(TILE_SIZE), 2, I_TP), dtype=compute_dtype, buffer=nl.sbuf)
  for h_i in nl.affine_range(h_outer_tripcount):
    for h_j in nl.affine_range(h_inner_tripcount):
      load_p, load_fgu, load_fi = nl.mgrid[0:TILE_SIZE, 0:2, 0:I_TP]
      load_dst[h_i, h_j, load_p, load_fgu, load_fi] = nl.load(
          gate_up_proj_weight[block_expert[0, 0],
                              PSUM_SIZE * h_i + TILE_SIZE * h_j + load_p,
                              load_fgu,
                              load_fi],
          dtype=gate_up_proj_weight.dtype,
          mask=mask,
          mode=oob_mode.skip if skip_dma.skip_weight else oob_mode.error)

      # TODO: should we try different cast size?
      if gate_up_proj_weight.dtype != compute_dtype:         
        gup_weights[h_i, h_j, 0:TILE_SIZE, 0:2, 0:I_TP] = nl.copy(load_dst[h_i, h_j, 0:TILE_SIZE, 0:2, 0:I_TP], 
                                                                  dtype=compute_dtype,
                                                                  mask=mask)

  return load_dst if gate_up_proj_weight.dtype == compute_dtype else gup_weights

def load_gate_up_proj_weights_shard(gate_up_proj_weight, block_expert, compute_dtype, 
                                    skip_dma, load_dst,
                                    num_shards, shard_id):
  assert len(gate_up_proj_weight.shape) == 4, "Unsupported gate_up_proj_weight layout, should be [E, H, 2, I_TP]"

  _, H, _, I_TP = gate_up_proj_weight.shape
  H_per_shard = H // num_shards
  h_offset = H_per_shard * shard_id

  h_outer_tripcount = int(np.ceil(H_per_shard/PSUM_SIZE))
  h_inner_tripcount = PSUM_SIZE // TILE_SIZE
  if load_dst is None:
    load_dst = nl.ndarray((h_outer_tripcount, h_inner_tripcount, nl.par_dim(TILE_SIZE), 2, I_TP), dtype=gate_up_proj_weight.dtype, buffer=nl.sbuf)

  if gate_up_proj_weight.dtype != compute_dtype:
    gup_weights = nl.ndarray((h_outer_tripcount, h_inner_tripcount, nl.par_dim(TILE_SIZE), 2, I_TP), dtype=compute_dtype, buffer=nl.sbuf)
  for h_i in nl.affine_range(h_outer_tripcount):
    for h_j in nl.affine_range(h_inner_tripcount):
      load_p, load_fgu, load_fi = nl.mgrid[0:TILE_SIZE, 0:2, 0:I_TP]
      load_dst[h_i, h_j, load_p, load_fgu, load_fi] = nl.load(
          gate_up_proj_weight[block_expert[0, 0],
                              PSUM_SIZE * h_i + TILE_SIZE * h_j + load_p + h_offset,
                              load_fgu,
                              load_fi],
          dtype=gate_up_proj_weight.dtype,
          mode=oob_mode.skip if skip_dma.skip_weight else oob_mode.error)

      # TODO: should we try different cast size?
      if gate_up_proj_weight.dtype != compute_dtype:         
        gup_weights[h_i, h_j, 0:TILE_SIZE, 0:2, 0:I_TP] = nl.copy(load_dst[h_i, h_j, 0:TILE_SIZE, 0:2, 0:I_TP], dtype=compute_dtype)

  return load_dst if gate_up_proj_weight.dtype == compute_dtype else gup_weights


def load_down_proj_weight(down_proj_weight, block_expert, compute_dtype, 
                          skip_dma: SkipMode = SkipMode(False, False), load_dst=None, mask=None):
  assert len(down_proj_weight.shape) == 3, "Unsupported down_proj_weight layout, should be [E, I_TP, H]"
  _, I_TP, H = down_proj_weight.shape
  gup_n_tile = int(np.ceil(I_TP/TILE_SIZE))
  if load_dst is None:
    load_dst = nl.ndarray((gup_n_tile, nl.par_dim(TILE_SIZE), H), dtype=down_proj_weight.dtype, buffer=nl.sbuf)

  if down_proj_weight.dtype != compute_dtype:
    dp_weights = nl.ndarray((gup_n_tile, nl.par_dim(TILE_SIZE), H), dtype=compute_dtype, buffer=nl.sbuf)
  for i_i in nl.sequential_range(gup_n_tile):
    load_p, load_f = nl.mgrid[0:TILE_SIZE, 0:H]
    mask = (load_p + TILE_SIZE * i_i < I_TP) if mask is None else (load_p + TILE_SIZE * i_i < I_TP) & mask
    load_dst[i_i, load_p, load_f] = nl.load(
                down_proj_weight[block_expert[0, 0], load_p + TILE_SIZE * i_i, load_f],
                dtype=down_proj_weight.dtype,
                mask=mask,
                mode=oob_mode.skip if skip_dma.skip_weight else oob_mode.error)
    # TODO: tile the copy
    if down_proj_weight.dtype != compute_dtype:
      dp_weights[i_i, load_p, load_f] = nl.copy(load_dst[i_i, load_p, load_f], 
                                                mask=mask,
                                                dtype=compute_dtype)

  return load_dst if down_proj_weight.dtype == compute_dtype else dp_weights

def load_down_proj_weight_shard(down_proj_weight, block_expert, compute_dtype, 
                                skip_dma, load_dst,
                                num_shards, shard_id):
  assert len(down_proj_weight.shape) == 3, "Unsupported down_proj_weight layout, should be [E, I_TP, H]"
  _, I_TP, H = down_proj_weight.shape
  gup_n_tile = int(np.ceil(I_TP/TILE_SIZE))
  H_per_shard = H // num_shards
  h_offset = H_per_shard * shard_id

  if load_dst is None:
    load_dst = nl.ndarray((gup_n_tile, nl.par_dim(TILE_SIZE), H_per_shard), dtype=down_proj_weight.dtype, buffer=nl.sbuf)

  dp_weights = nl.ndarray((gup_n_tile, nl.par_dim(TILE_SIZE), H), dtype=compute_dtype, buffer=nl.sbuf)
  for i_i in nl.affine_range(gup_n_tile):
    load_p, load_f = nl.mgrid[0:TILE_SIZE, 0:H_per_shard]
    load_dst[i_i, load_p, load_f] = nl.load(
                down_proj_weight[block_expert[0, 0], load_p + TILE_SIZE * i_i, load_f + h_offset],
                dtype=down_proj_weight.dtype,
                mask=(load_p + TILE_SIZE * i_i < I_TP),
                mode=oob_mode.skip if skip_dma.skip_weight else oob_mode.error)
    # TODO: tile the copy
    if down_proj_weight.dtype != compute_dtype:
      dp_weights[i_i, load_p, load_f] = nl.copy(load_dst[i_i, load_f, load_f], 
                                                mask=(load_p + TILE_SIZE * i_i < I_TP),
                                                dtype=compute_dtype)

  return load_dst if down_proj_weight.dtype == compute_dtype else dp_weights


def compute_same_weights(N, block_to_expert):
  # load all expert idx
  N_tile_minus_1 = int(np.ceil((N-1)/TILE_SIZE))
  load_p, load_f = nl.mgrid[0:TILE_SIZE, 0:N_tile_minus_1]
  all_expert_indices = nl.load(block_to_expert[load_f + N_tile_minus_1*load_p], dtype=np.int32, mask=load_f + N_tile_minus_1*load_p + 1 < N)
  all_expert_indices_off_one = nl.load(block_to_expert[load_f + N_tile_minus_1*load_p + 1], dtype=np.int32, mask=load_f + N_tile_minus_1*load_p + 1 < N)
  
  is_weight_same_as_prev = nl.ndarray((TILE_SIZE, N_tile_minus_1), dtype=np.uint8, buffer=nl.sbuf)
  is_weight_same_as_prev[0:TILE_SIZE, 0:N_tile_minus_1] = nisa.tensor_tensor(all_expert_indices, all_expert_indices_off_one, op=np.equal,
                                                                              mask=load_f + N_tile_minus_1*load_p + 1 < N)
  # add after equal to workaround alignment issue
  free_size = 4 - N_tile_minus_1%4
  zero_index = nl.zeros((1, free_size), dtype=np.uint8, buffer=nl.sbuf)

  # save to hbm to reduce sbuf mempressure
  is_weight_same_as_prev_hbm = nl.ndarray((N,), dtype=np.uint8, buffer=nl.hbm)
  nl.store(is_weight_same_as_prev_hbm[0], value=zero_index[0, 0])
  nl.store(is_weight_same_as_prev_hbm[load_f + N_tile_minus_1*load_p+ 1], 
            value=is_weight_same_as_prev[0:TILE_SIZE, 0:N_tile_minus_1],
            mask=load_f + N_tile_minus_1*load_p + 1 < N)
  return is_weight_same_as_prev_hbm

def compute_same_weights_block_parallel(N, block_to_expert, num_shards, shard_id):
  # load all expert idx
  N_shard = div_ceil(N, num_shards)
  N_tile_minus_1 = div_ceil(N_shard-1, TILE_SIZE)
  tile_size = min(TILE_SIZE, N_shard-1)
  load_p, load_f = nl.mgrid[0:tile_size, 0:N_tile_minus_1]
  mask = load_f + N_tile_minus_1*load_p + 1 + shard_id * N_shard < N
  all_expert_indices = nl.load(block_to_expert[load_f + N_tile_minus_1*load_p + shard_id * N_shard], dtype=np.int32, mask=mask)
  all_expert_indices_off_one = nl.load(block_to_expert[load_f + N_tile_minus_1*load_p + 1 + shard_id * N_shard], dtype=np.int32, mask=mask)
  
  # shard on neuron cores
  is_weight_same_as_prev = nl.ndarray((tile_size, N_tile_minus_1), dtype=np.uint8, buffer=nl.sbuf)
  is_weight_same_as_prev[0:tile_size, 0:N_tile_minus_1] = nisa.tensor_tensor(all_expert_indices, all_expert_indices_off_one, op=np.equal,
                                                                              mask=mask)
  # add after equal to workaround alignment issue
  free_size = 4 - N_tile_minus_1%4
  zero_index = nl.zeros((1, free_size), dtype=np.uint8, buffer=nl.sbuf)

  # save to hbm to reduce sbuf mempressure
  is_weight_same_as_prev_hbm = nl.ndarray((N_shard,), dtype=np.uint8, buffer=nl.hbm)
  nl.store(is_weight_same_as_prev_hbm[0], value=zero_index[0, 0])
  nl.store(is_weight_same_as_prev_hbm[load_f + N_tile_minus_1*load_p+ 1], 
            value=is_weight_same_as_prev[0:tile_size, 0:N_tile_minus_1],
            mask=mask)
  return is_weight_same_as_prev_hbm   

def create_block_hidden_states(H, NUM_TILES, dtype):
  block_hidden_states = nl.ndarray((NUM_TILES, nl.par_dim(TILE_SIZE), H), 
                                  dtype=dtype, buffer=nl.sbuf)
                                    
  return block_hidden_states

def load_token_indices(token_position_to_id, block_idx, B, NUM_TILES):
  """Load and transpose token indices for the current block."""
  return nl.load_transpose2d(
      token_position_to_id[TILE_SIZE * nl.arange(NUM_TILES)[:, None] + nl.arange(TILE_SIZE)[None, :] + block_idx * B],
      dtype=np.int32)

def load_block_expert(block_to_expert, block_idx, mask=None):
  """Load the expert for the current block."""
  return nl.load(block_to_expert[block_idx], dtype=np.int32, mask=mask)

def load_hidden_states(hidden_states, block_hidden_states, token_indices, NUM_TILES, dtype, skip_dma: SkipMode = SkipMode(False, False),):
  _, H = hidden_states.shape
  for n in nl.affine_range(0, NUM_TILES):
    load_p, load_f = nl.mgrid[0:TILE_SIZE, 0:H]
    if skip_dma.skip_token: 
      block_hidden_states[n, load_p, load_f] = nisa.memset((TILE_SIZE, H),
                                                    value=0, dtype=dtype)
    block_token_mapping = token_indices[nl.arange(TILE_SIZE)[:, None], n] 
    block_hidden_states[n, load_p, load_f] = nl.load(hidden_states[block_token_mapping, load_f],
                                                     dtype=dtype, 
                                                     mode=oob_mode.skip if skip_dma.skip_token else oob_mode.error)

def load_hidden_states_shard_with_scale(hidden_states, block_hidden_states, token_indices, expert_affinity, NUM_TILES, dtype, skip_dma, num_shards, shard_id):
  """Load hidden states for the current block."""
  _, H = hidden_states.shape
  H_per_shard = H // num_shards
  h_offset = H_per_shard * shard_id
  for n in nl.affine_range(0, NUM_TILES):
    load_p, load_f = nl.mgrid[0:TILE_SIZE, 0:H_per_shard]
    if skip_dma.skip_token: 
      block_hidden_states[n, load_p, load_f] = nisa.memset((TILE_SIZE, H_per_shard),
                                                    value=0, dtype=dtype)
    block_token_mapping = token_indices[nl.arange(TILE_SIZE)[:, None], n]
    block_hidden_states[n, load_p, load_f] = nl.load(hidden_states[block_token_mapping, load_f + h_offset],
                                                        dtype=dtype, mode=oob_mode.skip if skip_dma.skip_token else oob_mode.error)
    if expert_affinity is not None: # do prescaling
      # force it to use TSP
      block_hidden_states[n, 0:TILE_SIZE, 0:H_per_shard] = nisa.tensor_scalar(block_hidden_states[n, 0:TILE_SIZE, 0:H_per_shard],
                                                                       np.multiply,
                                                                       expert_affinity[n, 0:TILE_SIZE, 0],
                                                                       dtype=dtype)


def transpose_hidden_states_allocated(block_hidden_states, H, B, compute_dtype):
  """Transpose block hidden states from B x H to H x B."""
  h_outer_tripcount = int(np.ceil(H / PSUM_SIZE))
  h_inner_tripcount = PSUM_SIZE // TILE_SIZE
  block_psum_tiles = int(np.ceil(B / PSUM_SIZE))
  free_size = min(PSUM_SIZE, B)
  assert free_size % TILE_SIZE == 0, "Unsupported block size!"

  if nisa.get_nc_version() == nisa.nc_version.gen3:
    transpose_dtype = block_hidden_states.dtype
  else:
    transpose_dtype = np.float32

  block_hidden_states_T = nl.ndarray((h_outer_tripcount, h_inner_tripcount, 
                                      nl.par_dim(TILE_SIZE), block_psum_tiles, free_size), 
                                      dtype=compute_dtype, buffer=nl.sbuf)
  
  # TODO: swap lhs/rhs for better performance based on configs
  
  block_free_tiles = min(PSUM_SIZE // TILE_SIZE, B // TILE_SIZE)

  for n in nl.affine_range(block_psum_tiles):
    for h_i in nl.affine_range(h_outer_tripcount):
      for h_j in nl.affine_range(h_inner_tripcount):
        block_hidden_states_T_fp32 = nl.ndarray((nl.par_dim(TILE_SIZE), free_size), 
                                      dtype=transpose_dtype, buffer=nl.psum)
        for b_i in nl.affine_range(block_free_tiles):
          offset = TILE_SIZE * b_i
          trans_p, trans_f = nl.mgrid[0:TILE_SIZE, 0:TILE_SIZE]
          block_hidden_states_T_fp32[trans_p, trans_f + offset] = nisa.nc_transpose(
              block_hidden_states[block_free_tiles*n+b_i, 
                                  trans_p, 
                                  TILE_SIZE*h_j+PSUM_SIZE*h_i+trans_f],
              dtype=transpose_dtype)

        block_hidden_states_T[h_i, h_j, 0:TILE_SIZE, n, 0:free_size] = nl.copy(
          block_hidden_states_T_fp32[0:TILE_SIZE, 0:free_size], dtype=compute_dtype)
        
  return block_hidden_states_T

def transpose_hidden_states(block_hidden_states, H, B, compute_dtype=nl.bfloat16):
  """Transpose block hidden states from B x H to H x B."""
  h_outer_tripcount = int(np.ceil(H / PSUM_SIZE))
  h_inner_tripcount = PSUM_SIZE // TILE_SIZE
  block_psum_tiles = int(np.ceil(B / PSUM_SIZE))
  free_size = min(PSUM_SIZE, B)
  block_hidden_states_T = nl.ndarray((h_outer_tripcount, h_inner_tripcount, 
                                      nl.par_dim(TILE_SIZE), block_psum_tiles, free_size), 
                                      dtype=compute_dtype, buffer=nl.sbuf)
  
  block_free_tiles = min(PSUM_SIZE // TILE_SIZE, B // TILE_SIZE)
  
  for n in nl.affine_range(block_psum_tiles):
    for b_i in nl.affine_range(block_free_tiles):  
      for h_i in nl.affine_range(h_outer_tripcount):
        for h_j in nl.affine_range(h_inner_tripcount):
          offset = TILE_SIZE * b_i
          trans_p, trans_f = nl.mgrid[0:TILE_SIZE, 0:TILE_SIZE]
          block_hidden_states_T[h_i, h_j, trans_p, n, trans_f + offset] = nisa.nc_transpose(
              block_hidden_states[block_free_tiles*n+b_i, 
                                  trans_p,
                                  TILE_SIZE*h_j+PSUM_SIZE*h_i+trans_f],
              dtype=compute_dtype)
  return block_hidden_states_T

def compute_gate_and_up_projections(block_hidden_states_T, gup_weights, gate_up_activations_T, block_idx, B, H, I_TP):
  """Compute gate and up projections."""
  N_PSUM_TILE = int(np.ceil(B / PSUM_SIZE))
  gup_n_tile = int(np.ceil(I_TP / TILE_SIZE))
  h_inner_tripcount = PSUM_SIZE // TILE_SIZE

  free_size = block_hidden_states_T.shape[-1]
  h_outer_tripcount = int(np.ceil(H / PSUM_SIZE))
  gate_and_up_proj_states = nl.ndarray((2, N_PSUM_TILE, gup_n_tile, nl.par_dim(TILE_SIZE), free_size), dtype=np.float32, buffer=nl.psum, lazy_initialization=True)
  for gate_or_up in nl.affine_range(2):
    for i_i in nl.affine_range(gup_n_tile):
      for h_i in nl.affine_range(h_outer_tripcount):
        for h_j in nl.affine_range(h_inner_tripcount):
          for b_i in nl.affine_range(N_PSUM_TILE):
            if_k = nl.arange(TILE_SIZE)[None, :]
            idx = if_k + TILE_SIZE * i_i
            gate_and_up_proj_states[gate_or_up, b_i, i_i, 0:TILE_SIZE, 0:free_size] += nisa.nc_matmul(
                gup_weights[h_i, h_j, nl.arange(TILE_SIZE)[:, None], gate_or_up, idx][idx < I_TP],
                block_hidden_states_T[h_i, h_j, 0:TILE_SIZE, b_i, 0:free_size],
                mask=(idx < I_TP ))

      # checkpoint activations for training
      if gate_up_activations_T is not None:
        for b_i in nl.affine_range(N_PSUM_TILE):
          store_p, store_f = nl.mgrid[0:TILE_SIZE, 0:free_size]
          i_output_intermediate = i_i * TILE_SIZE + store_p
          i_output_block = b_i * PSUM_SIZE + store_f
          nl.store(gate_up_activations_T[block_idx, gate_or_up, i_output_intermediate, i_output_block],
                   gate_and_up_proj_states[gate_or_up, b_i, i_i, store_p, store_f],
                   mask=(i_output_intermediate < I_TP))          
    
  return gate_and_up_proj_states

def compute_gate_and_up_projections_shard(block_hidden_states_T, gup_weights, gate_up_activations_T, block_idx, B, H, I_TP, shard_id):
  """Compute gate and up projections."""
  N_PSUM_TILE = int(np.ceil(B / PSUM_SIZE))
  gup_n_tile = int(np.ceil(I_TP / TILE_SIZE))
  h_inner_tripcount = PSUM_SIZE // TILE_SIZE
  free_size = block_hidden_states_T.shape[-1]

  h_outer_tripcount = int(np.ceil(H / PSUM_SIZE))
  gate_and_up_proj_states_psum = nl.ndarray((2, N_PSUM_TILE, gup_n_tile, nl.par_dim(TILE_SIZE), free_size), dtype=np.float32, buffer=nl.psum, lazy_initialization=True)
  gate_and_up_proj_states_sbuf = nl.ndarray((2, N_PSUM_TILE, gup_n_tile, nl.par_dim(TILE_SIZE), free_size), dtype=np.float32, buffer=nl.sbuf)
  gate_and_up_proj_states_sbuf_lnc_reduce = nl.ndarray((2, N_PSUM_TILE, gup_n_tile, nl.par_dim(TILE_SIZE), free_size), dtype=np.float32, buffer=nl.sbuf)
  for gate_or_up in nl.affine_range(2):
    for b_i in nl.affine_range(N_PSUM_TILE):
      for i_i in nl.affine_range(gup_n_tile):
        for h_i in nl.affine_range(h_outer_tripcount):
          for h_j in nl.affine_range(h_inner_tripcount):
            if_k = nl.arange(TILE_SIZE)[None, :]
            idx = if_k + TILE_SIZE * i_i
            gate_and_up_proj_states_psum[gate_or_up, b_i, i_i, 0:TILE_SIZE, 0:free_size] += nisa.nc_matmul(
                gup_weights[h_i, h_j, nl.arange(TILE_SIZE)[:, None], gate_or_up, idx][idx < I_TP],
                block_hidden_states_T[h_i, h_j, 0:TILE_SIZE, b_i, 0:free_size],
                mask=(idx < I_TP ))

        i_p, b_f = nl.mgrid[0:TILE_SIZE, 0:free_size]
        gate_and_up_proj_states_sbuf[gate_or_up, b_i, i_i, i_p, b_f] = nl.copy(gate_and_up_proj_states_psum[gate_or_up, b_i, i_i, i_p, b_f],
                                                                                                mask=(i_p + TILE_SIZE * i_i < I_TP))
        # Reduce across LNC cores
        gate_and_up_proj_states_sbuf_lnc_reduce[gate_or_up, b_i, i_i, i_p, b_f] = \
          nl.all_reduce(gate_and_up_proj_states_sbuf[gate_or_up, b_i, i_i, i_p, b_f], op=np.add, program_axes=[shard_id],
                        mask=(i_p + TILE_SIZE * i_i < I_TP))

        #TODO: Avoid double writes
        if gate_up_activations_T is not None:
          i_output_intermediate = i_i * TILE_SIZE + i_p
          i_output_block = b_i * PSUM_SIZE + b_f
          nl.store(
              gate_up_activations_T[
                block_idx,
                gate_or_up, 
                i_output_intermediate,
                i_output_block                
              ],
              gate_and_up_proj_states_sbuf_lnc_reduce[gate_or_up, b_i, i_i, i_p, b_f],
              mask=(i_output_intermediate < I_TP))

  return gate_and_up_proj_states_sbuf_lnc_reduce

def compute_intermediate_states(gate_and_up_proj_states, B, I_TP, dtype, mask=None):
  """Compute intermediate states with activation and gating."""
  N_PSUM_TILE = int(np.ceil(B / PSUM_SIZE))
  gup_n_tile = int(np.ceil(I_TP / TILE_SIZE))
  intermediate_states = nl.ndarray((gup_n_tile, nl.par_dim(TILE_SIZE), B), dtype=dtype, buffer=nl.sbuf)
  tmp = nl.ndarray((gup_n_tile, nl.par_dim(TILE_SIZE), B), dtype=dtype, buffer=nl.sbuf)
  bias = nl.zeros((TILE_SIZE, 1), dtype=np.float32)
  free_size = gate_and_up_proj_states.shape[-1]

  for i_i in nl.affine_range(gup_n_tile):
    for b_i in nl.affine_range(N_PSUM_TILE):
      start_idx = b_i * PSUM_SIZE
      end_idx = start_idx + free_size
      mask = nl.arange(TILE_SIZE)[:, None] + TILE_SIZE * i_i < I_TP if mask is None else (nl.arange(TILE_SIZE)[:, None] + TILE_SIZE * i_i < I_TP) & mask
      tmp[i_i, 0:TILE_SIZE, start_idx:end_idx] = nisa.activation(op=nl.silu,
                                                                 data=gate_and_up_proj_states[0, b_i, i_i, 0:TILE_SIZE, 0:free_size],
                                                                 scale=1.0,
                                                                 mask=mask,
                                                                 bias=bias,
                                                                 dtype=dtype)
      intermediate_states[i_i, 0:TILE_SIZE, start_idx:end_idx] = nl.multiply(tmp[i_i, 0:TILE_SIZE, start_idx:end_idx],
                                                                             gate_and_up_proj_states[1, b_i, i_i, 0:TILE_SIZE, 0:free_size],
                                                                             mask=mask,
                                                                             dtype=dtype)
  return intermediate_states

def compute_intermediate_states_with_quant(gate_and_up_proj_states, gate_and_up_proj_states_scale_f32, B, I_TP, compute_dtype):
  """Compute intermediate states with activation and gating."""
  # gate_and_up_proj_states scaling with quantization
  # we can do it right after the matmul, or fuse up scale with activation and multiply gate scale after activation
  # option 1:
  # for gate_or_up in nl.affine_range(2):
  #   for i_i in nl.affine_range(gup_n_tile):
  #     idx = 1 * nl.arange(TILE_SIZE)[:, None] + TILE_SIZE * i_i
  #     gate_and_up_proj_states[gate_or_up, i_i, 0:TILE_SIZE, 0:B] = nl.multiply(gate_and_up_proj_states[gate_or_up, i_i, 0:TILE_SIZE, 0:B], 
  #                                                                   gate_and_up_proj_states_scale[gate_or_up, i_i, 0:TILE_SIZE, 0], 
  #                                                                   mask=(idx < I_TP ))
  gup_n_tile = int(np.ceil(I_TP / TILE_SIZE))
  # activation and gate
  scale_gate_states = nl.ndarray((gup_n_tile, nl.par_dim(TILE_SIZE), B), dtype=compute_dtype, buffer=nl.sbuf)
  intermediate_states = nl.ndarray((gup_n_tile, nl.par_dim(TILE_SIZE), B), dtype=compute_dtype, buffer=nl.sbuf)
  tmp = nl.ndarray((gup_n_tile, nl.par_dim(TILE_SIZE), B), dtype=compute_dtype, buffer=nl.sbuf)
  N_PSUM_TILE = int(np.ceil(B / PSUM_SIZE))
  for i_i in nl.affine_range(gup_n_tile):
    for b_i in nl.affine_range(N_PSUM_TILE):
      start_idx = b_i * PSUM_SIZE
      end_idx = start_idx + PSUM_SIZE

      # option 2: apply up proj scaling, scale AP must be fp32
      tmp[i_i, 0:TILE_SIZE, start_idx:end_idx] = nisa.activation(op=nl.silu,
                                              data=gate_and_up_proj_states[0, b_i, i_i, 0:TILE_SIZE, 0:PSUM_SIZE],
                                              scale=gate_and_up_proj_states_scale_f32[0, i_i, 0:TILE_SIZE, 0],
                                              mask=(-1 * nl.arange(TILE_SIZE)[:, None] + -TILE_SIZE * i_i + I_TP-1 >= 0),
                                              dtype=compute_dtype)
      
      # cannot combine it with next multiply because this is TSP and the next is TT
      scale_gate_states[i_i, 0:TILE_SIZE, start_idx:end_idx] = nl.multiply(
          gate_and_up_proj_states[1, b_i, i_i, 0:TILE_SIZE, start_idx:end_idx],
          gate_and_up_proj_states_scale_f32[1, i_i, 0:TILE_SIZE, 0],
          mask=(-1 * nl.arange(TILE_SIZE)[:, None] + -TILE_SIZE * i_i + I_TP-1 >= 0),
          dtype=compute_dtype)
      
      intermediate_states[i_i, 0:TILE_SIZE, start_idx:end_idx] = nl.multiply(
          tmp[i_i, 0:TILE_SIZE, start_idx:end_idx],
          scale_gate_states[i_i, 0:TILE_SIZE, start_idx:end_idx],
          mask=(-1 * nl.arange(TILE_SIZE)[:, None] + -TILE_SIZE * i_i + I_TP-1 >= 0),
          dtype=compute_dtype)
        
  return intermediate_states

def load_old_block(output, token_indices, NUM_TILES, dtype, skip_dma: SkipMode = SkipMode(False, False)):
  """Load the old block from the output tensor."""
  _, H = output.shape
  block_old = nl.ndarray((NUM_TILES, nl.par_dim(TILE_SIZE), H), dtype=dtype, buffer=nl.sbuf)
  for n in nl.affine_range(NUM_TILES):
    if skip_dma.skip_token: 
      block_old[n, 0:TILE_SIZE, 0:H] = nisa.memset((TILE_SIZE, H),
                                                    value=0, dtype=dtype)
    block_token_mapping = token_indices[nl.arange(TILE_SIZE)[:, None], n]
    block_old[n, 0:TILE_SIZE, 0:H] = nl.load(output[block_token_mapping, nl.arange(H)[None, :]], 
                                              dtype=dtype, mode=oob_mode.skip if skip_dma.skip_token else oob_mode.error)
  return block_old

def load_old_block_shard(output, token_indices, NUM_TILES, dtype, num_shards, shard_id, skip_dma: SkipMode = SkipMode(False, False),):
  """Load the old block from the output tensor."""
  _, H = output.shape
  H_per_shard = H // num_shards
  h_offset = H_per_shard * shard_id

  block_old = nl.ndarray((NUM_TILES, nl.par_dim(TILE_SIZE), H_per_shard), dtype=dtype, buffer=nl.sbuf)
  for n in nl.affine_range(NUM_TILES):
    if skip_dma: 
      block_old[n, 0:TILE_SIZE, 0:H_per_shard] = nisa.memset((TILE_SIZE, H_per_shard),
                                                              value=0, dtype=dtype)
    block_token_mapping = token_indices[nl.arange(TILE_SIZE)[:, None], n]
    block_old[n, 0:TILE_SIZE, 0:H_per_shard] = nl.load(output[block_token_mapping, nl.arange(H_per_shard)[None, :] + h_offset], 
                                              dtype=dtype, mode=oob_mode.skip if skip_dma.skip_token else oob_mode.error)
  return block_old

def calculate_expert_affinities(expert_affinities_masked, token_indices, block_expert, E, NUM_TILES, dtype, skip_dma: SkipMode = SkipMode(False, False)):
  """Calculate expert affinities for the current block."""
  v_expert = nl.ndarray((nl.par_dim(TILE_SIZE), 1), dtype=np.int32, buffer=nl.sbuf)

  for i in nl.static_range(4):
    v_expert[DVE_CHANNELS_PER_BANK * i:DVE_CHANNELS_PER_BANK * (i + 1), 0] = block_expert.broadcast_to([DVE_CHANNELS_PER_BANK, 1])

  expert_affinity_f32 = nl.ndarray((NUM_TILES, nl.par_dim(TILE_SIZE), 1), 
                                    dtype=nl.float32, buffer=nl.sbuf)

  for n in nl.affine_range(NUM_TILES):
    # Use pointer arithmetic to index into expert affinities
    addr = nl.multiply(token_indices[0:TILE_SIZE, n], E, dtype=np.int32)
    # Cast so that we can workaround TensorScalarAddr check
    # TODO: use TSA for add 
    v_expert = nl.copy(v_expert, dtype=np.float32)

    addr_fin = nl.add(addr[0:TILE_SIZE, 0], v_expert, dtype=np.int32) # Calculate address

    # Handle DMA skipping cases, not necessary?
    if skip_dma.skip_token: 
      addr_fin = nl.maximum(addr_fin, -1, dtype=np.int32)

    expert_affinity_dtype = nl.ndarray((TILE_SIZE, 1), dtype=dtype, buffer=nl.sbuf)
    if skip_dma.skip_token: 
      expert_affinity_dtype[0:TILE_SIZE, 0] = nisa.memset((TILE_SIZE, 1), value=0, dtype=dtype)

    expert_affinity_dtype[0:TILE_SIZE, 0] = nl.load(
        expert_affinities_masked[addr_fin[nl.arange(TILE_SIZE)[:, None], 0], nl.arange(1)[None, :]],
        dtype=dtype,
        mode=oob_mode.skip if skip_dma.skip_token else oob_mode.error)
    
    #if dtype != np.float32:
    # Cast to float32 to be compatible with tensorscalarptr
    expert_affinity_f32[n, 0:TILE_SIZE, 0] = nl.copy(expert_affinity_dtype[0:TILE_SIZE, 0], 
                                                      dtype=np.float32)
  return expert_affinity_f32
  #return (expert_affinity_f32 if dtype != np.float32 else expert_affinity_dtype)
    

def compute_block_output(intermediate_states, dp_weights, expert_affinity, block_old, down_activations, 
                         block_idx, H, I_TP, NUM_TILES, output_dtype, compute_dtype, is_tensor_update_accumulating, allocate=False,  mask=None):
  """Compute the new block output with down projection and expert affinity adjustment."""
  block_new = nl.ndarray((NUM_TILES, nl.par_dim(TILE_SIZE), H), dtype=output_dtype, buffer=nl.sbuf)
  gup_n_tile = int(np.ceil(I_TP / TILE_SIZE))
  h_i_upper = int(np.ceil(H/TOTAL_PSUM_SIZE))

  for n in nl.affine_range(NUM_TILES):
    for h_i in nl.affine_range(h_i_upper):
      buffer_type = nl.psum if not allocate else ncc.psum.mod_alloc(base_bank=0, num_bank_tiles=(N_PSUM_BANKS,))
      # TODO: fix psum initialization
      down_proj = nl.zeros((N_PSUM_BANKS, nl.par_dim(TILE_SIZE), PSUM_SIZE), dtype=np.float32, name='down_proj_psum', lazy_initialization=True,
                            buffer=buffer_type)
      for h_j in nl.affine_range(N_PSUM_BANKS):
        mask = (N_PSUM_BANKS*h_i+h_j < H//PSUM_SIZE) if mask is None else mask & (N_PSUM_BANKS*h_i+h_j < H//PSUM_SIZE)
        for i_i in nl.affine_range(gup_n_tile):
          down_proj[h_j, 0:TILE_SIZE, 0:PSUM_SIZE] += nisa.nc_matmul(
              intermediate_states[i_i, 0:TILE_SIZE, TILE_SIZE*n: TILE_SIZE*n + TILE_SIZE],
              dp_weights[i_i, nl.arange(TILE_SIZE)[:, None], TOTAL_PSUM_SIZE*h_i+PSUM_SIZE*h_j+nl.arange(PSUM_SIZE)[None, :]][-1*nl.arange(TILE_SIZE)[:, None]+-TILE_SIZE*i_i+I_TP-1 >= 0],
              mask=mask)

        if expert_affinity is not None:
          if is_tensor_update_accumulating:
            scaled_down_proj = nl.ndarray((nl.par_dim(TILE_SIZE), PSUM_SIZE), dtype=compute_dtype, buffer=nl.sbuf)
            scaled_down_proj[0:TILE_SIZE, 0:PSUM_SIZE] = nl.multiply(
                down_proj[h_j, 0:TILE_SIZE, 0:PSUM_SIZE],
                expert_affinity[n, 0:TILE_SIZE, 0],
                mask=mask,
                dtype=compute_dtype)

            block_new[n, nl.arange(TILE_SIZE)[:, None], TOTAL_PSUM_SIZE*h_i+PSUM_SIZE*h_j+nl.arange(PSUM_SIZE)[None, :]] = nl.add(
                block_old[n, nl.arange(TILE_SIZE)[:, None], TOTAL_PSUM_SIZE*h_i+nl.arange(PSUM_SIZE)[None, :]+PSUM_SIZE*h_j],
                scaled_down_proj[0:TILE_SIZE, 0:PSUM_SIZE],
                mask=mask,
                dtype=compute_dtype)
          else:
            block_new[n, nl.arange(TILE_SIZE)[:, None], TOTAL_PSUM_SIZE*h_i+PSUM_SIZE*h_j+nl.arange(PSUM_SIZE)[None, :]] = nl.multiply(
                down_proj[h_j, 0:TILE_SIZE, 0:PSUM_SIZE],
                expert_affinity[n, 0:TILE_SIZE, 0],
                mask=mask,
                dtype=compute_dtype)
        else:
          block_new[n, nl.arange(TILE_SIZE)[:, None], TOTAL_PSUM_SIZE*h_i+PSUM_SIZE*h_j+nl.arange(PSUM_SIZE)[None, :]] = nl.copy(
              down_proj[h_j, 0:TILE_SIZE, 0:PSUM_SIZE],
              mask=mask,
              dtype=compute_dtype)
          
        # checkpoint activations
        if down_activations is not None:
          i_output_block = n * TILE_SIZE + nl.arange(TILE_SIZE)[:, None]
          i_output_hidden = (N_PSUM_BANKS * h_i + h_j) * PSUM_SIZE + nl.arange(PSUM_SIZE)[None, :]
          nl.store(
              down_activations[
                block_idx, 
                i_output_block, 
                i_output_hidden,
              ],
              down_proj[h_j, 0:TILE_SIZE, 0:PSUM_SIZE],
              mask=mask)
          
  return block_new

def compute_block_output_with_quant(intermediate_states, dp_weights, down_proj_states_scale, expert_affinity, block_old, down_activations, 
                                    block_idx, H, I_TP, NUM_TILES, compute_dtype, output_dtype, is_tensor_update_accumulating):
  """Compute the new block output with down projection, dequant and expert affinity adjustment."""
  gup_n_tile = int(np.ceil(I_TP / TILE_SIZE))
  block_new = nl.ndarray((NUM_TILES, nl.par_dim(TILE_SIZE), H), dtype=output_dtype, buffer=nl.sbuf)

  assert down_activations is None, "checkpoint down projection activation is not supported yet"

  for n in nl.affine_range(NUM_TILES):
    
    down_proj = nl.zeros((H//PSUM_SIZE, nl.par_dim(TILE_SIZE), PSUM_SIZE), dtype=np.float32,  buffer=nl.psum, lazy_initialization=True)
    down_proj_scale = nl.ndarray((H//PSUM_SIZE, nl.par_dim(TILE_SIZE), PSUM_SIZE), dtype=np.float32,  buffer=nl.sbuf)
    h_i_upper = int(np.ceil(H/TOTAL_PSUM_SIZE))
    for h_i in nl.affine_range(h_i_upper):
      for h_j in nl.affine_range(N_PSUM_BANKS):
        for i_i in nl.affine_range(gup_n_tile):
          down_proj[N_PSUM_BANKS*h_i+h_j, 0:TILE_SIZE, 0:PSUM_SIZE] += nisa.nc_matmul(intermediate_states[i_i, 0:TILE_SIZE, TILE_SIZE*n: TILE_SIZE*n + TILE_SIZE],
                                                                           dp_weights[i_i, nl.arange(TILE_SIZE)[:, None], TOTAL_PSUM_SIZE*h_i+PSUM_SIZE*h_j+nl.arange(PSUM_SIZE)[None, :]][-1*nl.arange(TILE_SIZE)[:, None]+-TILE_SIZE*i_i+I_TP-1 >= 0], # I_TP x H
                                                                           mask=(N_PSUM_BANKS*h_i+h_j < H//PSUM_SIZE))

        # the two multiply will be combined
        down_proj_scale[N_PSUM_BANKS*h_i+h_j, 0:TILE_SIZE, 0:PSUM_SIZE] = nl.multiply(down_proj[N_PSUM_BANKS*h_i+h_j, 0:TILE_SIZE, 0:PSUM_SIZE],
                                                                           down_proj_states_scale[(N_PSUM_BANKS * h_i) + h_j, 0:1, 0:PSUM_SIZE],
                                                                           mask=(N_PSUM_BANKS*h_i+h_j < H//PSUM_SIZE),
                                                                           dtype=compute_dtype)
        if is_tensor_update_accumulating:
          # adjust affinities
          scaled_down_proj = nl.ndarray((nl.par_dim(TILE_SIZE), PSUM_SIZE), dtype=compute_dtype, buffer=nl.sbuf)
          scaled_down_proj[0:TILE_SIZE, 0:PSUM_SIZE] = nl.multiply(down_proj_scale[N_PSUM_BANKS*h_i+h_j, 0:TILE_SIZE, 0:PSUM_SIZE],
                                                                          expert_affinity[n, 0:TILE_SIZE, 0],
                                                                          mask=(N_PSUM_BANKS*h_i+h_j < H//PSUM_SIZE),
                                                                          dtype=compute_dtype)

          block_new[n, nl.arange(TILE_SIZE)[:, None], TOTAL_PSUM_SIZE*h_i+PSUM_SIZE*h_j+nl.arange(PSUM_SIZE)[None, :]] = nl.add(
            block_old[n, nl.arange(TILE_SIZE)[:, None], TOTAL_PSUM_SIZE*h_i+nl.arange(PSUM_SIZE)[None, :]+PSUM_SIZE*h_j],
            scaled_down_proj[0:TILE_SIZE, 0:PSUM_SIZE],
            mask=(N_PSUM_BANKS*h_i+h_j < H//PSUM_SIZE),
            dtype=output_dtype)
        else:
          block_new[n, nl.arange(TILE_SIZE)[:, None], TOTAL_PSUM_SIZE*h_i+PSUM_SIZE*h_j+nl.arange(PSUM_SIZE)[None, :]] = nl.multiply(down_proj_scale[N_PSUM_BANKS*h_i+h_j, 0:TILE_SIZE, 0:PSUM_SIZE],
                                                                          expert_affinity[n, 0:TILE_SIZE, 0],
                                                                          mask=(N_PSUM_BANKS*h_i+h_j < H//PSUM_SIZE),
                                                                          dtype=compute_dtype)
          
  return block_new

def compute_block_output_shard(intermediate_states, dp_weights, expert_affinity, block_old, down_activations, 
                               block_idx, H, I_TP, NUM_TILES, output_dtype, compute_dtype, 
                               is_tensor_update_accumulating, num_shards, shard_id, expert_affinities_scaling_mode=ExpertAffinityScaleMode.POST_SCALE):
  """Compute the new block output with down projection and expert affinity adjustment."""
  block_new = nl.ndarray((NUM_TILES, nl.par_dim(TILE_SIZE), H), dtype=output_dtype, buffer=nl.sbuf)
  gup_n_tile = int(np.ceil(I_TP / TILE_SIZE))

  H_per_shard = H // num_shards
  h_offset = H_per_shard * shard_id

  for n in nl.affine_range(NUM_TILES):
    down_proj = nl.zeros((H_per_shard//PSUM_SIZE, nl.par_dim(TILE_SIZE), PSUM_SIZE), dtype=np.float32,  buffer=nl.psum, lazy_initialization=True)
    h_i_upper = int(np.ceil(H_per_shard/TOTAL_PSUM_SIZE))
    for h_i in nl.affine_range(h_i_upper):
      for h_j in nl.affine_range(N_PSUM_BANKS):
        o_p, o_f = nl.mgrid[0:TILE_SIZE, 0:PSUM_SIZE]
        for i_i in nl.affine_range(gup_n_tile):
          i_p, i_f = nl.mgrid[0:TILE_SIZE, 0:TILE_SIZE]
          w_p, w_f = nl.mgrid[0:TILE_SIZE, 0:PSUM_SIZE]
          down_proj[N_PSUM_BANKS*h_i+h_j, o_p, o_f] += nisa.nc_matmul(intermediate_states[i_i, i_p, TILE_SIZE*n + i_f], 
                                                            dp_weights[i_i, w_p, TOTAL_PSUM_SIZE*h_i+PSUM_SIZE*h_j+w_f][w_p + TILE_SIZE*i_i < I_TP],
                                                            mask=(N_PSUM_BANKS*h_i+h_j < H_per_shard//PSUM_SIZE))

        if is_tensor_update_accumulating:
          assert expert_affinities_scaling_mode != ExpertAffinityScaleMode.PRE_SCALE, "affinity prescale with K > 1 not supported"
          if expert_affinity is not None:
            scaled_down_proj = nl.ndarray((nl.par_dim(TILE_SIZE), PSUM_SIZE), dtype=compute_dtype, buffer=nl.sbuf)
            # adjust affinities
            # using mgrid will generate tensortensor instead of TSP
            scaled_down_proj[0:TILE_SIZE, 0:PSUM_SIZE] = nl.multiply(down_proj[N_PSUM_BANKS*h_i+h_j, 0:TILE_SIZE, 0:PSUM_SIZE],
                                                    expert_affinity[n, 0:TILE_SIZE, 0],
                                                    mask=(N_PSUM_BANKS*h_i+h_j < H_per_shard//PSUM_SIZE),
                                                    dtype=compute_dtype)
          
            block_new[n, o_p, TOTAL_PSUM_SIZE*h_i+PSUM_SIZE*h_j+o_f] = nl.add(block_old[n, o_p, TOTAL_PSUM_SIZE*h_i + PSUM_SIZE*h_j + o_f],
                                                                              scaled_down_proj[o_p, o_f],
                                                                              mask=(N_PSUM_BANKS*h_i+h_j < H_per_shard//PSUM_SIZE),
                                                                              dtype=compute_dtype)
          else:
            block_new[n, o_p, TOTAL_PSUM_SIZE*h_i+PSUM_SIZE*h_j+o_f] = nl.add(block_old[n, o_p, TOTAL_PSUM_SIZE*h_i + PSUM_SIZE*h_j + o_f],
                                                                              down_proj[N_PSUM_BANKS*h_i+h_j, 0:TILE_SIZE, 0:PSUM_SIZE],
                                                                              mask=(N_PSUM_BANKS*h_i+h_j < H_per_shard//PSUM_SIZE),
                                                                              dtype=compute_dtype)
        elif expert_affinities_scaling_mode == ExpertAffinityScaleMode.POST_SCALE:
          assert block_old is None, "is_tensor_update_accumulating is False, shouldn't load output!"
          block_new[n, nl.arange(TILE_SIZE)[:, None], TOTAL_PSUM_SIZE*h_i+PSUM_SIZE*h_j+nl.arange(PSUM_SIZE)[None, :]] = nl.multiply(down_proj[N_PSUM_BANKS*h_i+h_j, nl.arange(TILE_SIZE)[:, None], nl.arange(PSUM_SIZE)[None, :]],
                                                                                expert_affinity[n, nl.arange(TILE_SIZE)[:, None], 0],
                                                                                mask=(N_PSUM_BANKS*h_i+h_j < H_per_shard//PSUM_SIZE),
                                                                                dtype=compute_dtype)
        else:
          block_new[n, nl.arange(TILE_SIZE)[:, None], TOTAL_PSUM_SIZE*h_i+PSUM_SIZE*h_j+nl.arange(PSUM_SIZE)[None, :]] = nl.copy(down_proj[N_PSUM_BANKS*h_i+h_j, nl.arange(TILE_SIZE)[:, None], nl.arange(PSUM_SIZE)[None, :]],
                                                                                                                                 mask=(N_PSUM_BANKS*h_i+h_j < H_per_shard//PSUM_SIZE))
        # checkpoint activations
        if down_activations is not None:
          i_output_intermediate = n * TILE_SIZE + o_p
          i_output_hidden = (N_PSUM_BANKS * h_i + h_j) * PSUM_SIZE + h_offset + o_f
          nl.store(
              down_activations[
                block_idx, 
                i_output_intermediate, 
                i_output_hidden,
              ],
              down_proj[N_PSUM_BANKS*h_i+h_j, o_p, o_f],
              mask=(N_PSUM_BANKS*h_i+h_j < H_per_shard//PSUM_SIZE))
          
  return block_new

def store_block_output(output, block_new, token_indices, NUM_TILES, skip_dma: SkipMode = SkipMode(False, False)):
  """Store the computed block output in the output tensor."""
  _, H = output.shape
  for n in nl.affine_range(NUM_TILES):
    nl.store(
        output[token_indices[nl.arange(TILE_SIZE)[:, None], n], nl.arange(H)[None, :]],
        value=block_new[n, 0:TILE_SIZE, 0:H],
        mode=oob_mode.skip if skip_dma.skip_token else oob_mode.error)

def store_block_output_shard(output, block_new, token_indices, NUM_TILES, num_shards, shard_id, skip_dma: SkipMode = SkipMode(False, False),):
  """Store the computed block output in the output tensor."""
  _, H = output.shape
  H_per_shard = H // num_shards
  h_offset = H_per_shard * shard_id
  for n in nl.affine_range(NUM_TILES):
    nl.store(
        output[token_indices[nl.arange(TILE_SIZE)[:, None], n], nl.arange(H_per_shard)[None, :] + h_offset],
        value=block_new[n, 0:TILE_SIZE, 0:H_per_shard],
        mode=oob_mode.skip if skip_dma.skip_token else oob_mode.error)

def load_gate_up_proj_scales(gate_up_proj_scale, gup_n_tile, I_TP):
  """Load and convert gate and up projection scales."""
  gate_and_up_proj_states_scale = nl.ndarray((2, gup_n_tile, nl.par_dim(
      TILE_SIZE), 1), dtype=gate_up_proj_scale.dtype, buffer=nl.sbuf)
  gate_and_up_proj_states_scale_f32 = nl.ndarray(
      (2, gup_n_tile, nl.par_dim(TILE_SIZE), 1), dtype=np.float32, buffer=nl.sbuf)
  
  for gate_or_up in nl.affine_range(2):
    for i_i in nl.affine_range(gup_n_tile):
      idx = nl.arange(TILE_SIZE)[:, ] + TILE_SIZE * \
          i_i + nl.arange(1)[None, :]
      gate_and_up_proj_states_scale[gate_or_up, i_i, 0:TILE_SIZE, 0] = nl.load(gate_up_proj_scale[0, 0, I_TP * gate_or_up + idx],
                                                                                mask=(idx < I_TP))
      gate_and_up_proj_states_scale_f32[gate_or_up, i_i, 0:TILE_SIZE, 0] = nl.copy(
          gate_and_up_proj_states_scale[gate_or_up, i_i, 0:TILE_SIZE, 0], dtype=np.float32)
  
  return gate_and_up_proj_states_scale_f32

def load_down_proj_scale(down_proj_scale, H):
  # Fix data orientation for broadcast along the dimension
  h_i_upper = int(np.ceil(H / PSUM_SIZE))
  down_proj_states_scale = nl.ndarray((h_i_upper, nl.par_dim(1), PSUM_SIZE),
                                       dtype=down_proj_scale.dtype,
                                       buffer=nl.sbuf)
  for h_i in nl.affine_range(h_i_upper):
    idx = 1 * nl.arange(PSUM_SIZE)[None, :] + PSUM_SIZE * h_i
    down_proj_states_scale[h_i, 0:1, 0:PSUM_SIZE] = nl.load(down_proj_scale[0, 0, idx],
                                                            mask=(idx < H))
  return down_proj_states_scale

@nki.compiler.enable_stack_allocator(log_level=logging.INFO)
@nki.compiler.skip_middle_end_transformations
@nki.jit
def blockwise_mm_baseline_allocated(hidden_states: nt.tensor,
                                    expert_affinities_masked: nt.tensor,
                                    gate_up_proj_weight: nt.tensor,
                                    down_proj_weight: nt.tensor,
                                    block_size: int,
                                    token_position_to_id: nt.tensor,
                                    block_to_expert: nt.tensor,
                                    skip_dma: SkipMode = SkipMode(False, False),
                                    compute_dtype=nl.bfloat16,
                                    is_tensor_update_accumulating=True,
                                    expert_affinities_scaling_mode=ExpertAffinityScaleMode.POST_SCALE):
  assert expert_affinities_scaling_mode == ExpertAffinityScaleMode.POST_SCALE, "only support postscale"
  # extract configuration shapes 
  T, H = hidden_states.shape
  B = block_size
  E, I_TP, _ = down_proj_weight.shape
  N = token_position_to_id.shape[0] // B
  NUM_TILES = B // TILE_SIZE

  check_blockwise_mm_kernel_compatibility(H, B, I_TP)

  output = nl.ndarray(hidden_states.shape, dtype=hidden_states.dtype,
                      buffer=nl.shared_hbm)
  
  output_initialization(output)
  h_outer_tripcount = int(np.ceil(H/PSUM_SIZE))
  h_inner_tripcount = PSUM_SIZE // TILE_SIZE
  gup_n_tile = int(np.ceil(I_TP/TILE_SIZE))
  
  gup_weights_load_dst = nl.ndarray((h_outer_tripcount, h_inner_tripcount, nl.par_dim(TILE_SIZE), 2, I_TP), dtype=gate_up_proj_weight.dtype, buffer=nl.sbuf)
  down_weights_load_dst = nl.ndarray((gup_n_tile, nl.par_dim(TILE_SIZE), H), dtype=down_proj_weight.dtype, buffer=nl.sbuf)

  if skip_dma.skip_weight:
    is_weight_same_as_prev_hbm = compute_same_weights(N, block_to_expert=block_to_expert)

  for block_idx in nl.sequential_range(N):
    token_indices = load_token_indices(token_position_to_id, block_idx, B, NUM_TILES)
    
    block_hidden_states = create_block_hidden_states(H, NUM_TILES, compute_dtype)

    load_hidden_states(hidden_states, block_hidden_states, token_indices, NUM_TILES, compute_dtype, skip_dma)
    
    block_hidden_states_T = transpose_hidden_states_allocated(block_hidden_states, H, B, compute_dtype)
    
    # MOVE it here to workaround the xbar-transpose alignment issue
    block_expert = load_block_expert(block_to_expert, block_idx)
    
    new_block_expert = nl.copy(block_expert)
    if skip_dma.skip_weight:
      # compute whether the expert idx is same as before
      need_skip = nl.load(is_weight_same_as_prev_hbm[block_idx], dtype=np.uint8)
      # add to workaround alignment issue
      bias2 = nl.zeros((1, 3), dtype=np.uint8, buffer=nl.sbuf)

      on_false = nl.full(shape=(1,1), fill_value=E, dtype=np.int32, buffer=nl.sbuf)
      new_block_expert = nl.where(need_skip, on_false, block_expert)

    gup_weights = load_gate_up_proj_weights(gate_up_proj_weight, new_block_expert, compute_dtype, skip_dma, load_dst=gup_weights_load_dst)
    
    gate_and_up_proj_states = compute_gate_and_up_projections(block_hidden_states_T, gup_weights, gate_up_activations_T=None,
                                                              block_idx=block_idx, B=B, H=H, I_TP=I_TP)
    
    intermediate_states = compute_intermediate_states(gate_and_up_proj_states, B, I_TP, compute_dtype)
    
    if is_tensor_update_accumulating:
      block_old = load_old_block(output, token_indices, NUM_TILES, compute_dtype, skip_dma)
    else:
      block_old = None

    dp_weights = load_down_proj_weight(down_proj_weight, new_block_expert, compute_dtype, skip_dma, load_dst=down_weights_load_dst)
 
    expert_affinity = calculate_expert_affinities(expert_affinities_masked, token_indices, block_expert, E, NUM_TILES, compute_dtype, skip_dma)

    block_new = compute_block_output(intermediate_states, dp_weights, expert_affinity, block_old, down_activations=None, 
                                     block_idx=block_idx, H=H, I_TP=I_TP, NUM_TILES=NUM_TILES, output_dtype=output.dtype, compute_dtype=compute_dtype, 
                                     is_tensor_update_accumulating=is_tensor_update_accumulating, allocate=True)
    
    store_block_output(output, block_new, token_indices, NUM_TILES, skip_dma)
  
  # TODO, support activation checkpoint
  if skip_dma.skip_weight:
    return output, is_weight_same_as_prev_hbm
  return output


def blockwise_mm_baseline(hidden_states: nt.tensor,
                          expert_affinities_masked: nt.tensor,
                          gate_up_proj_weight: nt.tensor,
                          down_proj_weight: nt.tensor,
                          block_size: int,
                          token_position_to_id: nt.tensor,
                          block_to_expert: nt.tensor,
                          output: nt.tensor,
                          gate_up_activations_T: nt.tensor=None,
                          down_activations: nt.tensor=None,
                          skip_dma: SkipMode = SkipMode(False, False),
                          compute_dtype=nl.bfloat16,
                          is_tensor_update_accumulating=True,
                          expert_affinities_scaling_mode=ExpertAffinityScaleMode.POST_SCALE):
  assert expert_affinities_scaling_mode == ExpertAffinityScaleMode.POST_SCALE, "only support postscale"
  # extract configuration shapes 
  T, H = hidden_states.shape
  B = block_size
  E, I_TP, _ = down_proj_weight.shape
  N = token_position_to_id.shape[0] // B
  NUM_TILES = B // TILE_SIZE

  check_blockwise_mm_kernel_compatibility(H, B, I_TP)

  output_initialization(output)

  if skip_dma.skip_weight:
    h_outer_tripcount = int(np.ceil(H/PSUM_SIZE))
    h_inner_tripcount = PSUM_SIZE // TILE_SIZE
    gup_n_tile = int(np.ceil(I_TP/TILE_SIZE))
    
    gup_weights_load_dst = nl.ndarray((h_outer_tripcount, h_inner_tripcount, nl.par_dim(TILE_SIZE), 2, I_TP), dtype=gate_up_proj_weight.dtype, buffer=nl.sbuf)
    down_weights_load_dst = nl.ndarray((gup_n_tile, nl.par_dim(TILE_SIZE), H), dtype=down_proj_weight.dtype, buffer=nl.sbuf)
    is_weight_same_as_prev_hbm = compute_same_weights(N, block_to_expert=block_to_expert)
  else:
    gup_weights_load_dst = None
    down_weights_load_dst = None

  for block_idx in nl.sequential_range(N):
    token_indices = load_token_indices(token_position_to_id, block_idx, B, NUM_TILES)
    block_expert = load_block_expert(block_to_expert, block_idx)

    new_block_expert = nl.copy(block_expert)
    if skip_dma.skip_weight:
      # compute whether the expert idx is same as before
      need_skip = nl.load(is_weight_same_as_prev_hbm[block_idx], dtype=np.uint8)
      on_false = nl.full(shape=(1,1), fill_value=E, dtype=np.int32, buffer=nl.sbuf)
      new_block_expert = nl.where(need_skip, on_false, block_expert)

    block_hidden_states = create_block_hidden_states(H, NUM_TILES, compute_dtype)
    load_hidden_states(hidden_states, block_hidden_states, token_indices, NUM_TILES, compute_dtype, skip_dma)
    
    block_hidden_states_T = transpose_hidden_states(block_hidden_states, H, B, compute_dtype)
    
    gup_weights = load_gate_up_proj_weights(gate_up_proj_weight, new_block_expert, compute_dtype, skip_dma, load_dst=gup_weights_load_dst)
    
    gate_and_up_proj_states = compute_gate_and_up_projections(block_hidden_states_T, gup_weights, gate_up_activations_T, block_idx, B, H, I_TP)
    
    intermediate_states = compute_intermediate_states(gate_and_up_proj_states, B, I_TP, compute_dtype)
    
    if is_tensor_update_accumulating:
      block_old = load_old_block(output, token_indices, NUM_TILES, compute_dtype, skip_dma=skip_dma)
    else:
      block_old = None
    
    dp_weights = load_down_proj_weight(down_proj_weight, new_block_expert, compute_dtype, skip_dma, load_dst=down_weights_load_dst)
    
    expert_affinity = calculate_expert_affinities(expert_affinities_masked, token_indices, block_expert, E, NUM_TILES, compute_dtype, skip_dma)
    
    block_new = compute_block_output(intermediate_states, dp_weights, expert_affinity, block_old, down_activations, 
                                     block_idx, H, I_TP, NUM_TILES, output_dtype=output.dtype, compute_dtype=compute_dtype, 
                                     is_tensor_update_accumulating=is_tensor_update_accumulating)
    
    store_block_output(output, block_new, token_indices, NUM_TILES, skip_dma)


def blockwise_mm_baseline_quant(hidden_states,
                                expert_affinities_masked,
                                gate_up_proj_weight,
                                down_proj_weight,
                                block_size,
                                token_position_to_id,
                                block_to_expert,
                                gate_up_proj_scale,
                                down_proj_scale,
                                output,
                                skip_dma: SkipMode = SkipMode(False, False),
                                compute_dtype=nl.bfloat16,
                                is_tensor_update_accumulating=True,
                                expert_affinities_scaling_mode=ExpertAffinityScaleMode.POST_SCALE):
  assert expert_affinities_scaling_mode == ExpertAffinityScaleMode.POST_SCALE, "only support post scale"
  # Infer Config from input shape
  T, H = hidden_states.shape
  B = block_size
  E, I_TP, _ = down_proj_weight.shape
  N = token_position_to_id.shape[0] // B
  NUM_TILES = B // TILE_SIZE

  assert gate_up_proj_scale.shape == (1, 1, 2 * I_TP), "Unsupported gate_up_proj_scale shape"
  assert down_proj_scale.shape == (1, 1, H), "Unsupported down_proj_scale shape"

  # FIXME: reenable after kwargs support
  # check_blockwise_mm_kernel_compatibility(
  #     H, B, I_TP, quant=True,
  #     gate_up_proj_weight_dtype=gate_up_proj_weight.dtype,
  #     down_proj_weight_dtype=down_proj_weight.dtype)

  output_initialization(output)
  
  gup_n_tile = int(np.ceil(I_TP / TILE_SIZE))
  # Since gate_up_proj_scale is not expert dependent, we can only it once before all blocks

  gate_and_up_proj_states_scale_f32 = load_gate_up_proj_scales(gate_up_proj_scale, gup_n_tile, I_TP)
  down_proj_states_scale = load_down_proj_scale(down_proj_scale, H)
  
  for block_idx in nl.sequential_range(N):
    token_indices = load_token_indices(token_position_to_id, block_idx, B, NUM_TILES)
    block_expert = load_block_expert(block_to_expert, block_idx)
    
    expert_affinity = calculate_expert_affinities(expert_affinities_masked, token_indices, block_expert, E, NUM_TILES, compute_dtype, skip_dma)
    
    block_hidden_states = create_block_hidden_states(H, NUM_TILES, compute_dtype)
    load_hidden_states(hidden_states, block_hidden_states, token_indices, NUM_TILES, compute_dtype, skip_dma)
    
    block_hidden_states_T = transpose_hidden_states(block_hidden_states, H, B, compute_dtype)
    
    gup_weights = load_gate_up_proj_weights(gate_up_proj_weight, block_expert, compute_dtype)

    dp_weights = load_down_proj_weight(down_proj_weight, block_expert, compute_dtype)

    gate_and_up_proj_states = compute_gate_and_up_projections(block_hidden_states_T, gup_weights, gate_up_activations_T=None,
                                                              block_idx=block_idx, B=B, H=H, I_TP=I_TP)
    if is_tensor_update_accumulating:
      block_old = load_old_block(output, token_indices, NUM_TILES, compute_dtype)
    else:
      block_old = None

    intermediate_states = compute_intermediate_states_with_quant(gate_and_up_proj_states, gate_and_up_proj_states_scale_f32, B, I_TP, compute_dtype)

    block_new = compute_block_output_with_quant(intermediate_states, dp_weights, down_proj_states_scale, expert_affinity, block_old, 
                                                down_activations=None, block_idx=block_idx, H=H, I_TP=I_TP, NUM_TILES=NUM_TILES, compute_dtype=compute_dtype, 
                                                output_dtype=output.dtype, is_tensor_update_accumulating=is_tensor_update_accumulating)
    store_block_output(output, block_new, token_indices, NUM_TILES)


# Best schedule for
#   H,    T,    E,  B,   TOPK, I_TP, dtype,    dma_skip
#   6144, 4224, 16, 512, 4,    336,  bfloat16, 0

def blockwise_mm_tuned_schedule_0(hidden_states,
                 expert_affinities_masked,
                 gate_up_proj_weight,
                 down_proj_weight,
                 block_size,
                 token_position_to_id,
                 block_to_expert,
                 output,
                 # Meta parameters
                 skip_dma: SkipMode = SkipMode(False, False),
                 compute_dtype=nl.bfloat16,
                 is_tensor_update_accumulating=True,
                 expert_affinities_scaling_mode=ExpertAffinityScaleMode.POST_SCALE):
  assert expert_affinities_scaling_mode == ExpertAffinityScaleMode.POST_SCALE, "only support postscale"

  # Infer Config from input shape
  T, H = hidden_states.shape
  B = block_size
  E, I_TP, _ = down_proj_weight.shape
  N = token_position_to_id.shape[0] // B
  NUM_TILES = B // TILE_SIZE

  check_blockwise_mm_kernel_compatibility(H, B, I_TP)

  output_initialization(output)

  for block_idx in nl.sequential_range(N):
    token_indices = load_token_indices(token_position_to_id, block_idx, B, NUM_TILES)
    block_expert = load_block_expert(block_to_expert, block_idx)

    block_hidden_states = create_block_hidden_states(H, NUM_TILES, compute_dtype)
    load_hidden_states(hidden_states, block_hidden_states, token_indices, NUM_TILES, compute_dtype, skip_dma)
    
    # prefetch expert_affinities
    expert_affinity = calculate_expert_affinities(expert_affinities_masked, token_indices, block_expert, E, NUM_TILES, compute_dtype, skip_dma)
    
    gup_weights = load_gate_up_proj_weights(gate_up_proj_weight, block_expert, compute_dtype=compute_dtype)
    
    block_hidden_states_T = transpose_hidden_states(block_hidden_states, H, B, compute_dtype=compute_dtype)
    
    # prefetch down proj weights
    dp_weights = load_down_proj_weight(down_proj_weight, block_expert, compute_dtype=compute_dtype)

    gate_and_up_proj_states = compute_gate_and_up_projections(block_hidden_states_T, gup_weights, gate_up_activations_T=None, 
                                                              block_idx=block_idx, B=B, H=H, I_TP=I_TP)
    if is_tensor_update_accumulating:
      block_old = load_old_block(output, token_indices, NUM_TILES, compute_dtype, skip_dma=skip_dma)
    else:
      block_old = None
    
    intermediate_states = compute_intermediate_states(gate_and_up_proj_states, B, I_TP, compute_dtype)

    block_new = compute_block_output(intermediate_states, dp_weights, expert_affinity, block_old, down_activations=None,
                                     block_idx=block_idx, H=H, I_TP=I_TP, NUM_TILES=NUM_TILES, output_dtype=output.dtype, compute_dtype=compute_dtype, 
                                     is_tensor_update_accumulating=is_tensor_update_accumulating)
    
    store_block_output(output, block_new, token_indices, NUM_TILES, skip_dma)


"""
Hand tuned cross block scheduling
"""
def blockwise_mm_prefetch_3_tiles(hidden_states,
                                  expert_affinities_masked,
                                  gate_up_proj_weight,
                                  down_proj_weight,
                                  block_size,
                                  token_position_to_id,
                                  block_to_expert,
                                  output,
                                  skip_dma: SkipMode = SkipMode(False, False),
                                  compute_dtype=nl.bfloat16,
                                  is_tensor_update_accumulating=True,
                                  expert_affinities_scaling_mode=ExpertAffinityScaleMode.POST_SCALE):
  assert expert_affinities_scaling_mode == ExpertAffinityScaleMode.POST_SCALE, "only support postscale"
  # Infer Config from input shape
  T, H = hidden_states.shape
  B = block_size
  E, I_TP, _ = down_proj_weight.shape
  N = token_position_to_id.shape[0] // B
  NUM_TILES = B // TILE_SIZE
  assert skip_dma.skip_token == False and skip_dma.skip_weight == False, "Don't support skip dma"
  assert is_tensor_update_accumulating == True, "only support is_tensor_update_accumulating = True"

  # FIXME: initialize a local HBM tile because the inoutparam doesn't work properly
  # the performance is slightly worse (155-165) if using the `output` tensor when first zeroing it
  output_initialization(output)
  # output = nl.zeros((T+1, H), dtype=nl.bfloat16, buffer=nl.hbm)

  ################################
  # PREFETCH block_hidden_states #
  ################################
  HIDDEN_PREFETCH_TILES = 3
  block_hidden_states = nl.ndarray((N, NUM_TILES, nl.par_dim(TILE_SIZE), H), dtype=nl.bfloat16, buffer=nl.sbuf)
  b_token_idxs0: nt.tensor[TILE_SIZE, NUM_TILES] = nl.load_transpose2d(
      token_position_to_id[TILE_SIZE * nl.arange(NUM_TILES)[:, None] + nl.arange(TILE_SIZE)[None, :] + 0 * B],
      dtype=np.int32)
  
  for n in nl.affine_range(0, HIDDEN_PREFETCH_TILES):
    block_hidden_states[0, n, 0:TILE_SIZE, 0:H] = nl.load(hidden_states[b_token_idxs0[nl.arange(TILE_SIZE)[:, None], n],
                                                                        nl.arange(H)[None, :]],
                                                            dtype=compute_dtype)

  for block_idx in nl.sequential_range(N):
    token_indices = load_token_indices(token_position_to_id, block_idx, B, NUM_TILES)
    block_expert = load_block_expert(block_to_expert, block_idx)

    ################################################
    # Loading block_hidden_states after prefetching#
    ################################################
    for n in nl.affine_range(HIDDEN_PREFETCH_TILES, NUM_TILES):
      block_hidden_states[block_idx, n, 0:TILE_SIZE, 0:H] = nl.load(hidden_states[token_indices[nl.arange(TILE_SIZE)[:, None], n],
                                                          nl.arange(H)[None, :]],
                                            dtype=compute_dtype)

    """Transpose block hidden states from B x H to H x B."""
    h_outer_tripcount = int(np.ceil(H / PSUM_SIZE))
    h_inner_tripcount = PSUM_SIZE // TILE_SIZE
    block_psum_tiles = int(np.ceil(B / PSUM_SIZE))
    free_size = min(PSUM_SIZE, B)
    block_hidden_states_T = nl.ndarray((h_outer_tripcount, h_inner_tripcount, 
                                        nl.par_dim(TILE_SIZE),block_psum_tiles, free_size), 
                                        dtype=compute_dtype, buffer=nl.sbuf)

    
    block_free_tiles = min(PSUM_SIZE // TILE_SIZE, B // TILE_SIZE)

    for n in nl.affine_range(block_psum_tiles):
      for h_i in nl.affine_range(h_outer_tripcount):
        for h_j in nl.affine_range(h_inner_tripcount):
          for b_i in nl.affine_range(block_free_tiles):
            offset = TILE_SIZE * b_i
            trans_p, trans_f = nl.mgrid[0:TILE_SIZE, 0:TILE_SIZE]
            block_hidden_states_T[h_i, h_j, trans_p, n, trans_f + offset] = nisa.nc_transpose(
                block_hidden_states[block_idx, 
                                    block_free_tiles*n+b_i, 
                                    trans_p,
                                    TILE_SIZE*h_j+PSUM_SIZE*h_i+trans_f],
                dtype=compute_dtype)
    
    gup_weights = load_gate_up_proj_weights(gate_up_proj_weight, block_expert, compute_dtype=compute_dtype)

    # prefetching it here significantly improve the performance
    dp_weights = load_down_proj_weight(down_proj_weight, block_expert, compute_dtype=compute_dtype)
    
    gate_and_up_proj_states = compute_gate_and_up_projections(block_hidden_states_T, gup_weights, gate_up_activations_T=None, 
                                                              block_idx=block_idx, B=B, H=H, I_TP=I_TP)

    intermediate_states = compute_intermediate_states(gate_and_up_proj_states, B, I_TP, compute_dtype)
    
    block_old = load_old_block(output, token_indices, NUM_TILES, compute_dtype)
    
    expert_affinity = calculate_expert_affinities(expert_affinities_masked, token_indices, block_expert, E, NUM_TILES, compute_dtype, skip_dma)
    
    ################################
    # PREFETCH block_hidden_states #
    ################################
    b_token_idxsN: nt.tensor[TILE_SIZE, NUM_TILES] = nl.load_transpose2d(
        token_position_to_id[TILE_SIZE * nl.arange(4)[:, None] + nl.arange(TILE_SIZE)[None, :] + (block_idx + 1) * B][block_idx < N-1],
        dtype=np.int32,
        mask=(block_idx < N-1))

    for n in nl.affine_range(0, HIDDEN_PREFETCH_TILES):
      block_hidden_states[block_idx+1, n, 0:TILE_SIZE, 0:H] = nl.load(hidden_states[b_token_idxsN[nl.arange(TILE_SIZE)[:, None], n],
                                                                        nl.arange(H)[None, :]],
                                                          dtype=compute_dtype,
                                                          mask=(block_idx < N-1))

    block_new = compute_block_output(intermediate_states, dp_weights, expert_affinity, block_old, down_activations=None, 
                                     block_idx=block_idx, H=H, I_TP=I_TP, NUM_TILES=NUM_TILES, output_dtype=output.dtype, compute_dtype=compute_dtype, 
                                     is_tensor_update_accumulating=True)
    
    store_block_output(output, block_new, token_indices, NUM_TILES)

    

def blockwise_mm_baseline_shard_hidden(hidden_states,
                                      expert_affinities_masked,
                                      gate_up_proj_weight,
                                      down_proj_weight,
                                      block_size,
                                      token_position_to_id,
                                      block_to_expert,
                                      output,
                                      gate_up_activations_T=None,
                                      down_activations=None,
                                      # Meta parameters
                                      skip_dma: SkipMode = SkipMode(False, False),
                                      compute_dtype=nl.bfloat16,
                                      is_tensor_update_accumulating=True,
                                      expert_affinities_scaling_mode=ExpertAffinityScaleMode.POST_SCALE):
  # Infer Config from input shape
  T, H = hidden_states.shape
  B = block_size
  E, I_TP, _ = down_proj_weight.shape
  N = token_position_to_id.shape[0] // B
  NUM_TILES = B // TILE_SIZE

  num_shards = nl.num_programs(axes=0)
  shard_id = nl.program_id(axis=0)

  # shard over H
  H_per_shard = H // num_shards
  assert H % num_shards == 0, f"Expect hidden dim is shardable by {num_shards}"

  output_initialization_shard(output, num_shards, shard_id)
  if skip_dma.skip_weight:
    h_outer_tripcount = int(np.ceil(H_per_shard/PSUM_SIZE))
    h_inner_tripcount = PSUM_SIZE // TILE_SIZE
    gup_n_tile = int(np.ceil(I_TP/TILE_SIZE))
    
    gup_weights_load_dst = nl.ndarray((h_outer_tripcount, h_inner_tripcount, nl.par_dim(TILE_SIZE), 2, I_TP), dtype=gate_up_proj_weight.dtype, buffer=nl.sbuf)
    down_weights_load_dst = nl.ndarray((gup_n_tile, nl.par_dim(TILE_SIZE), H_per_shard), dtype=down_proj_weight.dtype, buffer=nl.sbuf)
    is_weight_same_as_prev_hbm = compute_same_weights(N, block_to_expert=block_to_expert)
  else:
    gup_weights_load_dst = None
    down_weights_load_dst = None

  for block_idx in nl.sequential_range(N):
    token_indices = load_token_indices(token_position_to_id, block_idx, B, NUM_TILES)
    block_expert = load_block_expert(block_to_expert, block_idx)
    new_block_expert = nl.copy(block_expert)
    if skip_dma.skip_weight:
      # compute whether the expert idx is same as before
      need_skip = nl.load(is_weight_same_as_prev_hbm[block_idx], dtype=np.uint8)
      on_false = nl.full(shape=(1,1), fill_value=E, dtype=np.int32, buffer=nl.sbuf)
      new_block_expert = nl.where(need_skip, on_false, block_expert)

    expert_affinity = None
    if expert_affinities_scaling_mode == ExpertAffinityScaleMode.PRE_SCALE:
      expert_affinity = calculate_expert_affinities(expert_affinities_masked, token_indices, block_expert, E, NUM_TILES, compute_dtype, skip_dma)

    block_hidden_states = create_block_hidden_states(H_per_shard, NUM_TILES, compute_dtype)
    load_hidden_states_shard_with_scale(hidden_states, block_hidden_states, token_indices, expert_affinity, NUM_TILES, compute_dtype, skip_dma, num_shards, shard_id)
    block_hidden_states_T = transpose_hidden_states(block_hidden_states, H_per_shard, B, compute_dtype)

    gup_weights = load_gate_up_proj_weights_shard(gate_up_proj_weight, new_block_expert, compute_dtype, 
                                                  skip_dma=skip_dma,
                                                  load_dst=gup_weights_load_dst,
                                                  num_shards=num_shards, shard_id=shard_id)
    
    gate_and_up_proj_states = compute_gate_and_up_projections_shard(block_hidden_states_T, gup_weights, gate_up_activations_T, block_idx, B, H_per_shard, I_TP, shard_id)

    intermediate_states = compute_intermediate_states(gate_and_up_proj_states, B, I_TP, compute_dtype)

    if is_tensor_update_accumulating:
      block_old = load_old_block_shard(output, token_indices, NUM_TILES, compute_dtype, num_shards, shard_id, skip_dma=skip_dma)
    else:
      block_old = None

    dp_weights = load_down_proj_weight_shard(down_proj_weight, new_block_expert, compute_dtype, 
                                             skip_dma=skip_dma, 
                                             load_dst=down_weights_load_dst,
                                             num_shards=num_shards, shard_id=shard_id)

    if expert_affinities_scaling_mode == ExpertAffinityScaleMode.POST_SCALE:
      expert_affinity = calculate_expert_affinities(expert_affinities_masked, token_indices, block_expert, E, NUM_TILES, compute_dtype, skip_dma)

    block_new = compute_block_output_shard(intermediate_states, dp_weights, expert_affinity, block_old, down_activations, 
                                          block_idx, H, I_TP, NUM_TILES, output_dtype=output.dtype, compute_dtype=compute_dtype, 
                                          is_tensor_update_accumulating=is_tensor_update_accumulating,
                                          num_shards=num_shards, shard_id=shard_id, expert_affinities_scaling_mode=expert_affinities_scaling_mode)

    store_block_output_shard(output, block_new, token_indices, NUM_TILES, num_shards, shard_id, skip_dma)

def blockwise_mm_baseline_block_parallel(hidden_states,
                                        expert_affinities_masked,
                                        gate_up_proj_weight,
                                        down_proj_weight,
                                        block_size,
                                        token_position_to_id,
                                        block_to_expert,
                                        output,
                                        # Meta parameters
                                        skip_dma: SkipMode = SkipMode(False, False),
                                        compute_dtype=nl.bfloat16,
                                        is_tensor_update_accumulating=True,
                                        expert_affinities_scaling_mode=ExpertAffinityScaleMode.PRE_SCALE):


  # Infer Config from input shape
  T, H = hidden_states.shape
  B = block_size
  E, I_TP, _ = down_proj_weight.shape
  N = token_position_to_id.shape[0] // B
  NUM_TILES = B // TILE_SIZE

  assert is_tensor_update_accumulating == False, "shard on N blocks is not support for K > 1"

  num_shards = nl.num_programs(axes=0)
  shard_id = nl.program_id(axis=0)

  N_shard = div_ceil(N, num_shards)
  
  if skip_dma.skip_weight:
    h_outer_tripcount = int(np.ceil(H/PSUM_SIZE))
    h_inner_tripcount = PSUM_SIZE // TILE_SIZE
    gup_n_tile = int(np.ceil(I_TP/TILE_SIZE))
    
    gup_weights_load_dst = nl.ndarray((h_outer_tripcount, h_inner_tripcount, nl.par_dim(TILE_SIZE), 2, I_TP), dtype=gate_up_proj_weight.dtype, buffer=nl.sbuf)
    down_weights_load_dst = nl.ndarray((gup_n_tile, nl.par_dim(TILE_SIZE), H), dtype=down_proj_weight.dtype, buffer=nl.sbuf)
    is_weight_same_as_prev_hbm = compute_same_weights_block_parallel(N, block_to_expert=block_to_expert, num_shards=num_shards, shard_id=shard_id)
  else:
    gup_weights_load_dst = None
    down_weights_load_dst = None
  
  MODULO_FACTOR = 3
  N_shard_tile = div_ceil(N_shard, MODULO_FACTOR)
  block_hidden_states = nl.zeros((MODULO_FACTOR, NUM_TILES, nl.par_dim(TILE_SIZE), H), 
                                  dtype=compute_dtype, buffer=nl.sbuf)
  token_indices = nl.ndarray((MODULO_FACTOR, nl.par_dim(TILE_SIZE), NUM_TILES), 
                                    dtype=np.int32, buffer=nl.sbuf)
  for outer_block_idx in nl.sequential_range(N_shard_tile):
    h_outer_tripcount = int(np.ceil(H / PSUM_SIZE))
    h_inner_tripcount = PSUM_SIZE // TILE_SIZE
    block_psum_tiles = int(np.ceil(B / PSUM_SIZE))
    free_size = min(PSUM_SIZE, B)
    block_hidden_states_T = nl.ndarray((MODULO_FACTOR, h_outer_tripcount, h_inner_tripcount, 
                                        nl.par_dim(TILE_SIZE), block_psum_tiles, free_size), 
                                        dtype=compute_dtype, buffer=nl.sbuf)
    
    # parallel load and transpose input
    for inner_block_idx in nl.affine_range(MODULO_FACTOR):
      block_idx = outer_block_idx * MODULO_FACTOR + inner_block_idx
      new_block_idx = block_idx + shard_id * N_shard
      mask = (new_block_idx < N) & (block_idx < N_shard)

      token_indices[inner_block_idx, nl.arange(TILE_SIZE)[:, None],  nl.arange(NUM_TILES)[None, :]] = nl.load_transpose2d(
        token_position_to_id[TILE_SIZE * nl.arange(NUM_TILES)[:, None] + nl.arange(TILE_SIZE)[None, :] + new_block_idx * B],
      dtype=np.int32, mask=mask)

      if expert_affinities_scaling_mode == ExpertAffinityScaleMode.PRE_SCALE:
        v_expert = nl.ndarray((nl.par_dim(TILE_SIZE), 1), dtype=np.int32, buffer=nl.sbuf)
        block_expert = load_block_expert(block_to_expert, new_block_idx, mask=mask)
        for i in nl.static_range(4):
          v_expert[DVE_CHANNELS_PER_BANK * i:DVE_CHANNELS_PER_BANK * (i + 1), 0] = block_expert.broadcast_to([DVE_CHANNELS_PER_BANK, 1])

        expert_affinity_f32 = nl.ndarray((NUM_TILES, nl.par_dim(TILE_SIZE), 1), dtype=nl.float32, buffer=nl.sbuf)
        for n in nl.affine_range(NUM_TILES):
          # Use pointer arithmetic to index into expert affinities
          addr = nl.multiply(token_indices[inner_block_idx, 0:TILE_SIZE, n], E, dtype=np.int32)
          # Cast so that we can workaround TensorScalarAddr check
          # TODO: use TSA for add 
          v_expert = nl.copy(v_expert, dtype=np.float32)
          addr_fin = nl.add(addr[0:TILE_SIZE, 0], v_expert, dtype=np.int32) # Calculate address

          if skip_dma.skip_token: 
            addr_fin = nl.maximum(addr_fin, -1, dtype=np.int32)

          expert_affinity_dtype = nl.ndarray((TILE_SIZE, 1), dtype=compute_dtype, buffer=nl.sbuf)
          if skip_dma.skip_token: 
            expert_affinity_dtype[0:TILE_SIZE, 0] = nisa.memset((TILE_SIZE, 1), value=0, dtype=compute_dtype)

          expert_affinity_dtype[0:TILE_SIZE, 0] = nl.load(
              expert_affinities_masked[addr_fin[nl.arange(TILE_SIZE)[:, None], 0], nl.arange(1)[None, :]],
              dtype=compute_dtype,
              mode=oob_mode.skip if skip_dma.skip_token else oob_mode.error)

          # Cast to float32 to be compatible with tensorscalarptr
          expert_affinity_f32[n, 0:TILE_SIZE, 0] = nl.copy(expert_affinity_dtype[0:TILE_SIZE, 0], 
                                                            dtype=np.float32)

      for n in nl.affine_range(0, NUM_TILES):
        load_p, load_f = nl.mgrid[0:TILE_SIZE, 0:H]
        block_token_mapping = token_indices[inner_block_idx, nl.arange(TILE_SIZE)[:, None], n] 
        block_hidden_states[inner_block_idx, n, load_p, load_f] = nl.load(hidden_states[block_token_mapping, load_f],
                                                        dtype=compute_dtype, 
                                                        mask=mask,
                                                        mode=oob_mode.skip if skip_dma.skip_token else oob_mode.error)
      
        if expert_affinities_scaling_mode == ExpertAffinityScaleMode.PRE_SCALE:
          #TODO: use vector engine to speed up
          block_hidden_states[inner_block_idx, n, 0:TILE_SIZE, 0:H] = nisa.tensor_scalar(block_hidden_states[inner_block_idx, n, 0:TILE_SIZE, 0:H],
                                                                        np.multiply,
                                                                        expert_affinity_f32[n, 0:TILE_SIZE, 0],
                                                                        dtype=compute_dtype, 
                                                                        mask=mask,
                                                                        engine=nisa.vector_engine)
      
      block_free_tiles = min(PSUM_SIZE // TILE_SIZE, B // TILE_SIZE) 
      for n in nl.affine_range(block_psum_tiles):
        for b_i in nl.affine_range(block_free_tiles):  
          for h_i in nl.affine_range(h_outer_tripcount):
            for h_j in nl.affine_range(h_inner_tripcount):
              offset = TILE_SIZE * b_i
              trans_p, trans_f = nl.mgrid[0:TILE_SIZE, 0:TILE_SIZE]
              block_hidden_states_T[inner_block_idx, h_i, h_j, trans_p, n, trans_f + offset] = nisa.nc_transpose(
                  block_hidden_states[inner_block_idx, block_free_tiles*n+b_i, 
                                      trans_p,
                                      TILE_SIZE*h_j+PSUM_SIZE*h_i+trans_f],
                  dtype=compute_dtype,
                  mask=mask)

    # sequential load weights and compute
    for inner_block_idx in nl.sequential_range(MODULO_FACTOR):
      block_idx = outer_block_idx * MODULO_FACTOR + inner_block_idx
      new_block_idx = block_idx + shard_id * N_shard
      mask = (new_block_idx < N) & (block_idx < N_shard)

      block_expert = load_block_expert(block_to_expert, new_block_idx, mask=mask)

      new_block_expert = nl.copy(block_expert)
      if skip_dma.skip_weight:
        # compute whether the expert idx is same as before
        need_skip = nl.load(is_weight_same_as_prev_hbm[block_idx], dtype=np.uint8, mask=block_idx<N_shard)
        on_false = nl.full(shape=(1,1), fill_value=E, dtype=np.int32, buffer=nl.sbuf)
        new_block_expert = nl.where(need_skip, on_false, block_expert, mask=block_idx<N_shard)

      gup_weights = load_gate_up_proj_weights(gate_up_proj_weight, new_block_expert, compute_dtype, skip_dma, load_dst=gup_weights_load_dst, mask=mask)

      dp_weights = load_down_proj_weight(down_proj_weight, new_block_expert, compute_dtype, skip_dma, load_dst=down_weights_load_dst, mask=mask)

      N_PSUM_TILE = int(np.ceil(B / PSUM_SIZE))
      gup_n_tile = int(np.ceil(I_TP / TILE_SIZE))
      h_inner_tripcount = PSUM_SIZE // TILE_SIZE

      free_size = block_hidden_states_T.shape[-1]
      h_outer_tripcount = int(np.ceil(H / PSUM_SIZE))    
      gate_and_up_proj_states = nl.ndarray((2, N_PSUM_TILE, gup_n_tile, nl.par_dim(TILE_SIZE), free_size), dtype=np.float32, buffer=nl.psum, lazy_initialization=True)
      for gate_or_up in nl.affine_range(2):
        for i_i in nl.affine_range(gup_n_tile):
          for h_i in nl.affine_range(h_outer_tripcount):
            for h_j in nl.affine_range(h_inner_tripcount):
              for b_i in nl.affine_range(N_PSUM_TILE):
                if_k = nl.arange(TILE_SIZE)[None, :]
                idx = if_k + TILE_SIZE * i_i
                gate_and_up_proj_states[gate_or_up, b_i, i_i, 0:TILE_SIZE, 0:free_size] += nisa.nc_matmul(
                    gup_weights[h_i, h_j, nl.arange(TILE_SIZE)[:, None], gate_or_up, idx][idx < I_TP],
                    block_hidden_states_T[inner_block_idx, h_i, h_j, 0:TILE_SIZE, b_i, 0:free_size],
                    mask=(idx < I_TP ) & mask)

      intermediate_states = compute_intermediate_states(gate_and_up_proj_states, B, I_TP, compute_dtype, mask=mask)
      
      block_old = None
      down_activations = None
      block_new = compute_block_output(intermediate_states, dp_weights, None, block_old, down_activations, 
                                      new_block_idx, H, I_TP, NUM_TILES, output_dtype=output.dtype, compute_dtype=compute_dtype,
                                      is_tensor_update_accumulating=is_tensor_update_accumulating, mask=mask)
      
      for n in nl.affine_range(NUM_TILES):
        nl.store(
        output[token_indices[inner_block_idx, nl.arange(TILE_SIZE)[:, None], n], nl.arange(H)[None, :]],
        value=block_new[n, 0:TILE_SIZE, 0:H],
        mode=oob_mode.skip if skip_dma.skip_token else oob_mode.error, mask=mask)



# prepare for While loop support
@nki.compiler.skip_middle_end_transformations
@nki.jit(mode='trace', experimental_flags='enable-mutable-parameter')
def blockwise_mm_baseline_while(conditions: nt.tensor,
                                hidden_states: nt.tensor,
                                expert_affinities_masked: nt.tensor,
                                gate_up_proj_weight: nt.tensor,
                                down_proj_weight: nt.tensor,
                                block_size: int,
                                token_position_to_id: nt.tensor,
                                block_to_expert: nt.tensor,
                                output: nt.mutable_tensor,
                                gate_up_activations_T: nt.tensor=None,
                                down_activations: nt.tensor=None,
                                skip_dma=False,
                                compute_dtype=nl.bfloat16, 
                                is_tensor_update_accumulating=True,
                                expert_affinities_scaling_mode=ExpertAffinityScaleMode.POST_SCALE):
  assert expert_affinities_scaling_mode == ExpertAffinityScaleMode.POST_SCALE, "only support postscale"
  # extract configuration shapes 
  T, H = hidden_states.shape
  B = block_size
  E, I_TP, _ = down_proj_weight.shape
  N = token_position_to_id.shape[0] // B
  NUM_TILES = B // TILE_SIZE

  # cond will be updated inside kernel

  check_blockwise_mm_kernel_compatibility(H, B, I_TP)

  output_initialization(output)

  cond = nl.ndarray((1, 1), buffer=nl.hbm, dtype=np.uint8)
  temp = nl.load(conditions[0], dtype=np.uint8)
  nl.store(dst=cond[0], value=temp[0, 0])
  index = nl.ndarray((1, 1), buffer=nl.hbm, dtype=np.int32)
  nl.store(dst=index[0, 0], value=0)

  with do_while(cond):
    block_idx = nl.load(index[0], dtype=np.int32)
    token_indices = load_token_indices(token_position_to_id, block_idx, B, NUM_TILES)
    block_expert = load_block_expert(block_to_expert, block_idx)
    
    block_hidden_states = create_block_hidden_states(H, NUM_TILES, compute_dtype)
    load_hidden_states(hidden_states, block_hidden_states, token_indices, NUM_TILES, compute_dtype, skip_dma)
    
    block_hidden_states_T = transpose_hidden_states(block_hidden_states, H, B, compute_dtype)
    
    gup_weights = load_gate_up_proj_weights(gate_up_proj_weight, block_expert, compute_dtype)
    
    gate_and_up_proj_states = compute_gate_and_up_projections(block_hidden_states_T, gup_weights, gate_up_activations_T, block_idx, B, H, I_TP)
    
    intermediate_states = compute_intermediate_states(gate_and_up_proj_states, B, I_TP, compute_dtype)
    
    block_old = load_old_block(output, token_indices, NUM_TILES, compute_dtype)
    
    dp_weights = load_down_proj_weight(down_proj_weight, block_expert, compute_dtype)
    
    expert_affinity = calculate_expert_affinities(expert_affinities_masked, token_indices, block_expert, E, NUM_TILES, compute_dtype, skip_dma)
    
    block_new = compute_block_output(intermediate_states, dp_weights, expert_affinity, block_old, down_activations, 
                                      block_idx, H, I_TP, NUM_TILES, output_dtype=output.dtype, compute_dtype=compute_dtype, 
                                      is_tensor_update_accumulating=is_tensor_update_accumulating)
    
    store_block_output(output, block_new, token_indices, NUM_TILES)
    block_idx_next = nl.add(block_idx, 1)

    # update index
    nl.store(index[0], block_idx_next[0, 0])
    # update conditions
    cond_next = nl.load(conditions[block_idx_next])
    nl.store(dst=cond[0], value=cond_next[0, 0])

    return output
