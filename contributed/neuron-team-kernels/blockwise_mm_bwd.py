"""
Copyright (c) 2025, Amazon.com. All Rights Reserved

Kernels written by the AWS Neuron.

WARNING: These kernels:
   - Are tested only against internal nightly builds
   - May not be compatible with public NeuronSDK releases
   - Have not been extensively tested across all input configurations
   - Carry no compatibility guarantees
   - The behavior of these kernels may be modified without prior notice

Blockwise matmul backward kernels

"""
from blockwise_mm import load_block_expert, load_token_indices, SkipMode
import numpy as np

import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
import neuronxcc.nki.typing as nt


# TILE_SIZE is fixed to 128
TILE_SIZE = 128
PSUM_SIZE = 512
N_PSUM_BANKS = 8
DVE_CHANNELS_PER_BANK = 32
TOTAL_PSUM_SIZE = PSUM_SIZE * N_PSUM_BANKS

def check_blockwise_mm_bwd_kernel_compatibility(hidden_size,
                                                block_size,
                                                intermediate_size_tp,
                                                ):
  assert block_size == 512, f"Only support block_size = 512, found {block_size}"
  assert 4096 <= hidden_size <= 8192, f"Only support hidden dim size in range [4096, 8192], found {hidden_size}"
  assert hidden_size % TILE_SIZE == 0, f"Hidden dim size must be multiples of {TILE_SIZE}, found {hidden_size} "
  assert intermediate_size_tp > 128, \
    f"Intermediate size//tp_degree must be larger than {TILE_SIZE}, found {intermediate_size_tp} "


def blockwise_mm_bwd(
    hidden_states: nt.tensor,
    hidden_states_grad: nt.tensor, 
    expert_affinities_masked: nt.tensor,
    expert_affinities_masked_grad: nt.tensor,
    gate_up_proj_weight: nt.tensor,
    gate_up_proj_weight_grad: nt.tensor,
    gate_up_activations_T: nt.tensor,
    down_proj_weight: nt.tensor,
    down_proj_weight_grad: nt.tensor,
    down_activations: nt.tensor, 
    token_position_to_id: nt.tensor,
    block_to_expert: nt.tensor, 
    output_hidden_states_grad: nt.tensor, 
    block_size: int=512,
    skip_dma: SkipMode = SkipMode(False, False),
    compute_type=nl.bfloat16,
    is_tensor_update_accumulating: bool=True,
    lnc: int=1,
):

  """
  Blockwise matmul backward kernel for MoE layer.

  H: Hidden dimension size
  T: total token_size (batch * sequence length)
  B: block size
  N: number of blocks
  E: number of experts
  I_TP: intermediate size / tp degree

  IO tensor layouts:
    - hidden_states: shape (T + 1, H)
    - hidden_states_grad: (T + 1, H) 
    - expert_affinities_masked: shape ((T + 1) * E, 1)
    - expert_affinities_masked_grad: ((T + 1) * E, 1)
    - gate_up_proj_weight: shape (E, H, 2, I_TP)
    - gate_up_proj_weight_grad: (E, H, 2, I_TP)
    - gate_up_proj_act_checkpoint_T: (N, 2, I_TP, B)
    - down_proj_weight: shape (E, I_TP, H)
    - down_proj_weight_grad:  (E, I_IP, H)
    - down_proj_act_checkpoint: (N, B, H) 
    - token_position_to_id: shape (N * B,)
    - block_to_expert: shape (N,)
    - ouput_hidden_states_grad: (T + 1, H)
    - block_size: (B)

  IO tensor dtypes:
    - block_size: int
    - token_position_to_id: np.int32 tensor
    - block_to_expert: np.int32 tensor
    - hidden_states, expert_affinities_masked, output must have the same floating point dtype

  """
  # Infer Config from input shape
  T, H = hidden_states.shape
  B = block_size
  E, I_TP, _ = down_proj_weight.shape

  check_blockwise_mm_bwd_kernel_compatibility(H, B, I_TP)

  if lnc == 2:
    blockwise_mm_bwd_baseline_shard_hidden(
        hidden_states,
        hidden_states_grad,
        expert_affinities_masked,
        expert_affinities_masked_grad,
        gate_up_proj_weight,
        gate_up_proj_weight_grad,
        gate_up_activations_T,
        down_proj_weight,
        down_proj_weight_grad,
        down_activations,
        token_position_to_id,
        block_to_expert,
        output_hidden_states_grad,
        block_size=block_size,
        skip_dma=skip_dma,
        compute_dtype=compute_type,
        is_tensor_update_accumulating=is_tensor_update_accumulating,
    )
  else:
    blockwise_mm_bwd_baseline(hidden_states,
                            hidden_states_grad,
                            expert_affinities_masked,
                            expert_affinities_masked_grad,
                            gate_up_proj_weight,
                            gate_up_proj_weight_grad,
                            gate_up_activations_T,
                            down_proj_weight,
                            down_proj_weight_grad,
                            down_activations,
                            token_position_to_id,
                            block_to_expert,
                            output_hidden_states_grad,
                            # Meta parameters
                            block_size=512,
                            skip_dma=skip_dma,
                            compute_dtype=compute_type,
                            is_tensor_update_accumulating=is_tensor_update_accumulating
                            )

def initialize_gradient_outputs(
  hidden_states_grad,
  expert_affinities_masked_grad,
  gate_up_proj_weight_grad,
  down_proj_weight_grad,
):
  """
  Initialize gradient outputs for blockwise matrix multiplication backward propagation.
  
  Args:
    hidden_states_grad: Gradient tensor for hidden states
    expert_affinities_masked_grad: Gradient tensor for expert affinities
    gate_up_proj_weight_grad: Gradient tensor for gate up projection weights
    down_proj_weight_grad: Gradient tensor for down projection weights
  
  """
  T, H = hidden_states_grad.shape
  E, I_TP, _ = down_proj_weight_grad.shape

  # Initialize hidden states gradients
  for n in nl.affine_range(int(np.ceil(T/TILE_SIZE))):
    zeros = nl.zeros((TILE_SIZE, H), dtype=hidden_states_grad.dtype)
    store_p, store_f = nl.mgrid[0:TILE_SIZE, 0:H]
    nl.store(
      hidden_states_grad[n*TILE_SIZE+store_p, store_f],
      value=zeros[0:TILE_SIZE, 0:H],
      mask=n*TILE_SIZE + store_p < T,
    )

  # Initialize expert affinities gradients
  for n in nl.affine_range(int(np.ceil(T/TILE_SIZE))):
    zeros = nl.zeros((TILE_SIZE, E), dtype=expert_affinities_masked_grad.dtype)
    store_p, store_f = nl.mgrid[0:TILE_SIZE, 0:E]
    nl.store(
      expert_affinities_masked_grad[TILE_SIZE*E*n + E * store_p + store_f, 0],
      value=zeros[0:TILE_SIZE, 0:E],
      mask=n*TILE_SIZE + store_p < T,
    )

  # Initialize gate up projection weight gradients
  for i in nl.affine_range(E):
    for n in nl.affine_range(int(np.ceil(H/TILE_SIZE))):
      zeros = nl.zeros((TILE_SIZE, 2, I_TP), dtype=gate_up_proj_weight_grad.dtype)
      store_p, store_f1, store_f2 = nl.mgrid[0:TILE_SIZE, 0:2, 0:I_TP]
      nl.store(
        gate_up_proj_weight_grad[
            i,
            n*TILE_SIZE + store_p,
            store_f1,
            store_f2,
        ],
        value=zeros,
        mask=n*TILE_SIZE + store_p < H,
      )
  # Initialize down projection weight gradients
  for i in nl.affine_range(E):
    for n in nl.affine_range(int(np.ceil(T/I_TP))):
      zeros = nl.zeros((TILE_SIZE, H), dtype=down_proj_weight_grad.dtype)
      store_p, store_h = nl.mgrid[0:TILE_SIZE, 0:H]
      nl.store(
        down_proj_weight_grad[i, TILE_SIZE*n + store_p, store_h],
        value=zeros[0:TILE_SIZE, 0:H],
        mask=(store_p + TILE_SIZE*n < I_TP),
      )


def initialize_gradient_outputs_shard(
  hidden_states_grad,
  expert_affinities_masked_grad,
  gate_up_proj_weight_grad,
  down_proj_weight_grad,
  num_shards,
  shard_id,
):
  """
  Initialize gradient outputs for blockwise matrix multiplication backward propagation with LNC sharding
  
  Args:
    hidden_states_grad: Gradient tensor for hidden states
    expert_affinities_masked_grad: Gradient tensor for expert affinities
    gate_up_proj_weight_grad: Gradient tensor for gate up projection weights
    down_proj_weight_grad: Gradient tensor for down projection weights
  
  """
  T, H = hidden_states_grad.shape
  E, I_TP, _ = down_proj_weight_grad.shape

  shard_E = E // num_shards
  e_offset = shard_E * shard_id

  # FIXME: support shard T
  for n in nl.affine_range(int(np.ceil(T/TILE_SIZE))):
    zeros = nl.zeros((TILE_SIZE, shard_E), dtype=expert_affinities_masked_grad.dtype)
    nl.store(expert_affinities_masked_grad[128*E*n + E * nl.arange(TILE_SIZE)[:, None] + e_offset + nl.arange(shard_E)[None, :], 0], value=zeros[0:TILE_SIZE, 0:shard_E], mask=n*TILE_SIZE + nl.arange(TILE_SIZE)[:, None] < T)

  # shard over H
  H_per_shard = H // num_shards
  assert H % num_shards == 0, f"Expect hidden dim is shardable by {num_shards}"
  h_offset = H_per_shard * shard_id

  for i in nl.affine_range(E):
    for n in nl.affine_range(int(np.ceil(H_per_shard/TILE_SIZE))):
      zeros = nl.zeros((TILE_SIZE, 2, I_TP), dtype=gate_up_proj_weight_grad.dtype)
      store_p, store_f1, store_f2 = nl.mgrid[0:TILE_SIZE, 0:2, 0:I_TP]
      nl.store(gate_up_proj_weight_grad[
        i, 
        TILE_SIZE * n + store_p + h_offset, 
        store_f1,
        store_f2
      ], value=zeros[store_p, store_f1, store_f2], mask=n*TILE_SIZE + store_p + h_offset < H)

  for i in nl.affine_range(E):
    for n in nl.affine_range(int(np.ceil(T/I_TP))):
      zeros = nl.zeros((TILE_SIZE, H_per_shard), dtype=down_proj_weight_grad.dtype)
      store_p, store_f = nl.mgrid[0:TILE_SIZE, 0:H_per_shard]
      nl.store(down_proj_weight_grad[i, 128*n + store_p, store_f + h_offset], value=zeros[store_p, store_f], 
              mask=(store_p + TILE_SIZE * n < I_TP))

  # Initialize blockwise bwd outputs (gradients)
  for n in nl.affine_range(int(np.ceil(T/TILE_SIZE))):
    zeros = nl.zeros((TILE_SIZE, H_per_shard), dtype=hidden_states_grad.dtype)
    nl.store(hidden_states_grad[n*TILE_SIZE+nl.arange(TILE_SIZE)[:, None], nl.arange(H_per_shard)[None, :]+h_offset], 
             value=zeros[0:TILE_SIZE, 0:H_per_shard], 
             mask=n*TILE_SIZE + nl.arange(TILE_SIZE)[:, None] < T)

def compute_expert_affinity(
  token_indices,
  expert_idx,
  expert_affinities_masked,
  NUM_TILES,
  E,
  dtype,
):
  """Compute expert affinity scores for each token.

  Args:
    token_indices: Token indices in the block
    expert_idx: Index of the current expert
    expert_affinities_masked: Masked expert affinity scores
    NUM_TILES: Number of tiles
    E (int): Expert dimension size

  Returns:
    expert_affinity_f32: Expert affinity scores in float32
  """
  # Initialize output buffer for expert affinity scores in float32
  expert_affinity_f32 = nl.ndarray(
    (NUM_TILES, nl.par_dim(TILE_SIZE), 1),
    dtype=nl.float32,
    buffer=nl.sbuf,
  )

  for n in nl.affine_range(NUM_TILES):
    # Calculate address offsets for expert affinity lookup
    addr = nl.multiply(token_indices[0:TILE_SIZE, n], E, dtype=np.int32)
    addr_fin = nl.add(addr[0:TILE_SIZE, 0], expert_idx)

    # Temporary buffer for loading expert affinities in bfloat16
    expert_affinity_dtype = nl.ndarray(
      (TILE_SIZE, 1), dtype=dtype, buffer=nl.sbuf
    )

    # Load expert affinities using computed addresses
    expert_affinity_dtype[0:TILE_SIZE, 0] = nl.load(
      expert_affinities_masked[
        addr_fin[nl.arange(TILE_SIZE)[:, None], 0],
        nl.arange(1)[None, :],
      ],
      dtype=dtype,
    )

    # TensorScalarPtr operations require F32 data type on ptr
    expert_affinity_f32[n, 0:TILE_SIZE, 0] = nl.copy(
      expert_affinity_dtype[0:TILE_SIZE, 0],
      dtype=np.float32,
    )

  return expert_affinity_f32


def compute_expert_affinity_gradient(
  output_hidden_states_grad,
  down_proj_act_checkpoint,
  token_indices,
  expert_affinities_masked_grad,
  expert_idx,
  block_idx,
  NUM_TILES,
  E,
  dtype,
):
  """Compute expert affinity gradients using blockwise matrix multiplication and reduction.

  Args:
    output_hidden_states_grad: Gradient of output hidden states
    down_proj_act_checkpoint: Checkpoint for down projection activation
    token_indices: Token indices
    expert_affinities_masked_grad: Gradient of masked expert affinities
    expert_idx: Expert index
    block_idx: Block index
    NUM_TILES (int): Number of tiles
    E (int): Expert dimension size

  Returns:
    local_output_hidden_states_grad
  """
  _, H = output_hidden_states_grad.shape
  # Initialize local output hidden states gradient
  local_output_hidden_states_grad = nl.ndarray(
    (NUM_TILES, nl.par_dim(TILE_SIZE), H), dtype=dtype, buffer=nl.sbuf
  )

  # Load output hidden states gradients
  for ti in nl.affine_range(NUM_TILES):
    local_output_hidden_states_grad[ti, 0:TILE_SIZE, 0:H] = nl.load(
      output_hidden_states_grad[
        token_indices[nl.arange(TILE_SIZE)[:, None], ti], nl.arange(H)[None, :]
      ],
      dtype=dtype,
      mask=None,
    )

  hidden_outer_tripcount = int(np.ceil(H / PSUM_SIZE))

  # Initialize arrays for computation
  local_down_proj_act_checkpoint = nl.ndarray(
    (TILE_SIZE, nl.par_dim(TILE_SIZE), H), dtype=dtype, buffer=nl.sbuf
  )
  local_expert_affinity_grad = nl.ndarray(
    (NUM_TILES, nl.par_dim(TILE_SIZE), 1), dtype=dtype, buffer=nl.sbuf
  )
  local_reduce_tmp = nl.ndarray(
    (NUM_TILES, hidden_outer_tripcount, nl.par_dim(TILE_SIZE), 1),
    dtype=np.float32,
    buffer=nl.sbuf,
  )
  mul_2_tmp = nl.ndarray(
    (NUM_TILES, hidden_outer_tripcount, nl.par_dim(TILE_SIZE), PSUM_SIZE),
    dtype=np.float32,
    buffer=nl.sbuf,
  )

  # Compute gradients
  for ti in nl.affine_range(NUM_TILES):
    # Load checkpoint data
    local_down_proj_act_checkpoint[
      ti, nl.arange(TILE_SIZE)[:, None], nl.arange(H)[None, :]
    ] = nl.load(
      down_proj_act_checkpoint[
        block_idx,
        TILE_SIZE * ti + nl.arange(TILE_SIZE)[:, None],
        nl.arange(H)[None, :],
      ],
      dtype=dtype,
      mask=None,
    )

    # Compute gradients for each hidden dimension block
    for hi in nl.affine_range(hidden_outer_tripcount):
      # Multiply gradients
      mul_2_tmp[ti, hi, nl.arange(TILE_SIZE)[:, None], nl.arange(PSUM_SIZE)[None, :]] = nl.multiply(
        local_output_hidden_states_grad[
          ti,
          nl.arange(TILE_SIZE)[:, None],
          nl.arange(PSUM_SIZE)[None, :] + PSUM_SIZE * hi,
        ],
        local_down_proj_act_checkpoint[
          ti,
          nl.arange(TILE_SIZE)[:, None],
          nl.arange(PSUM_SIZE)[None, :] + PSUM_SIZE * hi,
        ],
        mask=None,
        dtype=np.float32,
      )

      # Reduce along specified axis
      local_reduce_tmp[ti, hi, nl.arange(TILE_SIZE)[:, None], 0] = nisa.tensor_reduce(
        np.add,
        data=mul_2_tmp[
          ti, hi, nl.arange(TILE_SIZE)[:, None], nl.arange(PSUM_SIZE)[None, :]
        ],
        mask=None,
        axis=[1],
        dtype=np.float32,
        negate=False,
      )

      # Compute final gradient
      local_expert_affinity_grad[ti, nl.arange(TILE_SIZE)[:, None], 0] = nl.loop_reduce(
        local_reduce_tmp[ti, hi, nl.arange(TILE_SIZE)[:, None], 0],
        op=np.add,
        loop_indices=[hi],
        mask=None,
        dtype=dtype,
      )

  # Store gradients
  for n in nl.affine_range(NUM_TILES):
    addr = nl.multiply(token_indices[0:TILE_SIZE, n], E, dtype=np.int32)
    addr_fin = nl.add(addr[0:TILE_SIZE, 0], expert_idx)
    nl.store(
      expert_affinities_masked_grad[
        addr_fin[nl.arange(TILE_SIZE)[:, None], 0], nl.arange(1)[None, :]
      ],
      value=local_expert_affinity_grad[n, 0:TILE_SIZE, 0],
    )

  return local_output_hidden_states_grad

def compute_expert_affinity_gradient_shard(
  output_hidden_states_grad,
  down_proj_act_checkpoint,
  token_indices,
  expert_affinities_masked_grad,
  expert_idx,
  block_idx,
  NUM_TILES,
  E,
  num_shards,
  shard_id,
  dtype,
):
  """Compute expert affinity gradients using blockwise matrix multiplication and reduction.

  Args:
    output_hidden_states_grad: Gradient of output hidden states
    down_proj_act_checkpoint: Checkpoint for down projection activation
    token_indices: Token indices
    expert_affinities_masked_grad: Gradient of masked expert affinities
    expert_idx: Expert index
    block_idx: Block index
    NUM_TILES (int): Number of tiles
    E (int): Expert dimension size
    num_shards: Number of shards
    shard_id: Shard ID

  Returns:
    local_output_hidden_states_grad
  """
  _, H = output_hidden_states_grad.shape
  H_per_shard = H // num_shards
  h_offset = H_per_shard * shard_id
  # Initialize local output hidden states gradient
  local_output_hidden_states_grad = nl.ndarray(
    (NUM_TILES, nl.par_dim(TILE_SIZE), H), dtype=dtype, buffer=nl.sbuf
  )

  # Load output hidden states gradients
  for ti in nl.affine_range(NUM_TILES):
    local_output_hidden_states_grad[ti, 0:TILE_SIZE, 0:H_per_shard] = nl.load(
      output_hidden_states_grad[
        token_indices[nl.arange(TILE_SIZE)[:, None], ti], nl.arange(H_per_shard)[None, :]+h_offset
      ],
      dtype=dtype,
      mask=None,
    )

  hidden_outer_tripcount = int(np.ceil(H_per_shard / PSUM_SIZE))

  # Initialize arrays for computation
  local_down_proj_act_checkpoint = nl.ndarray(
    (TILE_SIZE, nl.par_dim(TILE_SIZE), H_per_shard), dtype=dtype, buffer=nl.sbuf
  )
  local_expert_affinity_grad = nl.ndarray(
    (NUM_TILES, nl.par_dim(TILE_SIZE), 1), dtype=dtype, buffer=nl.sbuf
  )
  local_reduce_tmp = nl.ndarray(
    (NUM_TILES, hidden_outer_tripcount, nl.par_dim(TILE_SIZE), 1),
    dtype=np.float32,
    buffer=nl.sbuf,
  )
  mul_2_tmp = nl.ndarray(
    (NUM_TILES, hidden_outer_tripcount, nl.par_dim(TILE_SIZE), PSUM_SIZE),
    dtype=np.float32,
    buffer=nl.sbuf,
  )
  local_expert_affinity_grad_reduce = nl.ndarray(
    (NUM_TILES, nl.par_dim(TILE_SIZE), 1), 
    dtype=dtype, 
    name="reduce.lnc.1", buffer=nl.sbuf)
  
  # Compute gradients
  for ti in nl.affine_range(NUM_TILES):
    # Load checkpoint data
    local_down_proj_act_checkpoint[
      ti, nl.arange(TILE_SIZE)[:, None], nl.arange(H_per_shard)[None, :]
    ] = nl.load(
      down_proj_act_checkpoint[
        block_idx,
        TILE_SIZE * ti + nl.arange(TILE_SIZE)[:, None],
        nl.arange(H_per_shard)[None, :]+h_offset,
      ],
      dtype=dtype,
      mask=None,
    )

    # Compute gradients for each hidden dimension block
    for hi in nl.affine_range(hidden_outer_tripcount):
      # Multiply gradients
      mul_2_tmp[ti, hi, nl.arange(TILE_SIZE)[:, None], nl.arange(PSUM_SIZE)[None, :]] = nl.multiply(
        local_output_hidden_states_grad[
          ti,
          nl.arange(TILE_SIZE)[:, None],
          nl.arange(PSUM_SIZE)[None, :] + PSUM_SIZE * hi,
        ],
        local_down_proj_act_checkpoint[
          ti,
          nl.arange(TILE_SIZE)[:, None],
          nl.arange(PSUM_SIZE)[None, :] + PSUM_SIZE * hi,
        ],
        mask=None,
        dtype=np.float32,
      )

      # Reduce along specified axis
      local_reduce_tmp[ti, hi, nl.arange(TILE_SIZE)[:, None], 0] = nisa.tensor_reduce(
        np.add,
        data=mul_2_tmp[
          ti, hi, nl.arange(TILE_SIZE)[:, None], nl.arange(PSUM_SIZE)[None, :]
        ],
        mask=None,
        axis=[1],
        dtype=np.float32,
        negate=False,
      )

      # Compute final gradient
      local_expert_affinity_grad[ti, nl.arange(TILE_SIZE)[:, None], 0] = nl.loop_reduce(
        local_reduce_tmp[ti, hi, nl.arange(TILE_SIZE)[:, None], 0],
        op=np.add,
        loop_indices=[hi],
        mask=None,
        dtype=dtype,
      )

    local_expert_affinity_grad_reduce[ti, nl.arange(TILE_SIZE)[:, None], 0] = nl.all_reduce(
      local_expert_affinity_grad[ti, nl.arange(TILE_SIZE)[:, None], 0], 
      op=np.add, 
      program_axes=[shard_id], 
      mask=None, 
      dtype=dtype
    )

  # Store gradients
  for n in nl.affine_range(NUM_TILES):
    addr = nl.multiply(token_indices[0:TILE_SIZE, n], E, dtype=np.int32)
    addr_fin = nl.add(addr[0:TILE_SIZE, 0], expert_idx)
    nl.store(
      expert_affinities_masked_grad[
        addr_fin[nl.arange(TILE_SIZE)[:, None], 0], nl.arange(1)[None, :]
      ],
      value=local_expert_affinity_grad_reduce[n, 0:TILE_SIZE, 0],
    )

  return local_output_hidden_states_grad

def compute_down_proj_mm_gradient_and_transpose(
  local_output_hidden_states_grad,
  expert_affinity_f32,
  H,
  NUM_TILES,
  dtype,
):
  """Compute gradient of down projection result and perform transpose operation.

  Args:
    local_output_hidden_states_grad: Local gradient of output hidden states
    expert_affinity_f32: Expert affinity in float32
    H (int): tripcount for hidden dimension
    NUM_TILES (int): Number of tiles

  Returns:
    tuple: (multiply_1, multiply_1_transpose) containing the computed gradients
      and their transposed version
  """
  hidden_outer_tripcount = int(np.ceil(H / PSUM_SIZE))

  # Initialize arrays for computation
  multiply_1 = nl.ndarray(
    (NUM_TILES, hidden_outer_tripcount, nl.par_dim(TILE_SIZE), 4, TILE_SIZE),
    dtype=dtype,
    buffer=nl.sbuf,
  )

  multiply_1_transpose = nl.ndarray(
    (hidden_outer_tripcount, 4, nl.par_dim(TILE_SIZE), TILE_SIZE * NUM_TILES),
    dtype=dtype,
    buffer=nl.sbuf,
  )

  # Temporary array for local tensor size reduction
  mul_1_tmp = nl.ndarray(
    (NUM_TILES, nl.par_dim(TILE_SIZE), hidden_outer_tripcount, 4, TILE_SIZE),
    dtype=dtype,
    buffer=nl.sbuf,
  )

  # Compute gradients and perform transpose
  for ti in nl.affine_range(NUM_TILES):
    for hi in nl.affine_range(hidden_outer_tripcount):
      mul_1_tmp[
        ti,
        nl.arange(TILE_SIZE)[:, None, None],
        hi,
        nl.arange(4)[None, :, None],
        nl.arange(128)[None, None, :],
      ] = nl.multiply(
        local_output_hidden_states_grad[
          ti,
          nl.arange(TILE_SIZE)[:, None, None],
          128 * nl.arange(4)[None, :, None]
          + 512 * hi
          + nl.arange(128)[None, None, :],
        ],
        expert_affinity_f32[ti, nl.arange(TILE_SIZE)[:, None, None], 0],
        mask=None,
        dtype=dtype,
      )  # TODO: add mask for H

      multiply_1[
        ti,
        hi,
        nl.arange(TILE_SIZE)[:, None, None],
        nl.arange(4)[None, :, None],
        nl.arange(128)[None, None, :],
      ] = nl.copy(
        mul_1_tmp[
          ti,
          nl.arange(TILE_SIZE)[:, None, None],
          hi,
          nl.arange(4)[None, :, None],
          nl.arange(128)[None, None, :],
        ],
        dtype=dtype,
        mask=None,
      )

      # TODO: explore delaying the transpose operation
      for i10 in nl.affine_range(4):
        multiply_1_transpose[
          hi,
          i10,
          nl.arange(TILE_SIZE)[:, None],
          nl.arange(128)[None, :] + 128 * ti,
        ] = nisa.nc_transpose(
          multiply_1[
            ti, hi, nl.arange(TILE_SIZE)[:, None], i10, nl.arange(128)[None, :]
          ],
          dtype=dtype,
          mask=None,
        )

  return multiply_1, multiply_1_transpose

def recompute_silu_activation(gate_up_proj_act_checkpoint_T, block_idx, dtype):
  """Compute SILU activation for gate and up projections.

  Args:
    gate_up_proj_act_checkpoint_T: Input tensor for gate/up projections
    block_idx: Block index

  Returns:
    tuple: (local_gate_up_proj_act_checkpoint, silu_up_proj)
  """
  N, _, I_TP, B = gate_up_proj_act_checkpoint_T.shape

  NUM_TILES = B // TILE_SIZE
  gup_n_tile = int(np.ceil(I_TP / 128))
  local_gate_up_proj_act_checkpoint = nl.ndarray(
    (gup_n_tile, 2, nl.par_dim(TILE_SIZE), B),
    dtype=dtype,
    name="gate_up_proj_act_checkpoint_local",
    buffer=nl.sbuf,
  )

  silu_up_proj = nl.ndarray(
    (gup_n_tile, nl.par_dim(TILE_SIZE), B),
    dtype=dtype,
    name="fwd_silu_up_proj",
    buffer=nl.sbuf,
  )

  # load gate_up_proj_act_checkpoint and recompute silu
  for gi in nl.affine_range(gup_n_tile):
    for gate_or_up in nl.affine_range(2):
      for ti in nl.affine_range(NUM_TILES):
        p_dim, f_dim = nl.mgrid[0:TILE_SIZE, 0:128]
        local_gate_up_proj_act_checkpoint[
          gi, gate_or_up, p_dim, 128 * ti + f_dim
        ] = nl.load(
          gate_up_proj_act_checkpoint_T[
            block_idx,
            gate_or_up,
            128 * gi + nl.arange(128)[:, None],
            128 * ti + nl.arange(TILE_SIZE)[None, :],
          ],
          dtype=dtype,
          mask=(-1 * nl.arange(128)[:, None] + -128 * gi + I_TP - 1 >= 0),
        )

    # recompute silu for up proj in index 0
    p_load, b_load = nl.mgrid[0:TILE_SIZE, 0:B]
    silu_up_proj[gi, p_load, b_load] = nisa.activation(
      op=nl.silu,
      data=local_gate_up_proj_act_checkpoint[gi, 0, p_load, b_load],
      bias=None,
      scale=1.0,
      mask=(-1 * p_load + -128 * gi + I_TP - 1 >= 0),
      dtype=dtype,
    )

  return local_gate_up_proj_act_checkpoint, silu_up_proj


def load_and_transpose_down_proj_weights(down_proj_weight, expert_idx, dtype):
  """Load weights and perform transpose operations for down projection.

  Args:
    down_proj_weight: Original weight tensor
    expert_idx: Expert index

  Returns:
    down_proj_w_pftranspose
  """
  _, I_TP, H = down_proj_weight.shape
  gup_n_tile = int(np.ceil(I_TP / 128))
  hidden_outer_tripcount = int(np.ceil(H / PSUM_SIZE))

  # Initialize weight buffer
  dp_weights_sbuf = nl.ndarray(
    (gup_n_tile, nl.par_dim(TILE_SIZE), H),
    dtype=dtype,
    buffer=nl.sbuf,
  )

  # Load weights
  for gi in nl.affine_range(gup_n_tile):
    p_dim, f_dim = nl.mgrid[0:TILE_SIZE, 0:H]
    dp_weights_sbuf[gi, p_dim, f_dim] = nl.load(
      down_proj_weight[expert_idx[0, 0], p_dim + TILE_SIZE * gi, f_dim],
      dtype=dtype,
      mask=(p_dim + TILE_SIZE * gi < I_TP),
    )

  # Initialize transposed weight buffer
  down_proj_w_pftranspose = nl.ndarray(
    (gup_n_tile, hidden_outer_tripcount, 4, nl.par_dim(TILE_SIZE), 128),
    dtype=dtype,
    buffer=nl.sbuf,
  )

  # Perform transpose operations
  for gi in nl.affine_range(gup_n_tile):
    for hi in nl.affine_range(hidden_outer_tripcount):
      for ii in nl.affine_range(4):
        p_dim, f_dim = nl.mgrid[0:TILE_SIZE, 0:128]
        down_proj_w_pftranspose[gi, hi, ii, p_dim, f_dim] = nisa.nc_transpose(
          dp_weights_sbuf[gi, p_dim, f_dim + 128 * ii + 512 * hi],
          dtype=dtype,
          mask=(p_dim + TILE_SIZE * gi < I_TP),
        )

  return down_proj_w_pftranspose

def load_and_transpose_down_proj_weights_shard(
  down_proj_weight, 
  expert_idx,
  num_shards,
  shard_id,
  dtype,
):
  """Load weights and perform transpose operations for down projection.

  Args:
    down_proj_weight: Original weight tensor
    expert_idx: Expert index
    num_shards: Number of shards
    shard_id: Shard ID

  Returns:
    down_proj_w_pftranspose
  """
  _, I_TP, H = down_proj_weight.shape
  H_per_shard = H // num_shards
  h_offset = H_per_shard * shard_id

  gup_n_tile = int(np.ceil(I_TP / 128))
  hidden_outer_tripcount = int(np.ceil(H_per_shard / PSUM_SIZE))

  # Initialize weight buffer
  dp_weights_sbuf = nl.ndarray(
    (gup_n_tile, nl.par_dim(TILE_SIZE), H_per_shard),
    dtype=dtype,
    buffer=nl.sbuf,
  )

  # Load weights
  for gi in nl.affine_range(gup_n_tile):
    p_dim, f_dim = nl.mgrid[0:TILE_SIZE, 0:H_per_shard]
    dp_weights_sbuf[gi, p_dim, f_dim] = nl.load(
      down_proj_weight[expert_idx[0, 0], p_dim + TILE_SIZE * gi, f_dim + h_offset],
      dtype=dtype,
      mask=(p_dim + TILE_SIZE * gi < I_TP),
    )

  # Initialize transposed weight buffer
  down_proj_w_pftranspose = nl.ndarray(
    (gup_n_tile, hidden_outer_tripcount, 4, nl.par_dim(TILE_SIZE), 128),
    dtype=dtype,
    buffer=nl.sbuf,
  )

  # Perform transpose operations
  for gi in nl.affine_range(gup_n_tile):
    for hi in nl.affine_range(hidden_outer_tripcount):
      for ii in nl.affine_range(4):
        p_dim, f_dim = nl.mgrid[0:TILE_SIZE, 0:128]
        down_proj_w_pftranspose[gi, hi, ii, p_dim, f_dim] = nisa.nc_transpose(
          dp_weights_sbuf[gi, p_dim, f_dim + 128 * ii + 512 * hi],
          dtype=dtype,
          mask=(p_dim + TILE_SIZE * gi < I_TP),
        )

  return down_proj_w_pftranspose

def compute_silu_gradient(
  multiply_1_grad_sbuf, 
  silu_up_proj, 
  local_gate_up_proj_act_checkpoint, 
  B, 
  I_TP,
  dtype):
  """
  Compute gradient of down projection input using matrix multiplication and reduction.
  
  Args:
    multiply_1_grad_sbuf: 
    silu_up_proj: 
    local_gate_up_proj_act_checkpoint:
    B: block size
    I_TP: intermediate size
  
  Returns:
    tuple: (silu_up_mul_gate_grad, silu_up_mul_gate_grad_trans, silu_up_mul_gate_trans)
  """  
  N_PSUM_TILE = int(np.ceil(B / PSUM_SIZE))
  NUM_TILES = int(np.ceil(B / TILE_SIZE))
  gup_n_tile = int(np.ceil(I_TP / TILE_SIZE))

  # here innermost loop dimension (blocksize) is 512
  silu_up_mul_gate = nl.ndarray(
    (gup_n_tile, nl.par_dim(TILE_SIZE), B), dtype=dtype, buffer=nl.sbuf
  )
  gate_proj_grad = nl.ndarray(
    (gup_n_tile, nl.par_dim(TILE_SIZE), B), dtype=dtype, buffer=nl.sbuf
  )

  silu_up_proj_grad = nl.ndarray(
    (gup_n_tile, nl.par_dim(TILE_SIZE), B), dtype=dtype, buffer=nl.sbuf
  )

  silu_dx_1 = nl.ndarray(
    (gup_n_tile, nl.par_dim(TILE_SIZE), B), dtype=dtype, buffer=nl.sbuf
  )

  # silu of graident of up_projection and gate_projection
  silu_up_mul_gate_grad = nl.ndarray(
    (gup_n_tile, 2, nl.par_dim(TILE_SIZE), B), dtype=dtype, buffer=nl.sbuf
  )

  silu_up_mul_gate_grad_trans = nl.ndarray(
    (NUM_TILES, 2, nl.par_dim(TILE_SIZE), I_TP),
    dtype=dtype,
    buffer=nl.sbuf,
  )
  # 512 is I_TP
  silu_up_mul_gate_trans = nl.ndarray(
    (NUM_TILES, nl.par_dim(TILE_SIZE), I_TP),
    dtype=dtype,
    buffer=nl.sbuf,
  )

  for gi in nl.affine_range(gup_n_tile):

    # recompute forward silu(up_proj) * gate_proj_res
    p_dim, b_dim = nl.mgrid[0:TILE_SIZE, 0:B]
    silu_up_mul_gate[gi, p_dim, b_dim] = nl.multiply(
      silu_up_proj[gi, p_dim, b_dim],
      local_gate_up_proj_act_checkpoint[gi, 1, p_dim, b_dim],
      mask=(-1 * p_dim + -128 * gi + I_TP - 1 >= 0),
      dtype=dtype,
    )
      
    # down_proj_gradient * silu(up_proj) => gradient of gate_proj_res
    gate_proj_grad[
      gi, nl.arange(TILE_SIZE)[:, None], nl.arange(B)[None, :]
    ] = nl.multiply(
      multiply_1_grad_sbuf[
        gi, nl.arange(TILE_SIZE)[:, None], nl.arange(B)[None, :]
      ],
      silu_up_proj[gi, nl.arange(TILE_SIZE)[:, None], nl.arange(B)[None, :]],
      mask=(-1 * nl.arange(TILE_SIZE)[:, None] + -128 * gi + I_TP - 1 >= 0),
      dtype=dtype,
    )

    # gradient of gate_proj_res
    p_dim, b_dim = nl.mgrid[0:TILE_SIZE, 0:B]
    silu_up_mul_gate_grad[gi, 1, p_dim, b_dim] = nl.copy(
      gate_proj_grad[gi, p_dim, b_dim],
      dtype=dtype,
      mask=(-1 * p_dim + -128 * gi + I_TP - 1 >= 0),
    )

    # following instructions are used to compute silu_dx
    # TODO: try silu_dx activation func
    # silu_up_proj_grad = down_proj_gradient * gate_proj_res => gradient of silu(up_proj)
    silu_up_proj_grad[
      gi, nl.arange(TILE_SIZE)[:, None], nl.arange(B)[None, :]
    ] = nl.multiply(
      multiply_1_grad_sbuf[
        gi, nl.arange(TILE_SIZE)[:, None], nl.arange(B)[None, :]
      ],
      local_gate_up_proj_act_checkpoint[
        gi, 1, nl.arange(TILE_SIZE)[:, None], nl.arange(B)[None, :]
      ],
      mask=(-1 * nl.arange(TILE_SIZE)[:, None] + -128 * gi + I_TP - 1 >= 0),
      dtype=dtype,
    )

    silu_dx_1[
      gi, nl.arange(TILE_SIZE)[:, None], nl.arange(B)[None, :]
    ] = nisa.activation(
      op=nl.silu_dx,
      data=local_gate_up_proj_act_checkpoint[
        gi, 0, nl.arange(TILE_SIZE)[:, None], nl.arange(B)[None, :]
      ],
      bias=None,
      scale=1.0,
      mask=(-128 * gi + -1 * nl.arange(TILE_SIZE)[:, None] + I_TP - 1 >= 0),
      dtype=dtype,
    )

    # silu_up_proj_grad(grad) * silu_dx(up_proj_res)
    # silu_dx(up_proj_res) = sigmoid(up_proj_res) * (1 + up_proj_res * (1 - sigmoid(up_proj_res)))
    # gradient of silu(up_proj_res)
    silu_up_mul_gate_grad[
      gi, 0, nl.arange(TILE_SIZE)[:, None], nl.arange(B)[None, :]
    ] = nl.multiply(
      silu_up_proj_grad[gi, nl.arange(TILE_SIZE)[:, None], nl.arange(B)[None, :]],
      silu_dx_1[gi, nl.arange(TILE_SIZE)[:, None], nl.arange(B)[None, :]],
      mask=(-1 * nl.arange(TILE_SIZE)[:, None] + -128 * gi + I_TP - 1 >= 0),
      dtype=dtype,
    )

  for gui in nl.affine_range(2):  # gate or up
    for ti in nl.affine_range(NUM_TILES):
      for ii in nl.affine_range(gup_n_tile):
        p_dim, f_dim = nl.mgrid[0:TILE_SIZE, 0:128]
        silu_up_mul_gate_grad_trans[ti, gui, p_dim, 128 * ii + f_dim] = nisa.nc_transpose(
          silu_up_mul_gate_grad[ii, gui, p_dim, f_dim + 128 * ti],
          dtype=dtype,
          mask=(TILE_SIZE * ii + p_dim < I_TP),
        )

  # silu_up_mul_gate_trans is transpose multiply(silu(up_proj), gate_proj) recompute
  for ti in nl.affine_range(NUM_TILES):
    for ii in nl.affine_range(gup_n_tile):
      p_dim, f_dim = nl.mgrid[0:TILE_SIZE, 0:128]
      silu_up_mul_gate_trans[ti, p_dim, f_dim + 128 * ii] = nisa.nc_transpose(
        silu_up_mul_gate[ii, p_dim, f_dim + 128 * ti],
        dtype=dtype,
        mask=(TILE_SIZE * ii + p_dim < I_TP),
      )

  return (
    silu_up_mul_gate_grad,
    silu_up_mul_gate_grad_trans,
    silu_up_mul_gate_trans,
  )

def compute_multiply_1_gradient(down_proj_w_pftranspose, multiply_1_transpose,
                                B, I_TP, H, dtype):
  """
  Compute gradient of down projection input using matrix multiplication and reduction.
  
  Args:
    down_proj_w_pftranspose: Transposed down projection weights
    multiply_1_transpose: Transposed multiplication results
    silu_up_proj:
    local_gate_up_proj_act_checkpoint
    B: block size
    I_TP: intermediate size
    H: hidden size
  
  Returns:
    tuple: (silu_up_mul_gate_grad, silu_up_mul_gate_grad_trans, silu_up_mul_gate_trans)
  """

  N_PSUM_TILE = int(np.ceil(B / PSUM_SIZE))
  gup_n_tile = int(np.ceil(I_TP / TILE_SIZE))
  hidden_outer_tripcount = int(np.ceil(H / PSUM_SIZE))

  multiply_1_grad_sbuf = nl.ndarray(
    (gup_n_tile, nl.par_dim(TILE_SIZE), B), dtype=dtype, buffer=nl.sbuf
  )

  # step 3.2: compute gradient of down proj input (multiplication result of gate proj + silu(up proj)) 
  # shape: [B, I_TP] = [B, H] @ [I_TP, H](down_proj_weight)
  multiply_1_grad_psum = nl.zeros(
    (N_PSUM_TILE, 4, nl.par_dim(TILE_SIZE), PSUM_SIZE),
    dtype=np.float32,
    buffer=nl.psum,
    lazy_initialization=True,
  )

  for gi in nl.affine_range(gup_n_tile):
    for b_i in nl.affine_range(N_PSUM_TILE):
      for hi in nl.affine_range(hidden_outer_tripcount):
        for di in nl.affine_range(4):
          multiply_1_grad_psum[
            b_i, gi, nl.arange(TILE_SIZE)[:, None], nl.arange(PSUM_SIZE)[None, :]
          ] += nisa.nc_matmul(
            down_proj_w_pftranspose[
              gi,
              hi,
              di,
              nl.arange(TILE_SIZE)[:, None],
              nl.arange(128)[None, :],
            ][-1 * nl.arange(128)[None, :] + -128 * gi + I_TP - 1 >= 0],
            multiply_1_transpose[
              hi,
              di,
              nl.arange(TILE_SIZE)[:, None],
              nl.arange(PSUM_SIZE)[None, :] + b_i * PSUM_SIZE,
            ],
            is_stationary_onezero=False,
            is_moving_onezero=False,
            mask=None,
          )
  
    for b_i in nl.affine_range(N_PSUM_TILE):
      p_dim, f_dim = nl.mgrid[0:TILE_SIZE, 0:PSUM_SIZE]
      multiply_1_grad_sbuf[gi, p_dim, f_dim + b_i * PSUM_SIZE] = nl.copy(
        multiply_1_grad_psum[b_i, gi, p_dim, f_dim],
        dtype=dtype,
        mask=(-1 * p_dim + -128 * gi + I_TP - 1 >= 0),
      )
  # compute gradient in multiply of gate proj * silu(up proj)
  return multiply_1_grad_sbuf

def compute_multiply_1_gradient_shard(
  down_proj_w_pftranspose, 
  multiply_1_transpose,
  B, 
  I_TP, 
  H,
  num_shards,
  shard_id,
  dtype):
  """
  Compute gradient of down projection input using matrix multiplication and reduction.
  
  Args:
    down_proj_w_pftranspose: Transposed down projection weights
    multiply_1_transpose: Transposed multiplication results
    silu_up_proj:
    local_gate_up_proj_act_checkpoint
    B: block size
    I_TP: intermediate size
    H: hidden size
  
  Returns:
    tuple: (silu_up_mul_gate_grad, silu_up_mul_gate_grad_trans, silu_up_mul_gate_trans)
  """

  N_PSUM_TILE = int(np.ceil(B / PSUM_SIZE))
  gup_n_tile = int(np.ceil(I_TP / TILE_SIZE))
  H_per_shard = H // num_shards
  hidden_outer_tripcount = int(np.ceil(H_per_shard / PSUM_SIZE))

  multiply_1_grad_sbuf = nl.ndarray(
    (gup_n_tile, nl.par_dim(TILE_SIZE), B), dtype=dtype, buffer=nl.sbuf
  )
  multiply_1_grad_sbuf_reduce = nl.ndarray(
    (gup_n_tile, nl.par_dim(TILE_SIZE), B), dtype=dtype, name="multiply_1_grad_sbuf_copy", buffer=nl.sbuf)

  # step 3.2: compute gradient of down proj input (multiplication result of gate proj + silu(up proj)) 
  # shape: [512, I_TP] = [512, H] @ [I_TP, H](down_proj_weight)
  multiply_1_grad_psum = nl.zeros(
    (N_PSUM_TILE, 4, nl.par_dim(TILE_SIZE), PSUM_SIZE),
    dtype=np.float32,
    buffer=nl.psum,
    lazy_initialization=True,
  )

  for gi in nl.affine_range(gup_n_tile):
    for b_i in nl.affine_range(N_PSUM_TILE):
      for hi in nl.affine_range(hidden_outer_tripcount):
        for di in nl.affine_range(4):
          multiply_1_grad_psum[
            b_i, gi, nl.arange(TILE_SIZE)[:, None], nl.arange(PSUM_SIZE)[None, :]
          ] += nisa.nc_matmul(
            down_proj_w_pftranspose[
              gi,
              hi,
              di,
              nl.arange(TILE_SIZE)[:, None],
              nl.arange(128)[None, :],
            ][-1 * nl.arange(128)[None, :] + -128 * gi + I_TP - 1 >= 0],
            multiply_1_transpose[
              hi,
              di,
              nl.arange(TILE_SIZE)[:, None],
              nl.arange(PSUM_SIZE)[None, :] + b_i * PSUM_SIZE,
            ],
            is_stationary_onezero=False,
            is_moving_onezero=False,
            mask=None,
          )
      multiply_1_grad_sbuf[gi, nl.arange(TILE_SIZE)[:, None], nl.arange(PSUM_SIZE)[None, :] + b_i*PSUM_SIZE] = nl.copy(
        multiply_1_grad_psum[b_i, gi, nl.arange(TILE_SIZE)[:, None], nl.arange(PSUM_SIZE)[None, :]], 
        dtype=dtype, 
        mask=(-1*nl.arange(TILE_SIZE)[:, None]+-128*gi+I_TP-1 >= 0))
    
    multiply_1_grad_sbuf_reduce[gi, 0:TILE_SIZE, 0:B] = \
        nl.all_reduce(multiply_1_grad_sbuf[gi, 0:TILE_SIZE, 0:B], op=np.add, program_axes=[shard_id], 
                      mask=(-1*nl.arange(TILE_SIZE)[:, None]+-128*gi+I_TP-1 >= 0))

  # compute gradient in multiply of gate proj * silu(up proj)
  return multiply_1_grad_sbuf_reduce



def compute_down_proj_weight_gradients(
    down_proj_weight_grad,
    silu_up_mul_gate_trans,
    multiply_1,
    expert_idx,
    B,
    H,
    I_TP,
    dtype
):
  """
  Compute gradients for down projection weights.
  
  Args:
    down_proj_weight_grad: Gradient tensor for down projection weights
    silu_up_mul_gate_trans: Transposed silu-gate multiplication
    multiply_1: Multiplication tensor
    expert_idx: Expert tensor
    B: block size
    H: Hidden dimension size
    I_TP: Input tensor parameter
  """
  NUM_TILES = int(np.ceil(B / TILE_SIZE))
  gup_n_tile = int(np.ceil(I_TP / 128))
  hidden_outer_tripcount = int(np.ceil(H / PSUM_SIZE))

  # Initialize local gradient tensor
  local_dp_weight_grad = nl.ndarray(
    (gup_n_tile, nl.par_dim(TILE_SIZE), H),
    dtype=dtype,
    buffer=nl.sbuf
  )
  
  # Load initial gradients
  for i_i in nl.affine_range(gup_n_tile):
    p_dim, f_dim = nl.mgrid[0:TILE_SIZE, 0:H]
    local_dp_weight_grad[i_i, p_dim, f_dim] = nl.load(
        down_proj_weight_grad[expert_idx[0, 0], p_dim + TILE_SIZE * i_i, f_dim],
        dtype=dtype,
        mask=(p_dim + TILE_SIZE*i_i < I_TP)
    )

  # Initialize partial sum and accumulation tensors
  down_proj_w_grad_psum = nl.zeros(
    (hidden_outer_tripcount, gup_n_tile, nl.par_dim(TILE_SIZE), PSUM_SIZE),
    dtype=np.float32,
    buffer=nl.psum,
    lazy_initialization=True
  )
  
  down_proj_w_grad_accum = nl.ndarray(
    (gup_n_tile, nl.par_dim(TILE_SIZE), H),
    dtype=dtype,
    buffer=nl.sbuf
  )

  # Compute gradients
  for hi in nl.affine_range(hidden_outer_tripcount):
    for gi in nl.affine_range(gup_n_tile):
      for ti in nl.affine_range(NUM_TILES):
        down_proj_w_grad_psum[
          hi,
          gi,
          nl.arange(TILE_SIZE)[:, None, None],
          TILE_SIZE * nl.arange(4)[None, :, None]
          + nl.arange(TILE_SIZE)[None, None, :],
        ] += nisa.nc_matmul(
          silu_up_mul_gate_trans[
            ti,
            nl.arange(TILE_SIZE)[:, None],
            TILE_SIZE * gi + nl.arange(TILE_SIZE)[None, :],
          ],
          multiply_1[
            ti,
            hi,
            nl.arange(TILE_SIZE)[:, None, None],
            nl.arange(4)[None, :, None],
            nl.arange(TILE_SIZE)[None, None, :],
          ],
          is_stationary_onezero=False,
          is_moving_onezero=False,
          mask=(-1 * nl.arange(TILE_SIZE)[:, None] + -TILE_SIZE * gi + I_TP - 1 >= 0),
        )

      # Accumulate gradients
      down_proj_w_grad_accum[
        gi,
        nl.arange(TILE_SIZE)[:, None],
        PSUM_SIZE * hi + nl.arange(PSUM_SIZE)[None, :],
      ] = nl.add(
        local_dp_weight_grad[
          gi,
          nl.arange(TILE_SIZE)[:, None],
          PSUM_SIZE * hi + nl.arange(PSUM_SIZE)[None, :],
        ],
        down_proj_w_grad_psum[
          hi,
          gi,
          nl.arange(TILE_SIZE)[:, None],
          nl.arange(PSUM_SIZE)[None, :],
        ],
        mask=(-1 * nl.arange(TILE_SIZE)[:, None] + -TILE_SIZE * gi + I_TP - 1 >= 0),
        dtype=dtype,
      )

  # Store final gradients
  for i_i in nl.affine_range(gup_n_tile):
    p_dim, f_dim = nl.mgrid[0:TILE_SIZE, 0:H]
    nl.store(
      down_proj_weight_grad[expert_idx[0, 0], p_dim + TILE_SIZE * i_i, f_dim],
      value=down_proj_w_grad_accum[i_i, nl.arange(TILE_SIZE)[:, None], f_dim],
      mask=(-1 * p_dim + -TILE_SIZE * i_i + I_TP - 1 >= 0),
    )

def compute_down_proj_weight_gradients_shard(
    down_proj_weight_grad,
    silu_up_mul_gate_trans,
    multiply_1,
    expert_idx,
    B,
    H,
    I_TP,
    num_shards,
    shard_id,
    dtype
):
  """
  Compute gradients for down projection weights.
  
  Args:
    down_proj_weight_grad: Gradient tensor for down projection weights
    silu_up_mul_gate_trans: Transposed silu-gate multiplication
    multiply_1: Multiplication tensor
    expert_idx: Expert tensor
    gup_n_tile: Number of gate/up tiles
    hidden_outer_tripcount: Hidden layer outer trip count
    NUM_TILES: Number of tiles
    TILE_SIZE: Size of each tile
    PSUM_SIZE: Size of partial sum
    H: Hidden dimension size
    I_TP: Input tensor parameter
  """
  NUM_TILES = int(np.ceil(B / TILE_SIZE))
  gup_n_tile = int(np.ceil(I_TP / 128))

  H_per_shard = H // num_shards
  h_offset = H_per_shard * shard_id
  hidden_outer_tripcount = int(np.ceil(H_per_shard / PSUM_SIZE))

  # Initialize local gradient tensor
  local_dp_weight_grad = nl.ndarray(
    (gup_n_tile, nl.par_dim(TILE_SIZE), H_per_shard),
    dtype=dtype,
    buffer=nl.sbuf
  )
  
  # Load initial gradients
  for i_i in nl.affine_range(gup_n_tile):
    p_dim, f_dim = nl.mgrid[0:TILE_SIZE, 0:H_per_shard]
    local_dp_weight_grad[i_i, p_dim, f_dim] = nl.load(
        down_proj_weight_grad[expert_idx[0, 0], p_dim + TILE_SIZE * i_i, f_dim + h_offset],
        dtype=dtype,
        mask=(p_dim + TILE_SIZE*i_i < I_TP)
    )

  # Initialize partial sum and accumulation tensors
  down_proj_w_grad_psum = nl.zeros(
    (hidden_outer_tripcount, gup_n_tile, nl.par_dim(TILE_SIZE), PSUM_SIZE),
    dtype=np.float32,
    buffer=nl.psum,
    lazy_initialization=True
  )
  
  down_proj_w_grad_accum = nl.ndarray(
    (gup_n_tile, nl.par_dim(TILE_SIZE), H_per_shard),
    dtype=dtype,
    buffer=nl.sbuf
  )

  # Compute gradients
  for hi in nl.affine_range(hidden_outer_tripcount):
    for gi in nl.affine_range(gup_n_tile):
      for ti in nl.affine_range(NUM_TILES):
        down_proj_w_grad_psum[
          hi,
          gi,
          nl.arange(TILE_SIZE)[:, None, None],
          TILE_SIZE * nl.arange(4)[None, :, None]
          + nl.arange(TILE_SIZE)[None, None, :],
        ] += nisa.nc_matmul(
          silu_up_mul_gate_trans[
            ti,
            nl.arange(TILE_SIZE)[:, None],
            TILE_SIZE * gi + nl.arange(TILE_SIZE)[None, :],
          ],
          multiply_1[
            ti,
            hi,
            nl.arange(TILE_SIZE)[:, None, None],
            nl.arange(4)[None, :, None],
            nl.arange(TILE_SIZE)[None, None, :],
          ],
          is_stationary_onezero=False,
          is_moving_onezero=False,
          mask=(-1 * nl.arange(TILE_SIZE)[:, None] + -TILE_SIZE * gi + I_TP - 1 >= 0),
        )

      # Accumulate gradients
      down_proj_w_grad_accum[
        gi,
        nl.arange(TILE_SIZE)[:, None],
        PSUM_SIZE * hi + nl.arange(PSUM_SIZE)[None, :],
      ] = nl.add(
        local_dp_weight_grad[
          gi,
          nl.arange(TILE_SIZE)[:, None],
          PSUM_SIZE * hi + nl.arange(PSUM_SIZE)[None, :],
        ],
        down_proj_w_grad_psum[
          hi,
          gi,
          nl.arange(TILE_SIZE)[:, None],
          nl.arange(PSUM_SIZE)[None, :],
        ],
        mask=(-1 * nl.arange(TILE_SIZE)[:, None] + -TILE_SIZE * gi + I_TP - 1 >= 0),
        dtype=dtype,
      )

  # Store final gradients
  for i_i in nl.affine_range(gup_n_tile):
    p_dim, f_dim = nl.mgrid[0:TILE_SIZE, 0:H_per_shard]
    nl.store(
      down_proj_weight_grad[expert_idx[0, 0], p_dim + TILE_SIZE * i_i, f_dim + h_offset],
      value=down_proj_w_grad_accum[i_i, nl.arange(TILE_SIZE)[:, None], f_dim],
      mask=(-1 * p_dim + -TILE_SIZE * i_i + I_TP - 1 >= 0),
    )


def compute_input_hidden_states_gradients(
    hidden_states_grad,
    gate_up_proj_weight,
    silu_up_mul_gate_grad,
    token_indices,
    expert_idx,
    B,
    I_TP,
    is_tensor_update_accumulating,
    dtype,
):
  """
  Compute gradients for the gate up projection.
  
  Args:
    hidden_states_grad: Gradient tensor for hidden states
    gate_up_proj_weight: Gate up projection weights
    silu_up_mul_gate_grad: Gradient of silu-gate multiplication
    token_indices: Token indices
    expert_idx: Expert tensor
    B: Block size
    I_TP: Intermeidate size
    is_tensor_update_accumulating: True if topk is larger than 1
  """
  _, H = hidden_states_grad.shape
  NUM_TILES = int(np.ceil(B / TILE_SIZE))
  gup_n_tile = int(np.ceil(I_TP / 128))
  hidden_outer_tripcount = int(np.ceil(H / PSUM_SIZE))
  if is_tensor_update_accumulating:
    # Load input hidden states gradient
    local_input_hidden_states_grad = nl.ndarray(
      (NUM_TILES, nl.par_dim(TILE_SIZE), H),
      dtype=dtype,
      buffer=nl.sbuf
    )
    for ti in nl.affine_range(NUM_TILES):
      local_input_hidden_states_grad[ti, 0:TILE_SIZE, 0:H] = nl.load(
          hidden_states_grad[token_indices[nl.arange(TILE_SIZE)[:, None], ti], nl.arange(H)[None, :]],
          dtype=dtype,
          mask=None
      )

  # Load gate up weights
  gup_weights = nl.ndarray(
    (hidden_outer_tripcount, 4, nl.par_dim(TILE_SIZE), 2, I_TP),
    dtype=dtype,
    buffer=nl.sbuf
  )
  for h_i in nl.affine_range(hidden_outer_tripcount):
    for h_j in nl.affine_range(4):
      p_dim, i_dim, f_dim = nl.mgrid[0:TILE_SIZE, 0:2, 0:I_TP]
      gup_weights[h_i, h_j, 0:TILE_SIZE, 0:2, 0:I_TP] = nl.load(
        gate_up_proj_weight[
          expert_idx[0, 0],
          PSUM_SIZE * h_i + TILE_SIZE * h_j + p_dim,
          i_dim,
          f_dim,
        ],
        dtype=dtype,
      )

  # Initialize transposed weights and gradient accumulators
  gup_weights_trans = nl.ndarray(
    (hidden_outer_tripcount, 2, nl.par_dim(TILE_SIZE), gup_n_tile, PSUM_SIZE),
    dtype=dtype,
    buffer=nl.sbuf
  )
  input_hidden_states_grad_psum = nl.zeros(
    (NUM_TILES, hidden_outer_tripcount, nl.par_dim(TILE_SIZE), PSUM_SIZE),
    dtype=np.float32,
    buffer=nl.psum,
    lazy_initialization=True
  )
  input_hidden_states_grad_accum = nl.ndarray(
    (NUM_TILES, nl.par_dim(TILE_SIZE), H),
    dtype=dtype,
    buffer=nl.sbuf
  )
  # Transpose weights
  for hi in nl.affine_range(hidden_outer_tripcount):
    for i4 in nl.affine_range(4):
      for ii in nl.affine_range(2):
        for gi in nl.affine_range(gup_n_tile):
          p_dim, f_dim = nl.mgrid[0:TILE_SIZE, 0:TILE_SIZE]
          gup_weights_trans[
            hi, ii, p_dim, gi, TILE_SIZE * i4 + f_dim
          ] = nisa.nc_transpose(
            gup_weights[hi, i4, p_dim, ii, TILE_SIZE * gi + f_dim],
            dtype=dtype,
            mask=(-TILE_SIZE * gi + -1 * f_dim + I_TP - 1 >= 0),
          )

  # Compute gradients
  for n in nl.affine_range(NUM_TILES):
    for hi in nl.affine_range(hidden_outer_tripcount):
      for i2 in nl.affine_range(2):
        for gi in nl.affine_range(gup_n_tile):
          input_hidden_states_grad_psum[
            n,
            hi,
            nl.arange(TILE_SIZE)[:, None],
            nl.arange(PSUM_SIZE)[None, :],
          ] += nisa.nc_matmul(
            silu_up_mul_gate_grad[
              gi,
              i2,
              nl.arange(TILE_SIZE)[:, None],
              TILE_SIZE * n + nl.arange(TILE_SIZE)[None, :],
            ][-TILE_SIZE * gi + -1 * nl.arange(TILE_SIZE)[:, None] + I_TP - 1 >= 0],
            gup_weights_trans[
              hi,
              i2,
              nl.arange(TILE_SIZE)[:, None],
              gi,
              nl.arange(PSUM_SIZE)[None, :],
            ][-TILE_SIZE * gi + -1 * nl.arange(TILE_SIZE)[:, None] + I_TP - 1 >= 0],
            is_stationary_onezero=False,
            is_moving_onezero=False,
            mask=None,
          )
      res_p, res_f = nl.mgrid[0:TILE_SIZE, 0:PSUM_SIZE]
      if is_tensor_update_accumulating:
        # Accumulate gradients
        input_hidden_states_grad_accum[n, res_p, PSUM_SIZE * hi + res_f] = nl.add(
          local_input_hidden_states_grad[n, res_p, PSUM_SIZE * hi + res_f],
          input_hidden_states_grad_psum[n, hi, res_p, res_f],
          mask=None,
          dtype=dtype,
        )
      else:
        input_hidden_states_grad_accum[n, res_p, PSUM_SIZE * hi + res_f] = nl.copy(
          input_hidden_states_grad_psum[n, hi, res_p, res_f],
          mask=None,
          dtype=dtype,
        )

  # Store gradients
  for ti in nl.affine_range(NUM_TILES):
    nl.store(
      hidden_states_grad[
        token_indices[nl.arange(TILE_SIZE)[:, None], ti],
        nl.arange(H)[None, :],
      ],
      value=input_hidden_states_grad_accum[
        ti,
        nl.arange(TILE_SIZE)[:, None],
        nl.arange(H)[None, :],
      ],
      mask=None,
    )
  return input_hidden_states_grad_accum, gup_weights_trans

def compute_input_hidden_states_gradients_shard(
  hidden_states_grad,
  gate_up_proj_weight,
  silu_up_mul_gate_grad,
  token_indices,
  expert_idx,
  B,
  I_TP,
  is_tensor_update_accumulating,
  num_shards,
  shard_id,
  dtype
):
  """
  Compute gradients for the gate up projection.
  
  Args:
    hidden_states_grad: Gradient tensor for hidden states
    gate_up_proj_weight: Gate up projection weights
    silu_up_mul_gate_grad: Gradient of silu-gate multiplication
    token_indices: Token indices
    expert_idx: Expert tensor
    B: Block size
    I_TP: Intermeidate size
    is_tensor_update_accumulating: True if topk is larger than 1
    num_shards:
    shard_id:
  """
  _, H = hidden_states_grad.shape
  H_per_shard = H // num_shards
  h_offset = H_per_shard * shard_id
  NUM_TILES = int(np.ceil(B / TILE_SIZE))
  gup_n_tile = int(np.ceil(I_TP / 128))
  hidden_outer_tripcount = int(np.ceil(H_per_shard / PSUM_SIZE))

  if is_tensor_update_accumulating:
    # Load input hidden states gradient
    local_input_hidden_states_grad = nl.ndarray(
      (NUM_TILES, nl.par_dim(TILE_SIZE), H_per_shard),
      dtype=dtype,
      buffer=nl.sbuf
    )
    for ti in nl.affine_range(NUM_TILES):
      local_input_hidden_states_grad[ti, 0:TILE_SIZE, 0:H_per_shard] = nl.load(
          hidden_states_grad[token_indices[nl.arange(TILE_SIZE)[:, None], ti], nl.arange(H_per_shard)[None, :] + h_offset],
          dtype=dtype,
          mask=None
      )

  # Load gate up weights
  gup_weights = nl.ndarray(
    (hidden_outer_tripcount, 4, nl.par_dim(TILE_SIZE), 2, I_TP),
    dtype=dtype,
    buffer=nl.sbuf
  )
  for h_i in nl.affine_range(hidden_outer_tripcount):
    for h_j in nl.affine_range(4):
      p_dim, i_dim, f_dim = nl.mgrid[0:TILE_SIZE, 0:2, 0:I_TP]
      gup_weights[h_i, h_j, 0:TILE_SIZE, 0:2, 0:I_TP] = nl.load(
        gate_up_proj_weight[
          expert_idx[0, 0],
          PSUM_SIZE * h_i + TILE_SIZE * h_j + h_offset+ p_dim,
          i_dim,
          f_dim,
        ],
        dtype=dtype,
      )

  # Initialize transposed weights and gradient accumulators
  gup_weights_trans = nl.ndarray(
    (hidden_outer_tripcount, 2, nl.par_dim(TILE_SIZE), gup_n_tile, PSUM_SIZE),
    dtype=dtype,
    buffer=nl.sbuf
  )
  input_hidden_states_grad_psum = nl.zeros(
    (NUM_TILES, hidden_outer_tripcount, nl.par_dim(TILE_SIZE), PSUM_SIZE),
    dtype=np.float32,
    buffer=nl.psum,
    lazy_initialization=True
  )
  input_hidden_states_grad_accum = nl.ndarray(
    (NUM_TILES, nl.par_dim(TILE_SIZE), H),
    dtype=dtype,
    buffer=nl.sbuf
  )
  # Transpose weights
  for hi in nl.affine_range(hidden_outer_tripcount):
    for i4 in nl.affine_range(4):
      for ii in nl.affine_range(2):
        for gi in nl.affine_range(gup_n_tile):
          p_dim, f_dim = nl.mgrid[0:TILE_SIZE, 0:TILE_SIZE]
          gup_weights_trans[
            hi, ii, p_dim, gi, TILE_SIZE * i4 + f_dim
          ] = nisa.nc_transpose(
            gup_weights[hi, i4, p_dim, ii, TILE_SIZE * gi + f_dim],
            dtype=dtype,
            mask=(-TILE_SIZE * gi + -1 * f_dim + I_TP - 1 >= 0),
          )

  # Compute gradients
  for n in nl.affine_range(NUM_TILES):
    for hi in nl.affine_range(hidden_outer_tripcount):
      for i2 in nl.affine_range(2):
        for gi in nl.affine_range(gup_n_tile):
          input_hidden_states_grad_psum[
            n,
            hi,
            nl.arange(TILE_SIZE)[:, None],
            nl.arange(PSUM_SIZE)[None, :],
          ] += nisa.nc_matmul(
            silu_up_mul_gate_grad[
              gi,
              i2,
              nl.arange(TILE_SIZE)[:, None],
              TILE_SIZE * n + nl.arange(TILE_SIZE)[None, :],
            ][-TILE_SIZE * gi + -1 * nl.arange(TILE_SIZE)[:, None] + I_TP - 1 >= 0],
            gup_weights_trans[
              hi,
              i2,
              nl.arange(TILE_SIZE)[:, None],
              gi,
              nl.arange(PSUM_SIZE)[None, :],
            ][-TILE_SIZE * gi + -1 * nl.arange(TILE_SIZE)[:, None] + I_TP - 1 >= 0],
            is_stationary_onezero=False,
            is_moving_onezero=False,
            mask=None,
          )

      res_p, res_f = nl.mgrid[0:TILE_SIZE, 0:PSUM_SIZE]
      if is_tensor_update_accumulating:
        # Accumulate gradients
        input_hidden_states_grad_accum[n, res_p, PSUM_SIZE * hi + res_f] = nl.add(
          local_input_hidden_states_grad[n, res_p, PSUM_SIZE * hi + res_f],
          input_hidden_states_grad_psum[n, hi, res_p, res_f],
          mask=None,
          dtype=dtype,
        )
      else:
        input_hidden_states_grad_accum[n, res_p, PSUM_SIZE * hi + res_f] = nl.copy(
          input_hidden_states_grad_psum[n, hi, res_p, res_f],
          mask=None,
          dtype=dtype,
        )

  # Store gradients
  for ti in nl.affine_range(NUM_TILES):
    nl.store(
      hidden_states_grad[
        token_indices[nl.arange(TILE_SIZE)[:, None], ti],
        nl.arange(H_per_shard)[None, :] + h_offset,
      ],
      value=input_hidden_states_grad_accum[
        ti,
        nl.arange(TILE_SIZE)[:, None],
        nl.arange(H_per_shard)[None, :],
      ],
      mask=None,
    )
  return input_hidden_states_grad_accum, gup_weights_trans


def compute_gate_up_proj_weight_gradients(
  gate_up_proj_weight_grad,
  hidden_states,
  silu_up_mul_gate_grad_trans,
  expert_idx,
  token_indices,
  B,
  dtype,
):
  """
  Compute gradients for gate up projection weights.
  
  Args:
    gate_up_proj_weight_grad: Gradient tensor for gate up projection weights
    hidden_states: Forward hidden states
    silu_up_mul_gate_grad_trans: Transposed gradient of silu-gate multiplication
    expert_idx: Expert tensor
    token_indices: Token indices
    B: block size
  """
  E, H, _, I_TP = gate_up_proj_weight_grad.shape
  NUM_TILES = int(np.ceil(B / TILE_SIZE))

  hidden_outer_tripcount = int(np.ceil(H / PSUM_SIZE))

  # Load initial weight gradients
  local_gup_weight_grad = nl.ndarray(
    (hidden_outer_tripcount, 4, nl.par_dim(TILE_SIZE), 2, I_TP),
    dtype=dtype,
    buffer=nl.sbuf
  )
  for h_i in nl.affine_range(hidden_outer_tripcount):
    for h_j in nl.affine_range(4):
      p_dim, i_dim, f_dim = nl.mgrid[0:TILE_SIZE, 0:2, 0:I_TP]
      local_gup_weight_grad[h_i, h_j, 0:TILE_SIZE, 0:2, 0:I_TP] = nl.load(
        gate_up_proj_weight_grad[
          expert_idx[0, 0],
          PSUM_SIZE * h_i + TILE_SIZE * h_j + p_dim,
          i_dim,
          f_dim
        ],
        dtype=dtype
      )

  # Load forward hidden states
  local_hidden_states_fwd = nl.ndarray(
    (NUM_TILES, nl.par_dim(TILE_SIZE), H),
    dtype=dtype,
    buffer=nl.sbuf
  )
  for ti in nl.affine_range(NUM_TILES):
    p_dim, f_dim = nl.mgrid[0:TILE_SIZE, 0:H]
    local_hidden_states_fwd[ti, p_dim, f_dim] = nl.load(
      hidden_states[token_indices[nl.arange(TILE_SIZE)[:, None], ti], f_dim],
      dtype=dtype,
      mask=None
    )

  # Initialize gradient accumulators
  gup_weight_grad = nl.zeros(
    (hidden_outer_tripcount, 4, 2, nl.par_dim(TILE_SIZE), I_TP),
    dtype=np.float32,
    name="",
    buffer=nl.psum,
    lazy_initialization=True
  )
  gup_weight_grad_accum = nl.ndarray(
    (hidden_outer_tripcount, 4, nl.par_dim(TILE_SIZE), 2, I_TP),
    dtype=dtype,
    name="add.6",
    buffer=nl.sbuf
  )

  # Compute and accumulate gradients
  for hi in nl.affine_range(hidden_outer_tripcount):
    for i4 in nl.affine_range(4):
      for i2 in nl.affine_range(2):
        for ti in nl.affine_range(NUM_TILES):
          gup_weight_grad[hi, i4, i2, nl.arange(TILE_SIZE)[:, None], nl.arange(I_TP)[None, :]] += nisa.nc_matmul(
            local_hidden_states_fwd[ti, nl.arange(TILE_SIZE)[:, None],
                                  PSUM_SIZE*hi + TILE_SIZE*i4 + nl.arange(TILE_SIZE)[None, :]],
            silu_up_mul_gate_grad_trans[ti, i2, nl.arange(TILE_SIZE)[:, None],
                                        nl.arange(I_TP)[None, :]],
            is_stationary_onezero=False,
            is_moving_onezero=False,
            mask=None
          )

        gup_weight_grad_accum[hi, i4, nl.arange(TILE_SIZE)[:, None], i2, nl.arange(I_TP)[None, :]] = nl.add(
          local_gup_weight_grad[hi, i4, nl.arange(TILE_SIZE)[:, None], i2, nl.arange(I_TP)[None, :]],
          gup_weight_grad[hi, i4, i2, nl.arange(TILE_SIZE)[:, None], nl.arange(I_TP)[None, :]],
          mask=None,
          dtype=dtype
        )

  # Store final gradients
  for hi in nl.affine_range(hidden_outer_tripcount):
    for i4 in nl.affine_range(4):
      p_dim, i_dim, f_dim = nl.mgrid[0:TILE_SIZE, 0:2, 0:I_TP]
      nl.store(
        gate_up_proj_weight_grad[
          expert_idx[0, 0],
          PSUM_SIZE*hi + TILE_SIZE*i4 + p_dim,
          i_dim,
          f_dim
        ],
        value=gup_weight_grad_accum[hi, i4, p_dim, i_dim, f_dim],
        mask=None
      )

    return gate_up_proj_weight_grad


def compute_gate_up_proj_weight_gradients_shard(
  gate_up_proj_weight_grad,
  hidden_states,
  silu_up_mul_gate_grad_trans,
  expert_idx,
  token_indices,
  B,
  num_shards,
  shard_id,
  dtype
):
  """
  Compute gradients for gate up projection weights.
  
  Args:
    gate_up_proj_weight_grad: Gradient tensor for gate up projection weights
    hidden_states: Forward hidden states
    silu_up_mul_gate_grad_trans: Transposed gradient of silu-gate multiplication
    expert_idx: Expert tensor
    token_indices: Token indices
    B: block size
    num_shards:
    shard_id
  """
  E, H, _, I_TP = gate_up_proj_weight_grad.shape
  H_per_shard = H // num_shards
  h_offset = H_per_shard * shard_id

  hidden_outer_tripcount = int(np.ceil(H_per_shard / PSUM_SIZE))
  NUM_TILES = int(np.ceil(B / TILE_SIZE))

  # Load initial weight gradients
  local_gup_weight_grad = nl.ndarray(
    (hidden_outer_tripcount, 4, nl.par_dim(TILE_SIZE), 2, I_TP),
    dtype=dtype,
    buffer=nl.sbuf
  )
  for h_i in nl.affine_range(hidden_outer_tripcount):
    for h_j in nl.affine_range(4):
      p_dim, i_dim, f_dim = nl.mgrid[0:TILE_SIZE, 0:2, 0:I_TP]
      local_gup_weight_grad[h_i, h_j, 0:TILE_SIZE, 0:2, 0:I_TP] = nl.load(
        gate_up_proj_weight_grad[
          expert_idx[0, 0],
          PSUM_SIZE * h_i + TILE_SIZE * h_j + p_dim + h_offset,
          i_dim,
          f_dim
        ],
        dtype=dtype
      )

  # Load forward hidden states
  local_hidden_states_fwd = nl.ndarray(
    (NUM_TILES, nl.par_dim(TILE_SIZE), H_per_shard),
    dtype=dtype,
    buffer=nl.sbuf
  )
  for ti in nl.affine_range(NUM_TILES):
    p_dim, f_dim = nl.mgrid[0:TILE_SIZE, 0:H_per_shard]
    local_hidden_states_fwd[ti, p_dim, f_dim] = nl.load(
      hidden_states[token_indices[nl.arange(TILE_SIZE)[:, None], ti], f_dim + h_offset],
      dtype=dtype,
      mask=None
    )

  # Initialize gradient accumulators
  gup_weight_grad = nl.zeros(
    (hidden_outer_tripcount, 4, 2, nl.par_dim(TILE_SIZE), I_TP),
    dtype=np.float32,
    name="",
    buffer=nl.psum,
    lazy_initialization=True
  )
  gup_weight_grad_accum = nl.ndarray(
    (hidden_outer_tripcount, 4, nl.par_dim(TILE_SIZE), 2, I_TP),
    dtype=dtype,
    name="add.6",
    buffer=nl.sbuf
  )

  # Compute and accumulate gradients
  for hi in nl.affine_range(hidden_outer_tripcount):
    for i4 in nl.affine_range(4):
      for i2 in nl.affine_range(2):
        for ti in nl.affine_range(NUM_TILES):
          gup_weight_grad[hi, i4, i2, nl.arange(TILE_SIZE)[:, None], nl.arange(I_TP)[None, :]] += nisa.nc_matmul(
            local_hidden_states_fwd[ti, nl.arange(TILE_SIZE)[:, None],
                                  PSUM_SIZE*hi + TILE_SIZE*i4 + nl.arange(TILE_SIZE)[None, :]],
            silu_up_mul_gate_grad_trans[ti, i2, nl.arange(TILE_SIZE)[:, None],
                                        nl.arange(I_TP)[None, :]],
            is_stationary_onezero=False,
            is_moving_onezero=False,
            mask=None
          )

        gup_weight_grad_accum[hi, i4, nl.arange(TILE_SIZE)[:, None], i2, nl.arange(I_TP)[None, :]] = nl.add(
          local_gup_weight_grad[hi, i4, nl.arange(TILE_SIZE)[:, None], i2, nl.arange(I_TP)[None, :]],
          gup_weight_grad[hi, i4, i2, nl.arange(TILE_SIZE)[:, None], nl.arange(I_TP)[None, :]],
          mask=None,
          dtype=dtype
        )

  # Store final gradients
  for hi in nl.affine_range(hidden_outer_tripcount):
    for i4 in nl.affine_range(4):
      p_dim, i_dim, f_dim = nl.mgrid[0:TILE_SIZE, 0:2, 0:I_TP]
      nl.store(
        gate_up_proj_weight_grad[
          expert_idx[0, 0],
          PSUM_SIZE*hi + TILE_SIZE*i4 + p_dim + h_offset,
          i_dim,
          f_dim
        ],
        value=gup_weight_grad_accum[hi, i4, p_dim, i_dim, f_dim],
        mask=None
      )

    return gate_up_proj_weight_grad

# need accumulate: hidden_states_grad, gate_up_proj_weight_grad, down_proj_weight_grad, output_hidden_states_grad
def blockwise_mm_bwd_baseline(hidden_states: nt.tensor,
                              hidden_states_grad: nt.tensor,
                              expert_affinities_masked: nt.tensor,
                              expert_affinities_masked_grad: nt.tensor,
                              gate_up_proj_weight: nt.tensor,
                              gate_up_proj_weight_grad: nt.tensor,
                              gate_up_proj_act_checkpoint_T: nt.tensor,
                              down_proj_weight: nt.tensor,
                              down_proj_weight_grad: nt.tensor,
                              down_proj_act_checkpoint: nt.tensor,
                              token_position_to_id: nt.tensor,
                              block_to_expert: nt.tensor,
                              output_hidden_states_grad: nt.tensor,
                              # Meta parameters
                              block_size: int=512,
                              skip_dma: SkipMode = SkipMode(False, False),
                              compute_dtype=nl.bfloat16,
                              is_tensor_update_accumulating: bool=True,
                              ):
  
  assert skip_dma.skip_token == False and skip_dma.skip_weight == False, "Skip dma is not supported yet"

  # infer config from input shape
  T, H = hidden_states.shape
  B = block_size
  E, I_TP, _ = down_proj_weight.shape
  N = token_position_to_id.shape[0] // B
  NUM_TILES = B // TILE_SIZE

  initialize_gradient_outputs(
    hidden_states_grad=hidden_states_grad,
    expert_affinities_masked_grad=expert_affinities_masked_grad,
    gate_up_proj_weight_grad=gate_up_proj_weight_grad,
    down_proj_weight_grad=down_proj_weight_grad,
  )

  for block_idx in nl.sequential_range(N):
    token_indices = load_token_indices(token_position_to_id, block_idx, B, NUM_TILES)
    expert_idx = load_block_expert(block_to_expert, block_idx)
    expert_affinity_f32 = compute_expert_affinity(
      token_indices=token_indices, 
      expert_idx=expert_idx, 
      expert_affinities_masked=expert_affinities_masked, 
      NUM_TILES=NUM_TILES, 
      E=E,
      dtype=compute_dtype,
    )
    local_output_hidden_states_grad = compute_expert_affinity_gradient(
      output_hidden_states_grad=output_hidden_states_grad,
      down_proj_act_checkpoint=down_proj_act_checkpoint,
      token_indices=token_indices,
      expert_affinities_masked_grad=expert_affinities_masked_grad,
      expert_idx=expert_idx,
      block_idx=block_idx,
      NUM_TILES=NUM_TILES,
      E=E,
      dtype=compute_dtype,
    )
    multiply_1, multiply_1_transpose = compute_down_proj_mm_gradient_and_transpose(
      local_output_hidden_states_grad=local_output_hidden_states_grad,
      expert_affinity_f32=expert_affinity_f32,
      H=H,
      NUM_TILES=NUM_TILES,
      dtype=compute_dtype,
    )
    local_gate_up_proj_act_checkpoint, silu_up_proj = recompute_silu_activation(
      gate_up_proj_act_checkpoint_T=gate_up_proj_act_checkpoint_T,
      block_idx=block_idx,
      dtype=compute_dtype,
    )
    down_proj_w_pftranspose = load_and_transpose_down_proj_weights(
      down_proj_weight=down_proj_weight,
      expert_idx=expert_idx,
      dtype=compute_dtype,
    )
    multiply_1_grad_sbuf = compute_multiply_1_gradient(
      down_proj_w_pftranspose=down_proj_w_pftranspose, 
      multiply_1_transpose=multiply_1_transpose,
      B=B, 
      I_TP=I_TP, 
      H=H,
      dtype=compute_dtype,
    )
    silu_up_mul_gate_grad, silu_up_mul_gate_grad_trans, silu_up_mul_gate_trans = compute_silu_gradient(
      multiply_1_grad_sbuf=multiply_1_grad_sbuf, 
      silu_up_proj=silu_up_proj, 
      local_gate_up_proj_act_checkpoint=local_gate_up_proj_act_checkpoint, 
      B=B, 
      I_TP=I_TP,
      dtype=compute_dtype,
    )

    compute_down_proj_weight_gradients(
      down_proj_weight_grad=down_proj_weight_grad,
      silu_up_mul_gate_trans=silu_up_mul_gate_trans,
      multiply_1=multiply_1,
      expert_idx=expert_idx,
      B=B,
      H=H,
      I_TP=I_TP,
      dtype=compute_dtype
    )
    compute_input_hidden_states_gradients(
      hidden_states_grad=hidden_states_grad,
      gate_up_proj_weight=gate_up_proj_weight,
      silu_up_mul_gate_grad=silu_up_mul_gate_grad,
      token_indices=token_indices,
      expert_idx=expert_idx,
      B=B,
      I_TP=I_TP,
      is_tensor_update_accumulating=is_tensor_update_accumulating,
      dtype=compute_dtype
    )
    compute_gate_up_proj_weight_gradients(
      gate_up_proj_weight_grad=gate_up_proj_weight_grad,
      hidden_states=hidden_states,
      silu_up_mul_gate_grad_trans=silu_up_mul_gate_grad_trans,
      expert_idx=expert_idx,
      token_indices=token_indices,
      B=B,
      dtype=compute_dtype
  )


def blockwise_mm_bwd_baseline_shard_hidden(hidden_states,
                                            hidden_states_grad,
                                            expert_affinities_masked,
                                            expert_affinities_masked_grad,
                                            gate_up_proj_weight,
                                            gate_up_proj_weight_grad,
                                            gate_up_proj_act_checkpoint_T,
                                            down_proj_weight,
                                            down_proj_weight_grad,
                                            down_proj_act_checkpoint,
                                            token_position_to_id,
                                            block_to_expert,
                                            output_hidden_states_grad,
                                            block_size=512,
                                            skip_dma: SkipMode = SkipMode(False, False),
                                            compute_dtype=nl.bfloat16,
                                            is_tensor_update_accumulating=True):
  
  assert skip_dma.skip_token == False and skip_dma.skip_weight == False, "Skip dma is not supported yet"
  assert block_size in (512, 1024), "Block size must be 512 or 1024"
  # Infer Config from input shape
  T, H = hidden_states.shape
  B = block_size
  E, I_TP, _ = down_proj_weight.shape
  N = token_position_to_id.shape[0] // B
  NUM_TILES = B // TILE_SIZE
  # check_blockwise_mm_bwd_kernel_compatibility(H, B, I_TP)

  num_shards = nl.num_programs(axes=0)
  shard_id = nl.program_id(axis=0)

  initialize_gradient_outputs_shard(
    hidden_states_grad=hidden_states_grad,
    expert_affinities_masked_grad=expert_affinities_masked_grad,
    gate_up_proj_weight_grad=gate_up_proj_weight_grad,
    down_proj_weight_grad=down_proj_weight_grad,
    num_shards=num_shards,
    shard_id=shard_id,
  )

  H_per_shard = H // num_shards

  for block_idx in nl.sequential_range(N):
    token_indices = load_token_indices(token_position_to_id, block_idx, B, NUM_TILES)
    expert_idx = load_block_expert(block_to_expert, block_idx)
    expert_affinity_f32 = compute_expert_affinity(
      token_indices=token_indices, 
      expert_idx=expert_idx, 
      expert_affinities_masked=expert_affinities_masked, 
      NUM_TILES=NUM_TILES, 
      E=E,
      dtype=compute_dtype,
    )
    local_output_hidden_states_grad = compute_expert_affinity_gradient_shard(
      output_hidden_states_grad=output_hidden_states_grad,
      down_proj_act_checkpoint=down_proj_act_checkpoint,
      token_indices=token_indices,
      expert_affinities_masked_grad=expert_affinities_masked_grad,
      expert_idx=expert_idx,
      block_idx=block_idx,
      NUM_TILES=NUM_TILES,
      E=E,
      num_shards=num_shards,
      shard_id=shard_id,
      dtype=compute_dtype,
    )
    multiply_1, multiply_1_transpose = compute_down_proj_mm_gradient_and_transpose(
      local_output_hidden_states_grad=local_output_hidden_states_grad,
      expert_affinity_f32=expert_affinity_f32,
      H=H_per_shard,
      NUM_TILES=NUM_TILES,
      dtype=compute_dtype,
    )
    #TODO: add sharding
    local_gate_up_proj_act_checkpoint, silu_up_proj = recompute_silu_activation(
      gate_up_proj_act_checkpoint_T=gate_up_proj_act_checkpoint_T,
      block_idx=block_idx,
      dtype=compute_dtype,
    )
    down_proj_w_pftranspose = load_and_transpose_down_proj_weights_shard(
      down_proj_weight=down_proj_weight,
      expert_idx=expert_idx,
      num_shards=num_shards,
      shard_id=shard_id,
      dtype=compute_dtype,
    )
    multiply_1_grad_sbuf = compute_multiply_1_gradient_shard(
      down_proj_w_pftranspose=down_proj_w_pftranspose,
      multiply_1_transpose=multiply_1_transpose,
      B=B,
      I_TP=I_TP,
      H=H,
      num_shards=num_shards,
      shard_id=shard_id,
      dtype=compute_dtype,
    )
    silu_up_mul_gate_grad, silu_up_mul_gate_grad_trans, silu_up_mul_gate_trans = compute_silu_gradient(
      multiply_1_grad_sbuf=multiply_1_grad_sbuf,
      silu_up_proj=silu_up_proj,
      local_gate_up_proj_act_checkpoint=local_gate_up_proj_act_checkpoint,
      B=B,
      I_TP=I_TP,
      dtype=compute_dtype
    )
    compute_down_proj_weight_gradients_shard(
      down_proj_weight_grad=down_proj_weight_grad,
      silu_up_mul_gate_trans=silu_up_mul_gate_trans,
      multiply_1=multiply_1,
      expert_idx=expert_idx,
      B=B,
      H=H,
      I_TP=I_TP,
      num_shards=num_shards,
      shard_id=shard_id,
      dtype=compute_dtype
    )
    compute_input_hidden_states_gradients_shard(
      hidden_states_grad=hidden_states_grad,
      gate_up_proj_weight=gate_up_proj_weight,
      silu_up_mul_gate_grad=silu_up_mul_gate_grad,
      token_indices=token_indices,
      expert_idx=expert_idx,
      B=B,
      I_TP=I_TP,
      is_tensor_update_accumulating=is_tensor_update_accumulating,
      num_shards=num_shards,
      shard_id=shard_id,
      dtype=compute_dtype
    )
    compute_gate_up_proj_weight_gradients_shard(
      gate_up_proj_weight_grad=gate_up_proj_weight_grad,
      hidden_states=hidden_states,
      silu_up_mul_gate_grad_trans=silu_up_mul_gate_grad_trans,
      expert_idx=expert_idx,
      token_indices=token_indices,
      B=B,
      num_shards=num_shards,
      shard_id=shard_id,
      dtype=compute_dtype
    )
