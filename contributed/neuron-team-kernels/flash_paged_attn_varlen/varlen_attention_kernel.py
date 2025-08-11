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

import numpy as np
import neuronxcc.nki as nki
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
from neuronxcc.nki.language import par_dim
from neuronxcc.nki.isa.constants import oob_mode

from constants import B_P_SIZE, B_FMAX_SIZE
from utils import (
    is_power_of_2,
    PF_transpose_with_PE,
)
from flash_attn_impl import (
    prepare_q_update_pred,
    allocate_prefill_accum_buffers,
    allocate_decode_accum_buffers,
    decode_gather_token_last_accum_tile,
    prefill_context_tokens,
    decode_context_tokens,
    active_and_epilogue,
)


def check_input_shape(
    query,
    key,
    value,
    key_cache,
    value_cache,
    prefill_tile_masks,
    decode_tile_masks,
    active_mask,
):
    # tile size from prefill
    PREFILL_INNER_Q_TILE_SIZE = prefill_tile_masks.shape[0]
    # check decode input
    DECODE_K_TILE_SIZE = decode_tile_masks.shape[0]
    assert DECODE_K_TILE_SIZE == B_P_SIZE

    b, h, seqlen_q, d = query.shape
    assert seqlen_q <= 8192, f"Large {seqlen_q=} consumes too much sbuf space"
    if seqlen_q <= B_P_SIZE:
        assert is_power_of_2(
            seqlen_q
        ), f"{seqlen_q=} is expected to be power of 2"
    elif seqlen_q <= B_FMAX_SIZE:
        assert (
            seqlen_q % B_P_SIZE == 0
        ), f"{seqlen_q=} must be mulitple of {B_P_SIZE=}"
    else:
        assert (
            seqlen_q % B_FMAX_SIZE == 0
        ), f"{seqlen_q=} must be multiple of {B_FMAX_SIZE=}"

    assert (
        seqlen_q % PREFILL_INNER_Q_TILE_SIZE == 0
    ), f"{seqlen_q=} must be multiple of {PREFILL_INNER_Q_TILE_SIZE=}"
    assert b == 1, f"Batch size must be 1 for Ragged Tensor, got {b}"
    assert (
        d >= 16 and d <= 128 and is_power_of_2(d)
    ), f" we head_dim must be power of 2 in range [16, 128], got head dim {d}"
    num_blocks, k_h, block_size, _ = key_cache.shape
    assert tuple(key_cache.shape) == (
        num_blocks,
        k_h,
        block_size,
        d,
    ), f"{key_cache.shape=} mismatch!"
    assert tuple(value_cache.shape) == (
        num_blocks,
        k_h,
        block_size,
        d,
    ), f"{value_cache.shape=} mismatch!"
    assert key is None or tuple(key.shape) == (
        1,
        k_h,
        d,
        seqlen_q,
    ), f"key shape {key.shape} mismatch!"
    assert value is None or tuple(value.shape) == (
        1,
        k_h,
        seqlen_q,
        d,
    ), f"value shape {value.shape} mismatch!"
    assert (
        prefill_tile_masks.dtype == nl.uint8
    ), f"{prefill_tile_masks.dtype=} is expected to be uint8"
    assert (
        decode_tile_masks.dtype == nl.uint8
    ), f"{decode_tile_masks.dtype=} is expected to be uint8"
    assert (
        active_mask.dtype == nl.uint8
    ), f"{active_mask.dtype=} is expected to be uint8"

    return (
        b,
        h,
        k_h,
        seqlen_q,
        d,
        PREFILL_INNER_Q_TILE_SIZE,
    )


def merge_decode_buffer(
    olm_buffer,
    decode_tile_q_indices,
    decode_olm_buffer,
    last_tile_indices_sbuf,
):
    q_h_per_k_h = olm_buffer.shape[1]
    B_D_SIZE = olm_buffer.shape[2] - 2
    decode_olm_sbuf = decode_gather_token_last_accum_tile(
        olm_buffer=decode_olm_buffer,
        last_tile_indices_sbuf=last_tile_indices_sbuf,
        q_h_per_k_h=q_h_per_k_h,
        B_D_SIZE=B_D_SIZE,
    )
    TILE_SIZE, NUM_TILES, _, _ = decode_olm_sbuf.shape
    decode_q_start_offset = nl.ndarray((1, 1), dtype=nl.int32)
    decode_q_start_offset[...] = nl.load(decode_tile_q_indices[0, 0])
    for i in nl.affine_range(NUM_TILES):
        base_offset_iota = nisa.iota(
            nl.arange(TILE_SIZE)[None, :] + i * TILE_SIZE,
            dtype=nl.int32,
        )
        offsets = nisa.tensor_tensor(
            base_offset_iota,
            decode_q_start_offset,
            nl.add,
        )
        offsets_transposed = nl.ndarray((par_dim(TILE_SIZE), 1), dtype=nl.int32)
        PF_transpose_with_PE(
            offsets,
            offsets_transposed,
            out_in_psum=False,
        )
        i_p = nl.arange(TILE_SIZE)[:, None, None]
        i_f_h = nl.arange(q_h_per_k_h)[None, :, None]
        i_f_d = nl.arange(B_D_SIZE + 2)[None, None, :]
        nl.store(
            olm_buffer[offsets_transposed[i_p, 0], i_f_h, i_f_d],
            decode_olm_sbuf[i_p, i, i_f_h, i_f_d],
            mode=oob_mode.skip,
        )


@nki.compiler.skip_middle_end_transformations
@nki.jit(
    experimental_flags="experimental-native-scalar-support, experimental-local-tensor-parent",
    enable_out_of_bound_check=False,
)
def flash_paged_attention_varlen(
    *,
    query,
    key,
    value,
    key_cache,
    value_cache,
    active_mask,
    prefill_tile_q_indices,
    prefill_tile_block_tables,
    prefill_tile_masks,
    prefill_num_dynamic_loop_steps,
    prefill_last_tile_indices,
    decode_tile_q_indices,
    decode_tile_block_tables,
    decode_tile_masks,
    decode_num_dynamic_loop_steps,
    decode_last_tile_indices,
    prefill_q_update_pred=None,
    decode_q_update_pred=None,
    dynamic_loop_unroll_factor=1,
    softmax_scale=None,
    mixed_precision=True,
    skip_active=False,
):
    """
    Flash PagedAttention Forward Kernel with Both Prefill and Decode Requests.
      - PagedAttention Paper: https://arxiv.org/abs/2309.06180
      - Chunked Prefill Paper: https://arxiv.org/abs/2403.02310

    IO tensor layouts:
      - query: shape (1, n_q_heads, seq_q, d)
      - key:   shape (1, n_kv_heads, d, seq_k)
      - value: shape (1, n_kv_heads, seq_v, d)
      - key_cache: (max_num_blocks, n_kv_heads, block_size, d)
      - value_cache: (max_num_blocks, n_kv_heads, block_size, d)
      - prefill_tile_q_indices: (max_num_prefill_tiles, large_tile_size_q)
      - prefill_tile_block_tables: (max_num_prefill_tiles, num_block_per_large_tile)
      - prefill_tile_masks: (B_P_SIZE, max_num_prefill_tiles, large_tile_size_q // B_P_SIZE, large_tile_size_k)
      - prefill_num_dynamic_loop_steps: (1, 1)
      - decode_tile_q_indices: (max_num_decode_tiles, 1)
      - decode_tile_block_tables: (max_num_decode_tiles, num_block_per_large_tile)
      - decode_tile_masks: (B_P_SIZE, max_num_decode_tiles, large_tile_size_k // B_P_SIZE)
      - decode_num_dynamic_loop_steps: (1, 1)
      - active_mask: (seq_q, seq_q)
      - decode_last_tile_indices: (max_batch_size, 1)
      - decode_q_update_pred: None or (max_num_decode_tiles, 1)

      - This kernel requires seq_k == seq_v
      - We use continuous batching by default, so the batch dimension is always 1, and different
        requests are concatenated along sequence dimension.
      - We use paged cache blocks (key_cache, value_cache) to store KV cache.

    IO tensor dtypes:
      - This kernel assumes all IO tensors have the same dtype except for block_tables (uint32) and mask (uint8)
      - If mixed_percision is True, then all Tensor Engine operation will be performed in
        bfloat16 and accumulation will be performed in float32. Otherwise the intermediates
        will be in the same type as the inputs.

    Compile-time Constants:
      - sequence_parallel_group: sequence parallel group to shard the cache blocks, List[int].
      - softmax_scale: scaling for softmax, is None, default is `1.0/(d**0.5)`
      - mixed_precision: flag to set non-matmul ops in fp32 precision, defualt is set to `true`,
          if false, we use same precision as input types

    GQA support Notes:
      the spmd kernel for launching kernel should be on kv_heads instead of nheads

    Example usage:
      MHA: q: [b, h, d, s], k: [b, h, d, s], v: [b, h, s, d]
        usage: `flash_fwd[b, h](q, k, v, ...)`
      GQA: q: [b, h, d, s], k: [b, kv_h, d, s], v: [b, kv_h, s, d]
        usage: `flash_fwd[b, kv_h](q, k, v, ...)`
    """
    (b, h, k_h, seqlen_q, d, INNER_Q_TILE_SIZE) = check_input_shape(
        query,
        key,
        value,
        key_cache,
        value_cache,
        prefill_tile_masks,
        decode_tile_masks,
        active_mask,
    )

    kernel_dtype = nl.bfloat16 if mixed_precision else query.dtype
    acc_type = np.dtype(np.float32) if mixed_precision else kernel_dtype
    o = nl.ndarray((b, h, seqlen_q, d), dtype=query.dtype, buffer=nl.shared_hbm)

    assert (
        nl.program_ndim() == 2
    ), f"Expect spmd grid with 2 dimensions, got {nl.program_ndim()} instead!"
    assert nl.num_programs(0) == 1
    assert nl.num_programs(1) == k_h
    batch_id = nl.program_id(axis=0)  # equals 0
    head_id = nl.program_id(axis=1)

    softmax_scale = softmax_scale or (1.0 / (d**0.5))

    B_D_SIZE = d
    q_h_per_k_h = h // k_h

    assert prefill_num_dynamic_loop_steps.dtype == nl.int32
    assert decode_num_dynamic_loop_steps.dtype == nl.int32
    B_F_SIZE = B_FMAX_SIZE

    (olm_buffer,) = allocate_prefill_accum_buffers(
        seqlen_q=seqlen_q,
        INNER_Q_TILE_SIZE=INNER_Q_TILE_SIZE,
        q_h_per_k_h=q_h_per_k_h,
        B_D_SIZE=B_D_SIZE,
        acc_type=acc_type,
    )
    PREFILL_MAX_NUM_TILE = prefill_tile_masks.shape[1]
    assert PREFILL_MAX_NUM_TILE % dynamic_loop_unroll_factor == 0
    if prefill_q_update_pred is None:
        prefill_last_tile_indices_sbuf = nl.load(prefill_last_tile_indices)
        prefill_q_update_pred_hbm = prepare_q_update_pred(
            prefill_last_tile_indices_sbuf,
            PREFILL_MAX_NUM_TILE,
        )
    else:
        prefill_q_update_pred_hbm = prefill_q_update_pred
    prefill_q_update_pred = prefill_q_update_pred_hbm.reshape(
        (
            PREFILL_MAX_NUM_TILE // dynamic_loop_unroll_factor,
            dynamic_loop_unroll_factor,
            1,
        )
    )
    prefill_context_tokens(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        tile_q_indices=prefill_tile_q_indices,
        tile_block_tables=prefill_tile_block_tables,
        tile_masks=prefill_tile_masks,
        num_dynamic_loop_steps=prefill_num_dynamic_loop_steps,
        olm_buffer=olm_buffer,
        q_update_pred=prefill_q_update_pred,
        kernel_dtype=kernel_dtype,
        acc_type=acc_type,
        loop_unroll_factor=dynamic_loop_unroll_factor,
        batch_id=batch_id,
        head_id=head_id,
        k_h=k_h,
        q_h_per_k_h=q_h_per_k_h,
        softmax_scale=softmax_scale,
        B_F_SIZE=B_F_SIZE,
        B_D_SIZE=B_D_SIZE,
    )
    DECODE_MAX_NUM_TILE = decode_tile_masks.shape[1]
    assert DECODE_MAX_NUM_TILE % dynamic_loop_unroll_factor == 0
    (decode_olm_buffer,) = allocate_decode_accum_buffers(
        MAX_NUM_TILE=DECODE_MAX_NUM_TILE,
        q_h_per_k_h=q_h_per_k_h,
        B_D_SIZE=B_D_SIZE,
        acc_type=acc_type,
    )
    decode_last_tile_indices_sbuf = nl.load(decode_last_tile_indices)
    if decode_q_update_pred is None:
        decode_q_update_pred_hbm = prepare_q_update_pred(
            decode_last_tile_indices_sbuf, DECODE_MAX_NUM_TILE
        )
        decode_q_update_pred = decode_q_update_pred_hbm.reshape(
            (
                DECODE_MAX_NUM_TILE // dynamic_loop_unroll_factor,
                dynamic_loop_unroll_factor,
            )
        )

    decode_context_tokens(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        tile_q_indices=decode_tile_q_indices,
        tile_masks=decode_tile_masks,
        tile_block_tables=decode_tile_block_tables,
        num_dynamic_loop_steps=decode_num_dynamic_loop_steps,
        olm_buffer=decode_olm_buffer,
        q_update_pred=decode_q_update_pred,
        kernel_dtype=kernel_dtype,
        acc_type=acc_type,
        loop_unroll_factor=dynamic_loop_unroll_factor,
        batch_id=batch_id,
        head_id=head_id,
        k_h=k_h,
        q_h_per_k_h=q_h_per_k_h,
        softmax_scale=softmax_scale,
        B_D_SIZE=B_D_SIZE,
    )
    # merge decode olm buffer into global accum buffers
    merge_decode_buffer(
        olm_buffer,
        decode_tile_q_indices=decode_tile_q_indices,
        decode_olm_buffer=decode_olm_buffer,
        last_tile_indices_sbuf=decode_last_tile_indices_sbuf,
    )
    active_and_epilogue(
        o=o,
        query=query,
        key=key,
        value=value,
        active_mask=active_mask,
        softmax_scale=softmax_scale,
        olm_buffer=olm_buffer,
        ACTIVE_Q_TILE_SIZE=INNER_Q_TILE_SIZE,
        seqlen_q=seqlen_q,
        batch_id=batch_id,
        head_id=head_id,
        q_h_per_k_h=q_h_per_k_h,
        kernel_dtype=kernel_dtype,
        acc_type=acc_type,
        B_F_SIZE=B_F_SIZE,
        B_D_SIZE=B_D_SIZE,
        skip_active=skip_active,
    )
    return o


def flash_attn_varlen_nkifunc(
    *,
    query,
    key,
    value,
    key_cache,
    value_cache,
    active_mask,
    prefill_tile_q_indices,
    prefill_tile_block_tables,
    prefill_tile_masks,
    prefill_num_dynamic_loop_steps,
    prefill_last_tile_indices,
    prefill_q_update_pred,
    decode_tile_q_indices,
    decode_tile_block_tables,
    decode_tile_masks,
    decode_num_dynamic_loop_steps,
    decode_last_tile_indices,
    decode_q_update_pred,
    dynamic_loop_unroll_factor=1,
    n_kv_head=None,
    head_size=None,
    mixed_precision=True,
    skip_active=False,
    save_artifact_dir=None,
):
    if n_kv_head is None:
        n_kv_head = key_cache.shape[1]
    assert key_cache.shape[1] == n_kv_head
    if head_size is None:
        head_size = key_cache.shape[-1]
    kwargs = dict(
        query=query,
        key=key,
        value=value,
        key_cache=key_cache,
        value_cache=value_cache,
        active_mask=active_mask,
        prefill_tile_q_indices=prefill_tile_q_indices,
        prefill_tile_block_tables=prefill_tile_block_tables,
        prefill_tile_masks=prefill_tile_masks,
        prefill_num_dynamic_loop_steps=prefill_num_dynamic_loop_steps,
        prefill_last_tile_indices=prefill_last_tile_indices,
        prefill_q_update_pred=prefill_q_update_pred,
        decode_tile_q_indices=decode_tile_q_indices,
        decode_tile_block_tables=decode_tile_block_tables,
        decode_tile_masks=decode_tile_masks,
        decode_num_dynamic_loop_steps=decode_num_dynamic_loop_steps,
        decode_last_tile_indices=decode_last_tile_indices,
        decode_q_update_pred=decode_q_update_pred,
        dynamic_loop_unroll_factor=dynamic_loop_unroll_factor,
        softmax_scale=1.0 / (head_size**0.5),
        mixed_precision=mixed_precision,
        skip_active=skip_active,
    )
    assert (
        n_kv_head == 1
    ), f"SPMD launch is not supported with on-chip dynamic control flow"
    if save_artifact_dir:
        assert isinstance(
            query, np.ndarray
        ), "Only Numpy Kernel supports saving artifact"
        return nki.baremetal(
            flash_paged_attention_varlen,
            debug_kernel=True,
            artifacts_dir=save_artifact_dir,
        )[1, n_kv_head](**kwargs)
    else:
        return flash_paged_attention_varlen[1, n_kv_head](**kwargs)
