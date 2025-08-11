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

import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
from neuronxcc import nki
from neuronxcc.nki.language import par_dim

from constants import B_P_SIZE, B_FMAX_SIZE, NEG_INF
from utils import broadcast_partition_with_PE, PF_transpose_with_PE


@nki.jit
def transpose_p_local(
    p_local_transposed,
    p_local,
    Q_TILE_SIZE,
    LARGE_KV_TILE_SIZE,
    B_F_SIZE=B_FMAX_SIZE,
    enable_dma_transpose=False,
):
    assert p_local.shape == (Q_TILE_SIZE, LARGE_KV_TILE_SIZE)
    REDUCTION_SIZE = min(B_P_SIZE, LARGE_KV_TILE_SIZE)
    is_nc_gen2 = nisa.get_nc_version() == nisa.nc_version.gen2
    if is_nc_gen2 or not enable_dma_transpose:
        for i in nl.affine_range(LARGE_KV_TILE_SIZE // B_F_SIZE):
            p_local_t_tmp = nl.ndarray(
                (
                    par_dim(REDUCTION_SIZE),
                    B_F_SIZE // REDUCTION_SIZE * Q_TILE_SIZE,
                ),
                buffer=nl.psum,
                dtype=np.float32 if is_nc_gen2 else p_local.dtype,
            )
            for j in nl.affine_range(B_F_SIZE // REDUCTION_SIZE):

                j_128_slice = nl.ds(j * Q_TILE_SIZE, Q_TILE_SIZE)
                i_j_128_slice = nl.ds(
                    i * B_F_SIZE + j * REDUCTION_SIZE, REDUCTION_SIZE
                )
                p_local_t_tmp[:, j_128_slice] = nisa.nc_transpose(
                    p_local[:, i_j_128_slice],
                    engine=nisa.tensor_engine,
                )
            p_local_transposed[
                :,
                nl.ds(
                    i * (B_F_SIZE // REDUCTION_SIZE * Q_TILE_SIZE),
                    (B_F_SIZE // REDUCTION_SIZE * Q_TILE_SIZE),
                ),
            ] = nl.copy(p_local_t_tmp, dtype=p_local_transposed.dtype)
    else:
        for i in nl.affine_range(LARGE_KV_TILE_SIZE // REDUCTION_SIZE):
            p_local_transposed[:, nl.ds(i * Q_TILE_SIZE, Q_TILE_SIZE)] = (
                nisa.dma_transpose(
                    p_local[:, nl.ds(i * REDUCTION_SIZE, REDUCTION_SIZE)]
                )
            )


def _flash_attention_core(
    q_local_tile,
    k,
    v,
    olm_buffer,
    kernel_dtype,
    acc_type,
    tile_mask,
    use_causal_mask,
    q_tile_idx=None,
    Q_TILE_SIZE=128,
    LARGE_KV_TILE_SIZE=2048,
    B_F_SIZE=512,
    B_D_SIZE=128,
):
    """
    The flash attention core function to calculate self attention between a tile
    of q and a block of K and V.
    q_local_tile: (B_D_SIZE, Q_TILE_SIZE)
    k: (B_D_SIZE, LARGE_KV_TILE_SIZE)
    v: (B_P_SIZE, LARGE_KV_TILE_SIZE // B_P_SIZE, B_D_SIZE)
    The results are stored in the following three buffers
    olm_buffer: (Q_TILE_SIZE, B_D_SIZE + 2)
    """
    assert (
        LARGE_KV_TILE_SIZE % B_P_SIZE == 0
    ), f"{LARGE_KV_TILE_SIZE=} not divisive by {B_P_SIZE=}"
    assert (
        LARGE_KV_TILE_SIZE % B_F_SIZE == 0
    ), f"{LARGE_KV_TILE_SIZE=} not divisive by {B_F_SIZE=}"
    num_k_tile_per_large_tile = LARGE_KV_TILE_SIZE // B_F_SIZE

    qk_res_buf = nl.ndarray(
        (par_dim(Q_TILE_SIZE), LARGE_KV_TILE_SIZE),
        buffer=nl.sbuf,
        dtype=acc_type,
    )
    max_local = nl.zeros(
        (par_dim(Q_TILE_SIZE), num_k_tile_per_large_tile),
        dtype=acc_type,
    )
    is_nc_gen2 = nisa.get_nc_version() == nisa.nc_version.gen2
    for k_i in nl.affine_range(num_k_tile_per_large_tile):
        k_i_b_f_slice = nl.ds(k_i * B_F_SIZE, B_F_SIZE)

        if use_causal_mask:
            multiplication_required_selection = (
                q_tile_idx * Q_TILE_SIZE >= k_i * B_F_SIZE
            )
        else:
            multiplication_required_selection = True

        if multiplication_required_selection:
            qk_psum = nl.ndarray(
                (par_dim(Q_TILE_SIZE), B_F_SIZE),
                dtype=np.float32,
                buffer=nl.psum,
            )  # (128, 512)
            qk_psum[:, :] = nl.matmul(
                q_local_tile, k[:, k_i_b_f_slice], transpose_x=True
            )  # (p(128), 512)
            if is_nc_gen2:
                # XXX: nisa.select_reduce produces wrong results on Trn1
                qk_res_buf[:, k_i_b_f_slice] = nl.where(
                    tile_mask[:, k_i_b_f_slice],
                    qk_psum[:, nl.ds(0, B_F_SIZE)],
                    NEG_INF,
                    dtype=acc_type,
                )
                # Calculate max of the current tile
                max_local[:, k_i] = nisa.tensor_reduce(
                    np.max,
                    qk_res_buf[:, k_i_b_f_slice],
                    axis=(1,),
                    dtype=acc_type,
                    negate=False,
                )
            else:
                nisa.select_reduce(
                    dst=qk_res_buf[:, k_i_b_f_slice],
                    predicate=tile_mask[:, k_i_b_f_slice],
                    on_true=qk_psum[:, nl.ds(0, B_F_SIZE)],
                    on_false=NEG_INF,
                    reduce_cmd=nisa.reduce_cmd.reset_reduce,
                    reduce_res=max_local[:, k_i],
                    reduce_op=np.max,
                )
        else:
            qk_res_buf[:, k_i_b_f_slice] = NEG_INF
            max_local[:, k_i] = NEG_INF

    # Calculate max of the current tile
    max_ = nisa.tensor_reduce(
        np.max,
        max_local[:, :],
        axis=(1,),
        dtype=acc_type,
        negate=False,
    )

    o_previous_scaled = nl.ndarray(
        (par_dim(Q_TILE_SIZE), B_D_SIZE),
        dtype=olm_buffer.dtype,
    )

    m_previous = olm_buffer[:, B_D_SIZE + 1]
    m_current_neg = nisa.tensor_scalar(
        max_,
        nl.maximum,
        m_previous,
        op1=nl.multiply,
        operand1=-1,
    )

    p_local = nl.ndarray(
        (par_dim(Q_TILE_SIZE), LARGE_KV_TILE_SIZE),
        dtype=kernel_dtype,
    )
    REDUCTION_TILE = min(2048, LARGE_KV_TILE_SIZE // 2)

    p_partial_sum = nl.ndarray(
        (par_dim(Q_TILE_SIZE), LARGE_KV_TILE_SIZE // REDUCTION_TILE),
        dtype=acc_type,
    )

    for k_r_i in nl.affine_range(LARGE_KV_TILE_SIZE // REDUCTION_TILE):
        k_r_i_reduce_slice = nl.ds(k_r_i * REDUCTION_TILE, REDUCTION_TILE)

        # compute exp(qk - max)
        # Compute partial row - tile sum of exp(qk - max))
        # FIXME : Use activation accumulate to accumulate over k_r_i loop ?
        p_local[:, k_r_i_reduce_slice] = nisa.activation_reduce(
            np.exp,
            qk_res_buf[:, k_r_i_reduce_slice],
            bias=m_current_neg,
            scale=1.0,
            reduce_op=nl.add,
            reduce_res=p_partial_sum[:, k_r_i],
            dtype=kernel_dtype,
        )

    ps = nl.sum(p_partial_sum, axis=1, dtype=acc_type)

    p_local_transposed = nl.ndarray(
        (par_dim(B_P_SIZE), LARGE_KV_TILE_SIZE // B_P_SIZE * Q_TILE_SIZE),
        dtype=kernel_dtype,
    )
    transpose_p_local(
        p_local_transposed=p_local_transposed,
        p_local=p_local,
        Q_TILE_SIZE=Q_TILE_SIZE,
        LARGE_KV_TILE_SIZE=LARGE_KV_TILE_SIZE,
        B_F_SIZE=B_F_SIZE,
    )

    pv_psum = nl.zeros(
        (par_dim(Q_TILE_SIZE), B_D_SIZE),
        dtype=np.float32,
        buffer=nl.psum,
    )
    for k_i in nl.affine_range(LARGE_KV_TILE_SIZE // B_P_SIZE):
        pv_psum[:, :] += nl.matmul(
            p_local_transposed[:, nl.ds(k_i * Q_TILE_SIZE, Q_TILE_SIZE)],
            v[:, nl.ds(k_i * B_D_SIZE, B_D_SIZE)],
            transpose_x=True,
        )  # (128, 128) (p(Br), d)

    # Compute scaling factor
    alpha = nisa.activation(
        np.exp,
        m_previous,
        bias=m_current_neg,
        scale=1.0,
    )

    olm_buffer[:, B_D_SIZE + 1] = nisa.activation(
        nl.copy,
        m_current_neg,
        scale=-1.0,
    )
    o_previous_scaled[...] = nl.multiply(
        olm_buffer[:, nl.ds(0, B_D_SIZE)],
        alpha,
    )
    olm_buffer[:, nl.ds(0, B_D_SIZE)] = nl.add(o_previous_scaled, pv_psum)

    l_prev = olm_buffer[:, B_D_SIZE] * alpha
    olm_buffer[:, B_D_SIZE] = l_prev + ps


def _flash_attention_core_kq_matmul(
    q_local_tile,
    o_buffer_sbuf,
    l_buffer_sbuf,
    m_buffer_sbuf,
    kernel_dtype,
    acc_type,
    tile_mask,
    q_update_pred_broadcast,
    key_cache,
    value_cache,
    block_tables_sbuf,
    large_tile_idx,
    num_blocks_per_large_tile,
    block_size,
    k_load_buffer,
    v_load_buffer,
    sumexp_ones,
    identity_for_transpose_k,
    identity_for_transpose_m_step1,
    identity_for_transpose_m_step2,
):
    """
    The flash attention core function to calculate self attention between a tile
    of q and a block of K and V.
    Input:
    q_local_tile: (B_D_SIZE, q_h_per_k_h)
    k: (B_D_SIZE, kv_tile_size)
    v: (B_P_SIZE, kv_tile_size // B_P_SIZE, B_D_SIZE)
    The results are stored in the following three buffers
    o_buffer_sbuf: (B_D_SIZE, NUM_LARGE_TILE, q_h_per_k_h)
    l_buffer_sbuf: (1, NUM_LAEGE_TILE, q_h_per_k_h)
    m_buffer_sbuf: (1, NUM_LAEGE_TILE, q_h_per_k_h)
    """
    MULTI_BUFFER = k_load_buffer.shape[1]
    B_D_SIZE, NUM_LARGE_TILE, Q_TILE_SIZE = q_local_tile.shape
    assert num_blocks_per_large_tile % B_P_SIZE == 0
    num_loads = num_blocks_per_large_tile // B_P_SIZE
    assert Q_TILE_SIZE <= B_P_SIZE, f"{Q_TILE_SIZE=} > {B_P_SIZE=}"
    k = nl.ndarray(
        (par_dim(B_D_SIZE), num_loads * block_size * B_P_SIZE),
        dtype=kernel_dtype,
    )
    k_reshaped = k.reshape(
        (B_D_SIZE, num_loads, block_size, B_P_SIZE),
    )
    for load_idx in nl.affine_range(num_loads):
        for tb_i in nl.affine_range(block_size):
            if k_load_buffer.dtype != kernel_dtype:
                k_src = nl.copy(
                    k_load_buffer[
                        :,
                        large_tile_idx % MULTI_BUFFER,
                        load_idx,
                        nl.ds(tb_i * B_D_SIZE, B_D_SIZE),
                    ],
                    dtype=kernel_dtype,
                )
            else:
                k_src = k_load_buffer[
                    :,
                    large_tile_idx % MULTI_BUFFER,
                    load_idx,
                    nl.ds(tb_i * B_D_SIZE, B_D_SIZE),
                ]
            if nisa.get_nc_version() == nisa.nc_version.gen2:
                PF_transpose_with_PE(
                    src=k_src,
                    out=k_reshaped[
                        :,
                        load_idx,
                        tb_i,
                    ],
                    identity_for_transpose=identity_for_transpose_k,
                )
            else:
                k_t_psum = nl.ndarray(
                    (B_D_SIZE, B_P_SIZE),
                    dtype=kernel_dtype,
                    buffer=nl.psum,
                )
                k_t_psum[...] = nisa.nc_transpose(
                    k_src,
                    engine=nisa.tensor_engine,
                )
                k[
                    :,
                    nl.ds(
                        load_idx * block_size * B_P_SIZE + tb_i * B_P_SIZE,
                        B_P_SIZE,
                    ),
                ] = nl.copy(k_t_psum, dtype=kernel_dtype)

    # load value cache
    for load_idx in nl.affine_range(num_loads):
        i_p = nl.arange(B_P_SIZE)[:, None]
        i_f = nl.arange(block_size * B_D_SIZE)[None, :]
        if kernel_dtype == value_cache.dtype:
            v_load_buffer[
                i_p,
                large_tile_idx % MULTI_BUFFER,
                i_f + load_idx * block_size * B_D_SIZE,
            ] = nl.load(
                value_cache[
                    block_tables_sbuf[i_p, load_idx, large_tile_idx],
                    i_f,
                ],
            )
        else:
            v_loaded = nl.ndarray(
                (B_P_SIZE, block_size * B_D_SIZE),
                dtype=value_cache.dtype,
            )
            v_loaded[...] = nl.load(
                value_cache[
                    block_tables_sbuf[i_p, load_idx, large_tile_idx],
                    i_f,
                ],
            )
            v_load_buffer[
                i_p,
                large_tile_idx % MULTI_BUFFER,
                i_f + load_idx * block_size * B_D_SIZE,
            ] = nl.copy(
                v_loaded,
                dtype=kernel_dtype,
            )
    # load next k
    if large_tile_idx + 1 < NUM_LARGE_TILE:
        for load_idx in nl.affine_range(num_loads):
            i_p = nl.arange(B_P_SIZE)[:, None]
            i_f = nl.arange(block_size * B_D_SIZE)[None, :]
            k_load_buffer[
                i_p, (large_tile_idx + 1) % MULTI_BUFFER, load_idx, i_f
            ] = nl.load(
                key_cache[
                    block_tables_sbuf[i_p, load_idx, large_tile_idx + 1],
                    i_f,
                ]
            )
    KV_TILE_SIZE = k.shape[1]
    assert (
        KV_TILE_SIZE % B_P_SIZE == 0
    ), f"{KV_TILE_SIZE=} not divisive by {B_P_SIZE=}"
    num_k_tile = KV_TILE_SIZE // B_P_SIZE

    kq_res_buf = nl.full(
        (par_dim(B_P_SIZE), num_k_tile, Q_TILE_SIZE),
        NEG_INF,
        dtype=acc_type,
    )
    kq_res_psum = nl.ndarray(
        (par_dim(B_P_SIZE), num_k_tile, Q_TILE_SIZE),
        dtype=nl.float32,
        buffer=nl.psum,
    )
    for k_i in nl.affine_range(num_k_tile):
        kq_res_psum[:, k_i, :] = nl.matmul(
            k[:, nl.ds(k_i * B_P_SIZE, B_P_SIZE)],
            q_local_tile[:, large_tile_idx],
            transpose_x=True,
        )  # (p(128), Q_TILE_SIZE)

    nisa.tensor_copy_predicated(
        src=kq_res_psum,
        dst=kq_res_buf,
        predicate=tile_mask[:, large_tile_idx],
    )

    max_partial = nl.ndarray((par_dim(B_P_SIZE), Q_TILE_SIZE), dtype=acc_type)
    # Calculate max of the current tile
    max_partial[:, :] = nisa.tensor_reduce(
        np.max,
        kq_res_buf,
        axis=(1,),
        dtype=acc_type,
        negate=False,
    )
    max_partial_transposed = nl.ndarray(
        (par_dim(Q_TILE_SIZE), B_P_SIZE), dtype=acc_type, buffer=nl.psum
    )
    PF_transpose_with_PE(
        src=max_partial,
        out=max_partial_transposed,
        identity_for_transpose=identity_for_transpose_m_step1,
        out_in_psum=True,
    )
    max_ = nisa.tensor_reduce(
        np.max,
        max_partial_transposed,
        axis=(1,),
        dtype=acc_type,
        negate=False,
        keepdims=True,
    )
    o_previous_scaled = nl.ndarray(
        (par_dim(B_D_SIZE), Q_TILE_SIZE),
        dtype=o_buffer_sbuf.dtype,
    )
    max_transposed = nl.ndarray(
        (par_dim(1), Q_TILE_SIZE),
        dtype=acc_type,
    )
    PF_transpose_with_PE(
        src=max_,
        out=max_transposed,
        identity_for_transpose=identity_for_transpose_m_step2,
    )

    m_previous = m_buffer_sbuf[:, large_tile_idx]
    m_current = nisa.tensor_tensor(
        m_previous,
        max_transposed,
        np.maximum,
    )  # (128,1)

    # Compute scaling factor
    bias = nl.zeros((par_dim(1), 1), dtype=m_previous.dtype)
    alpha = nisa.activation(
        np.exp,
        m_previous - m_current,
        scale=1.0,
        bias=bias,
    )
    alpha_broadcasted = nl.ndarray(
        (par_dim(B_D_SIZE), Q_TILE_SIZE),
        dtype=acc_type,
        buffer=nl.psum,
    )
    broadcast_partition_with_PE(alpha, alpha_broadcasted, out_in_psum=True)
    o_previous_scaled[...] = nl.multiply(
        o_buffer_sbuf[:, large_tile_idx],
        alpha_broadcasted,
    )

    max_broadcasted = nl.ndarray(
        (par_dim(B_P_SIZE), 1, Q_TILE_SIZE),
        dtype=acc_type,
        buffer=nl.psum,
    )
    broadcast_partition_with_PE(
        m_current,
        max_broadcasted[:, 0, :],
        out_in_psum=True,
    )

    p_local = nl.ndarray(
        (par_dim(B_P_SIZE), num_k_tile, Q_TILE_SIZE),
        dtype=kernel_dtype,
    )
    bias = nl.zeros((par_dim(B_P_SIZE), 1), dtype=max_broadcasted.dtype)
    p_local[...] = nisa.activation(
        np.exp,
        kq_res_buf - max_broadcasted,
        scale=1.0,
        bias=bias,
        dtype=kernel_dtype,
    )
    pv_psum = nl.zeros(
        (par_dim(B_D_SIZE), Q_TILE_SIZE),
        dtype=np.float32,
        buffer=nl.psum,
    )
    v = v_load_buffer[:, large_tile_idx % MULTI_BUFFER, :]
    for k_i in nl.affine_range(num_k_tile):
        pv_psum[:, :] += nl.matmul(
            v[:, nl.ds(k_i * B_D_SIZE, B_D_SIZE)],
            p_local[:, k_i],
            transpose_x=True,
        )
    ps_partial = nl.ndarray(
        (par_dim(1), num_k_tile, Q_TILE_SIZE),
        dtype=acc_type,
        buffer=nl.psum,
    )
    ps_partial_reshape = ps_partial.reshape((1, num_k_tile * Q_TILE_SIZE))
    ps_partial_reshape[...] = nisa.nc_matmul(
        sumexp_ones,
        p_local.reshape((B_P_SIZE, num_k_tile * Q_TILE_SIZE)),
        is_stationary_onezero=True,
    )
    ps = nl.ndarray(
        (par_dim(1), Q_TILE_SIZE),
        dtype=acc_type,
    )
    ps[...] = nisa.tensor_reduce(
        nl.add,
        ps_partial,
        axis=(1,),
        dtype=acc_type,
        negate=False,
        keepdims=False,
    )

    m_buffer_sbuf[:, large_tile_idx] = m_current
    o_buffer_sbuf[:, large_tile_idx, :] = nl.add(o_previous_scaled, pv_psum)

    l_prev = l_buffer_sbuf[:, large_tile_idx] * alpha
    l_buffer_sbuf[:, large_tile_idx] = l_prev + ps
    nisa.tensor_copy_predicated(
        src=m_buffer_sbuf[:, large_tile_idx],
        dst=m_buffer_sbuf[:, large_tile_idx + 1],
        predicate=q_update_pred_broadcast[nl.ds(0, 1), large_tile_idx],
    )
    nisa.tensor_copy_predicated(
        src=l_buffer_sbuf[:, large_tile_idx],
        dst=l_buffer_sbuf[:, large_tile_idx + 1],
        predicate=q_update_pred_broadcast[nl.ds(0, 1), large_tile_idx],
    )
    nisa.tensor_copy_predicated(
        src=o_buffer_sbuf[:, large_tile_idx],
        dst=o_buffer_sbuf[:, large_tile_idx + 1],
        predicate=q_update_pred_broadcast[:, large_tile_idx],
    )
