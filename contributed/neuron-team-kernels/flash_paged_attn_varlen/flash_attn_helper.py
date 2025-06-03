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

from utils import broadcast_partition_with_PE, PF_transpose_with_PE


def transpose_p_local(
    p_local_transposed,
    p_local,
    identity_for_transpose,
    Q_TILE_SIZE,
    LARGE_KV_TILE_SIZE,
    B_F_SIZE=512,
):
    B_P_SIZE = 128
    REDUCTION_SIZE = min(B_P_SIZE, LARGE_KV_TILE_SIZE)
    for i in nl.affine_range(LARGE_KV_TILE_SIZE // B_F_SIZE):
        if nisa.get_nc_version() == nisa.nc_version.gen3:
            p_local_t_tmp = nl.ndarray(
                (par_dim(REDUCTION_SIZE), B_F_SIZE // REDUCTION_SIZE * Q_TILE_SIZE),
                buffer=nl.sbuf,
                dtype=p_local.dtype,
            )
        else:
            p_local_t_tmp = nl.ndarray(
                (par_dim(REDUCTION_SIZE), B_F_SIZE // REDUCTION_SIZE * Q_TILE_SIZE),
                buffer=nl.psum,
                dtype=np.float32,
            )

        for j in nl.affine_range(B_F_SIZE // REDUCTION_SIZE):
            j_128_slice = nl.ds(j * Q_TILE_SIZE, Q_TILE_SIZE)
            i_j_128_slice = nl.ds(i * B_F_SIZE + j * REDUCTION_SIZE, REDUCTION_SIZE)

            if nisa.get_nc_version() == nisa.nc_version.gen3:
                p_local_t_tmp[:, j_128_slice] = nisa.dma_transpose(
                    p_local[:, i_j_128_slice]
                )
            else:
                p_local_t_tmp[:, j_128_slice] = nisa.nc_matmul(
                    p_local[:, i_j_128_slice],
                    identity_for_transpose,
                    is_moving_onezero=True,
                    is_transpose=True,
                )

        p_local_transposed[
            :,
            nl.ds(
                i * (B_F_SIZE // REDUCTION_SIZE * Q_TILE_SIZE),
                (B_F_SIZE // REDUCTION_SIZE * Q_TILE_SIZE),
            ),
        ] = nl.copy(p_local_t_tmp, dtype=p_local_transposed.dtype)


def _flash_attention_core(
    q_local_tile,
    k,
    v,
    o_buffer,
    l_buffer,
    m_buffer,
    kernel_dtype,
    acc_type,
    tile_mask,
    use_causal_mask,
    identity_for_transpose,
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
    o_buffer: (Q_TILE_SIZE, B_D_SIZE)
    l_buffer: (Q_TILE_SIZE, 1)
    m_buffer: (Q_TILE_SIZE, 1)
    """
    B_P_SIZE = 128
    assert (
        LARGE_KV_TILE_SIZE % B_P_SIZE == 0
    ), f"{LARGE_KV_TILE_SIZE=} not divisive by {B_P_SIZE=}"
    assert (
        LARGE_KV_TILE_SIZE % B_F_SIZE == 0
    ), f"{LARGE_KV_TILE_SIZE=} not divisive by {B_F_SIZE=}"
    num_k_tile_per_large_tile = LARGE_KV_TILE_SIZE // B_F_SIZE

    # mask are used to only apply computation to the lower half of the matrix,
    # which reduce the arithmetic intensity by half
    qk_res_buf = nl.ndarray(
        (par_dim(Q_TILE_SIZE), LARGE_KV_TILE_SIZE),
        buffer=nl.sbuf,
        dtype=acc_type,
    )
    max_local = nl.ndarray(
        (par_dim(Q_TILE_SIZE), num_k_tile_per_large_tile),
        dtype=acc_type,
    )
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
                (par_dim(Q_TILE_SIZE), B_F_SIZE), dtype=np.float32, buffer=nl.psum
            )  # (128, 512)
            qk_psum[:, :] = nl.matmul(
                q_local_tile, k[:, k_i_b_f_slice], transpose_x=True
            )  # (p(128), 512)
            qk_res_buf[:, k_i_b_f_slice] = nl.where(
                tile_mask[:, k_i_b_f_slice],
                qk_psum[:, nl.ds(0, B_F_SIZE)],
                -9984.0,
                dtype=acc_type,
            )
        else:
            qk_res_buf[:, k_i_b_f_slice] = -9984.0

        # Calculate max of the current tile
        max_local[:, k_i] = nisa.tensor_reduce(
            np.max,
            qk_res_buf[:, k_i_b_f_slice],
            axis=(1,),
            dtype=acc_type,
            negate=False,
        )

    max_ = nisa.tensor_reduce(
        np.max,
        max_local[:, :],
        axis=(1,),
        dtype=acc_type,
        negate=False,
    )

    o_previous_scaled = nl.ndarray(
        (par_dim(Q_TILE_SIZE), B_D_SIZE),
        dtype=o_buffer.dtype,
    )

    m_previous = nl.copy(m_buffer[:, 0])
    m_buffer[:, 0] = nl.maximum(m_previous, max_)  # (128,1)

    m_current = m_buffer[:, 0]
    # Compute scaling factor
    alpha = nisa.activation(
        np.exp,
        m_previous,
        bias=-1 * m_current,
        scale=1.0,
    )
    o_previous_scaled[...] = nl.multiply(o_buffer[:, :], alpha)

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
            bias=-1 * m_current,
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
        identity_for_transpose=identity_for_transpose,
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

    o_buffer[:, :] = nl.add(o_previous_scaled, pv_psum)

    l_prev = l_buffer[:, 0] * alpha
    l_buffer[:, 0] = l_prev + ps


def _flash_attention_core_kq_matmul(
    q_local_tile,
    o_buffer_sbuf,
    l_buffer_sbuf,
    m_buffer_sbuf,
    kernel_dtype,
    acc_type,
    tile_mask,
    q_update_pred_sbuf,
    q_update_pred_broadcast,
    key_cache,
    value_cache,
    block_tables_sbuf,
    large_tile_idx,
    MULTI_BUFFER_SIZE,
    num_blocks_per_large_tile,
    block_size,
    k_load_buffer,
    v_load_buffer,
    kq_res_ones,
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
    B_P_SIZE = 128
    B_D_SIZE, NUM_LARGE_TILE, Q_TILE_SIZE = q_local_tile.shape
    assert num_blocks_per_large_tile % B_P_SIZE == 0
    num_loads = num_blocks_per_large_tile // B_P_SIZE
    assert Q_TILE_SIZE <= B_P_SIZE, f"{Q_TILE_SIZE=} > {B_P_SIZE=}"
    k = k_load_buffer[:, large_tile_idx % MULTI_BUFFER_SIZE]
    v = v_load_buffer[:, large_tile_idx % MULTI_BUFFER_SIZE]

    KV_TILE_SIZE = k.shape[1]
    assert KV_TILE_SIZE % B_P_SIZE == 0, f"{KV_TILE_SIZE=} not divisive by {B_P_SIZE=}"
    num_k_tile = KV_TILE_SIZE // B_P_SIZE

    kq_res_buf = nl.ndarray(
        (par_dim(B_P_SIZE), num_k_tile, Q_TILE_SIZE),
        dtype=acc_type,
    )
    kq_res_buf[...] = kq_res_ones * (-9984.0)
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
    if large_tile_idx + 1 < NUM_LARGE_TILE:
        k_load_buffer_reshaped = k_load_buffer.reshape(
            (B_D_SIZE, MULTI_BUFFER_SIZE, num_loads, block_size, B_P_SIZE),
        )
        for load_idx in nl.affine_range(num_loads):
            i_p = nl.arange(B_P_SIZE)[:, None]
            i_f = nl.arange(block_size * B_D_SIZE)[None, :]
            loaded = nl.ndarray(
                (B_P_SIZE, block_size * B_D_SIZE), dtype=key_cache.dtype
            )
            loaded[i_p, i_f] = nl.load(
                key_cache[block_tables_sbuf[i_p, load_idx, large_tile_idx + 1], i_f]
            )
            for tb_i in nl.affine_range(block_size):
                if loaded.dtype != kernel_dtype:
                    k_src = nl.copy(
                        loaded[:, nl.ds(tb_i * B_D_SIZE, B_D_SIZE)],
                        dtype=kernel_dtype,
                    )
                else:
                    k_src = loaded[:, nl.ds(tb_i * B_D_SIZE, B_D_SIZE)]
                PF_transpose_with_PE(
                    src=k_src,
                    out=k_load_buffer_reshaped[
                        :,
                        (large_tile_idx + 1) % MULTI_BUFFER_SIZE,
                        load_idx,
                        tb_i,
                    ],
                    identity_for_transpose=identity_for_transpose_k,
                )

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
    m_current = nisa.tensor_tensor(m_previous, max_transposed, np.maximum)  # (128,1)

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
        o_buffer_sbuf[:, large_tile_idx], alpha_broadcasted
    )

    max_broadcasted = nl.ndarray(
        (par_dim(B_P_SIZE), Q_TILE_SIZE, 1),
        dtype=acc_type,
        buffer=nl.psum,
    )
    broadcast_partition_with_PE(m_current, max_broadcasted[:, :, 0], out_in_psum=True)

    ps = nl.zeros(
        (par_dim(1), Q_TILE_SIZE),
        dtype=acc_type,
        buffer=nl.psum,
    )
    p_local = nl.ndarray(
        (par_dim(B_P_SIZE), num_k_tile, Q_TILE_SIZE),
        dtype=kernel_dtype,
    )
    bias = nl.zeros((par_dim(B_P_SIZE), 1), dtype=max_broadcasted.dtype)
    for k_r_i in nl.affine_range(num_k_tile):
        p_local[:, k_r_i, :] = nisa.activation(
            np.exp,
            kq_res_buf[:, k_r_i, :] - max_broadcasted,
            scale=1.0,
            bias=bias,
            dtype=kernel_dtype,
        )
        ps[:, :] += nisa.nc_matmul(
            sumexp_ones,
            p_local[:, k_r_i, :],
            is_stationary_onezero=True,
        )

    pv_psum = nl.zeros(
        (par_dim(B_D_SIZE), Q_TILE_SIZE),
        dtype=np.float32,
        buffer=nl.psum,
    )
    for k_i in nl.affine_range(num_k_tile):
        pv_psum[:, :] += nl.matmul(
            v[:, nl.ds(k_i * B_D_SIZE, B_D_SIZE)],
            p_local[:, k_i],
            transpose_x=True,
        )
    if large_tile_idx + 1 < NUM_LARGE_TILE:
        # load value cache
        for load_idx in nl.affine_range(num_loads):
            i_p = nl.arange(B_P_SIZE)[:, None]
            i_f = nl.arange(block_size * B_D_SIZE)[None, :]
            if kernel_dtype == value_cache.dtype:
                v_load_buffer[
                    i_p,
                    (large_tile_idx + 1) % MULTI_BUFFER_SIZE,
                    i_f + load_idx * block_size * B_D_SIZE,
                ] = nl.load(
                    value_cache[
                        block_tables_sbuf[i_p, load_idx, large_tile_idx + 1], i_f
                    ],
                )
            else:
                loaded = nl.ndarray(
                    (B_P_SIZE, block_size * B_D_SIZE), dtype=value_cache.dtype
                )
                loaded[...] = nl.load(
                    value_cache[
                        block_tables_sbuf[i_p, load_idx, large_tile_idx + 1], i_f
                    ],
                )
                v_load_buffer[
                    i_p,
                    (large_tile_idx + 1) % MULTI_BUFFER_SIZE,
                    i_f + load_idx * block_size * B_D_SIZE,
                ] = nl.copy(
                    loaded,
                    dtype=kernel_dtype,
                )

    m_buffer_sbuf[:, large_tile_idx] = m_current
    o_buffer_sbuf[:, large_tile_idx, :] = nl.add(o_previous_scaled, pv_psum)

    l_prev = l_buffer_sbuf[:, large_tile_idx] * alpha
    l_buffer_sbuf[:, large_tile_idx] = l_prev + ps
    nisa.tensor_copy_predicated(
        src=m_buffer_sbuf[:, large_tile_idx],
        dst=m_buffer_sbuf[:, large_tile_idx + 1],
        predicate=q_update_pred_sbuf[:, large_tile_idx],
    )
    nisa.tensor_copy_predicated(
        src=l_buffer_sbuf[:, large_tile_idx],
        dst=l_buffer_sbuf[:, large_tile_idx + 1],
        predicate=q_update_pred_sbuf[:, large_tile_idx],
    )
    nisa.tensor_copy_predicated(
        src=o_buffer_sbuf[:, large_tile_idx],
        dst=o_buffer_sbuf[:, large_tile_idx + 1],
        predicate=q_update_pred_broadcast[:, large_tile_idx],
    )


def _active_attention_core_batched(
    q,
    k,
    v,
    o_buffer,
    l_buffer,
    m_buffer,
    q_h_per_k_h,
    kernel_dtype,
    acc_type,
    TILE_SIZE=128,
    B_D_SIZE=128,
):
    """
    The flash attention core function to calculate self attention between a tile
    of q and a block of K and V.
    q: (par_dim(B_D_SIZE), q_h_per_k_h, TILE_SIZE)
    k: (par_dim(B_D_SIZE), TILE_SIZE)
    v: (par_dim(TILE_SIZE), B_D_SIZE)
    The results are stored in the following three buffers
    o_buffer: (par_dim(TILE_SIZE), q_h_per_k_h, B_D_SIZE)
    l_buffer: (par_dim(TILE_SIZE), q_h_per_k_h, 1)
    m_buffer: (par_dim(TILE_SIZE), q_h_per_k_h, 1)
    """
    qk_psum = nl.ndarray(
        (par_dim(TILE_SIZE), q_h_per_k_h, 1),
        buffer=nl.psum,
        dtype=np.float32,
    )
    ones = nl.ones((par_dim(B_D_SIZE), 1), dtype=acc_type)
    for i_q_h in nl.affine_range(q_h_per_k_h):
        qk_mul = nl.ndarray(
            (par_dim(B_D_SIZE), TILE_SIZE),
            dtype=acc_type,
        )
        qk_mul[...] = nl.multiply(q[:, i_q_h], k, dtype=qk_mul.dtype)
        qk_psum[:, i_q_h, :] = nisa.nc_matmul(
            qk_mul,
            ones,
            is_moving_onezero=True,
        )

    o_previous_scaled = nl.ndarray(
        (par_dim(TILE_SIZE), q_h_per_k_h, B_D_SIZE), dtype=o_buffer.dtype
    )
    max_ = qk_psum

    m_previous = nl.ndarray((par_dim(TILE_SIZE), q_h_per_k_h, 1), dtype=acc_type)
    m_previous[...] = nl.copy(m_buffer)
    m_buffer[...] = nl.maximum(m_previous, max_)

    alpha = nl.ndarray((par_dim(TILE_SIZE), q_h_per_k_h, 1), dtype=acc_type)
    bias = nl.zeros((par_dim(TILE_SIZE), 1), dtype=m_previous.dtype)
    alpha[...] = nisa.activation(
        np.exp,
        m_previous - m_buffer,
        scale=1.0,
        bias=bias,
    )
    o_previous_scaled[...] = nl.multiply(o_buffer, alpha)

    p_local = nl.ndarray((par_dim(TILE_SIZE), q_h_per_k_h, 1), dtype=kernel_dtype)
    ps = nl.ndarray(
        (par_dim(TILE_SIZE), q_h_per_k_h, 1),
        dtype=acc_type,
    )
    for i_q_h in nl.affine_range(q_h_per_k_h):
        p_local[:, i_q_h, :] = nisa.activation_reduce(
            np.exp,
            qk_psum[:, i_q_h, :],
            bias=-1 * m_buffer[:, i_q_h, :],
            scale=1.0,
            reduce_op=nl.add,
            reduce_res=ps[:, i_q_h, :],
            dtype=kernel_dtype,
        )

    pv_sbuf = nl.ndarray(
        (par_dim(TILE_SIZE), q_h_per_k_h, B_D_SIZE),
        dtype=np.float32,
        buffer=nl.sbuf,
    )
    pv_sbuf[...] = nisa.tensor_tensor(p_local, v, nl.multiply, dtype=np.float32)

    o_buffer[...] = nl.add(o_previous_scaled, pv_sbuf)

    l_prev = l_buffer * alpha
    l_buffer[...] = l_prev + ps
