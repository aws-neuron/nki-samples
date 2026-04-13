"""
Copyright (C) 2026, Amazon.com. All Rights Reserved

NKI implementations for forward pass of attention. Each
subsequent implementation uses NKI functions to get better
hardware performance for attention.

"""

import numpy as np
import nki
import nki.isa as nisa
from attention_kernel_utils import softmax_isa
import nki.language as nl
import logging



####################################################################
# v1: toy example with 128 seqlen and nki.lang APIs
####################################################################
@nki.jit
def attn_fwd_v1(q, k, v):
    """nki.lang APIs"""
    d_head, seqlen_q = q.shape
    seqlen_kv = seqlen_q

    assert q.shape == k.shape == v.shape
    assert d_head == 128
    assert seqlen_q == 128

    kernel_out = nl.ndarray((seqlen_q, d_head), dtype=q.dtype, buffer=nl.shared_hbm)

    # load inputs into SBUF:
    q_sbuf = nl.load(q)
    k_sbuf = nl.load(k)
    v_sbuf = nl.load(v)

    # Q @ K, contract along d_head #
    qk_psum = nl.matmul(x=q_sbuf, y=k_sbuf, transpose_x=True)

    # Move QK result to SBUF for softmax operations
    qk = nl.ndarray(qk_psum.shape, dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=qk, src=qk_psum)

    # Softmax #
    # reduce max along seqlen_k
    row_max = nl.max(qk, axis=1, keepdims=True)

    # subtract max from row
    norm_row = nl.ndarray(qk.shape, dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=norm_row, data=qk, op0=nl.subtract, operand0=row_max)

    # exponentiation
    exp_row = nl.exp(norm_row)

    # sum of exp results
    sum_row = nl.sum(exp_row, axis=1, keepdims=True)

    # divide exp results by sum
    inverse_sum_row = nl.reciprocal(sum_row)
    scores = nl.ndarray(exp_row.shape, dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=scores, data=exp_row, op0=nl.multiply, operand0=inverse_sum_row)

    # v has the wrong layout
    v_psum_t = nl.transpose(v_sbuf)
    v_sbuf_t = nl.ndarray(v_psum_t.shape, dtype=v_sbuf.dtype, buffer=nl.sbuf)
    nisa.tensor_copy(dst=v_sbuf_t, src=v_psum_t)

    # scores @ V, contract along seqlen_kv
    # nl.matmul with transpose_x=False internally transposes to PSUM which
    # can't be used as stationary on gen3+. Pre-transpose scores instead.
    scores_t_psum = nl.transpose(scores)
    scores_t = nl.ndarray(scores_t_psum.shape, dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=scores_t, src=scores_t_psum)
    attn_out = nl.matmul(scores_t, v_sbuf_t, transpose_x=True)

    # store output
    attn_out_sbuf = nl.ndarray(attn_out.shape, dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=attn_out_sbuf, src=attn_out)
    nl.store(dst=kernel_out, value=attn_out_sbuf)
    return kernel_out

####################################################################
# v2: use nki.isa APIs
####################################################################
@nki.jit
def attn_fwd_v2(q, k, v):
    """ISA-level attention with explicit dst allocation."""
    d_head, seqlen_q = q.shape
    seqlen_kv = seqlen_q
    assert q.shape == k.shape == v.shape
    assert d_head == 128
    assert seqlen_q == 128

    kernel_out = nl.ndarray((seqlen_q, d_head), dtype=q.dtype, buffer=nl.shared_hbm)

    # Load inputs
    q_sbuf = nl.ndarray(q.shape, dtype=q.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=q_sbuf, src=q)
    k_sbuf = nl.ndarray(k.shape, dtype=k.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=k_sbuf, src=k)
    v_sbuf = nl.ndarray(v.shape, dtype=v.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=v_sbuf, src=v)

    # Q^T @ K -> PSUM -> SBUF
    qk = nl.ndarray((seqlen_q, seqlen_kv), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=qk, stationary=q_sbuf, moving=k_sbuf)
    qk_sbuf = nl.ndarray(qk.shape, dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=qk_sbuf, src=qk)

    # Softmax
    scores = softmax_isa(qk_sbuf)

    # Transpose scores and v for matmul
    scores_t_psum = nl.ndarray((seqlen_kv, seqlen_q), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_transpose(dst=scores_t_psum, data=scores)
    scores_t = nl.ndarray(scores_t_psum.shape, dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=scores_t, src=scores_t_psum)

    v_t_psum = nl.ndarray((seqlen_kv, d_head), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_transpose(dst=v_t_psum, data=v_sbuf)
    v_t = nl.ndarray(v_t_psum.shape, dtype=v_sbuf.dtype, buffer=nl.sbuf)
    nisa.tensor_copy(dst=v_t, src=v_t_psum)

    # scores @ V^T
    attn_out = nl.ndarray((seqlen_q, d_head), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=attn_out, stationary=scores_t, moving=v_t)

    # PSUM -> SBUF -> HBM
    attn_out_sbuf = nl.ndarray(attn_out.shape, dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=attn_out_sbuf, src=attn_out)
    nisa.dma_copy(dst=kernel_out, src=attn_out_sbuf)

    return kernel_out

####################################################################
# v3: large sequence length with tiling
####################################################################
@nki.jit
def attn_fwd_v3(q, k, v):
    d_head, seqlen_q = q.shape
    seqlen_kv = seqlen_q

    PMAX = nl.tile_size.pmax
    FMAX_STATIONARY = nl.tile_size.gemm_stationary_fmax
    FMAX_MOVING = nl.tile_size.gemm_moving_fmax

    assert q.shape == k.shape == v.shape
    assert d_head == PMAX
    assert seqlen_q >= 512

    kernel_out = nl.ndarray((seqlen_q, d_head), dtype=q.dtype, buffer=nl.shared_hbm)

    # load inputs into SBUF
    q_sbuf = nl.load(q)
    k_sbuf = nl.load(k)
    v_sbuf = nl.load(v)

    # Tile along seqlen_q #
    # for this example we assume that seqlen_q is divisible by PMAX and 
    # seqlen_kv is divisible by FMAX_MOVING, otherwise need to use mask or "final multiplication"
    qk = nl.ndarray((seqlen_q // PMAX, seqlen_kv // FMAX_MOVING, PMAX, FMAX_MOVING),
                     dtype=nl.float32, buffer=nl.shared_hbm)
    for i_tile_q in nl.affine_range(seqlen_q // FMAX_STATIONARY): # loop on stationary_free
        for i_tile_kv in nl.affine_range(seqlen_kv // FMAX_MOVING): # loop on moving_free
            # Q @ K, contract along d_head #
            qk_psum = nl.ndarray((PMAX, FMAX_MOVING), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=qk_psum, 
                stationary=q_sbuf[0:PMAX, nl.ds(i_tile_q*FMAX_STATIONARY, FMAX_STATIONARY)],
                moving=k_sbuf[0:PMAX, nl.ds(i_tile_kv*FMAX_MOVING, FMAX_MOVING)])
            qk_sbuf = nl.ndarray((PMAX, FMAX_MOVING), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=qk_sbuf, src=qk_psum)
            nisa.dma_copy(dst=qk[i_tile_q, i_tile_kv, :, :], src=qk_sbuf)

    # Softmax #
    # reduce max along seqlen_k
    row_max = nl.ndarray((PMAX, seqlen_q // PMAX), dtype=nl.float32, buffer=nl.sbuf)
    for i_tile_q in nl.affine_range(seqlen_q // PMAX):

        row_max_kv = nl.ndarray((PMAX, seqlen_kv // FMAX_MOVING), dtype=nl.float32, buffer=nl.sbuf)
        for i_tile_kv in nl.affine_range(seqlen_kv // FMAX_MOVING):
            qk_tile = nl.ndarray((PMAX, FMAX_MOVING), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=qk_tile, src=qk[i_tile_q, i_tile_kv, :, :])
            nisa.tensor_reduce(dst=row_max_kv[:, nl.ds(i_tile_kv, 1)], op=nl.maximum, data=qk_tile, axis=(1,))
 
        nisa.tensor_reduce(dst=row_max[:, nl.ds(i_tile_q, 1)], op=nl.maximum, data=row_max_kv[:, :], axis=(1,))

    # subtract max from row
    norm_row = nl.ndarray((seqlen_q // PMAX, PMAX, seqlen_kv),
                       dtype=nl.float32, buffer=nl.shared_hbm)
    for i_tile_q in nl.affine_range(seqlen_q // PMAX):
        norm_buf = nl.ndarray(shape=(PMAX, seqlen_kv), dtype=nl.float32, buffer=nl.sbuf)
        for i_tile_kv in nl.affine_range(seqlen_kv // FMAX_MOVING):
            qk_tile_sub = nl.ndarray((PMAX, FMAX_MOVING), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=qk_tile_sub, src=qk[i_tile_q, i_tile_kv, :, :])
            nisa.tensor_scalar(dst=norm_buf[:, nl.ds(i_tile_kv*FMAX_MOVING, FMAX_MOVING)], 
                data=qk_tile_sub,
                op0=nl.subtract,
                operand0=row_max[:, nl.ds(i_tile_q, 1)])
        nl.store(norm_row[i_tile_q], norm_buf[:,:])

    # exponentiation
    exp_row = nl.ndarray((seqlen_q // PMAX, PMAX, seqlen_kv), dtype=nl.float32, buffer=nl.shared_hbm)
    for i_tile_q in nl.affine_range(seqlen_q // PMAX):
        # norm_buf = nl.ndarray(shape=(PMAX, seqlen_kv), dtype=nl.float32, buffer=nl.sbuf)
        exp_buf = nl.ndarray(shape=(PMAX, seqlen_kv), dtype=nl.float32, buffer=nl.sbuf)
        norm_buf = nl.load(norm_row[i_tile_q])
        nisa.activation(dst=exp_buf[:,:], op=nl.exp, data=norm_buf)
        nl.store(exp_row[i_tile_q], exp_buf[:,:])

    # sum of exp results
    sum_row = nl.ndarray((PMAX, seqlen_q // PMAX), dtype=nl.float32, buffer=nl.sbuf)
    for i_tile_q in nl.affine_range(seqlen_q // PMAX):
        exp_buf = nl.load(exp_row[i_tile_q])
        nisa.tensor_reduce(dst=sum_row[:, nl.ds(i_tile_q, 1)], op=nl.add,
                                                         data=exp_buf,
                                                         axis=(1,))

    # reciprocal of sum_row, tile shape is [PMAX, seqlen_q // PMAX]
    inverse_sum_row = nl.ndarray(sum_row.shape, dtype=nl.float32, buffer=nl.sbuf)
    nisa.reciprocal(dst=inverse_sum_row, data=sum_row)
    
    scores = nl.ndarray((seqlen_q // PMAX, PMAX, seqlen_kv), dtype=nl.float32, buffer=nl.shared_hbm)
    for i_tile_q in nl.affine_range(seqlen_q // PMAX):
        scores_buf = nl.ndarray(shape=(PMAX, seqlen_kv), dtype=nl.float32, buffer=nl.sbuf)
        exp_buf = nl.load(exp_row[i_tile_q])
        nisa.tensor_scalar(dst=scores_buf[:,:], data=exp_buf,
                                               op0=nl.multiply,
                                               operand0=inverse_sum_row[:, i_tile_q])
        nl.store(scores[i_tile_q], scores_buf[:,:])
        
    # v has the wrong layout
    v_t = nl.ndarray((seqlen_kv // PMAX, PMAX, d_head), dtype=nl.float32, buffer=nl.shared_hbm)
    for i_tile_kv in nl.affine_range(seqlen_kv // PMAX):
        v_psum_t = nl.ndarray((PMAX, d_head), dtype=v_sbuf.dtype, buffer=nl.psum)
        v_psum_t = nl.ndarray((PMAX, d_head), dtype=v_sbuf.dtype, buffer=nl.psum)
        nisa.nc_transpose(dst=v_psum_t, data=v_sbuf[:, nl.ds(i_tile_kv*PMAX, PMAX)]) # TensorE
        v_sbuf_t = nl.ndarray((PMAX, d_head), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=v_sbuf_t[:, :], src=v_psum_t)                   # ScalarE
        nl.store(v_t[i_tile_kv], v_sbuf_t[:,:])

    # scores has the wrong layout
    # PMAX restriction on both free and partition dimension when performing transpose.
    # scores_sbuf_t = nl.ndarray((PMAX, seqlen_kv // PMAX, seqlen_q // PMAX, PMAX),
    #                            dtype=nl.float32, buffer=nl.sbuf)
    scores_t = nl.ndarray((seqlen_kv // PMAX, seqlen_q // PMAX, PMAX, PMAX), dtype=nl.float32, buffer=nl.shared_hbm)
    for i_tile_q in nl.affine_range(seqlen_q // PMAX):
        for i_tile_kv in nl.affine_range(seqlen_kv // PMAX):
            scores_buf = nl.load(scores[i_tile_q, :, nl.ds(i_tile_kv*PMAX, PMAX)])
            scores_psum_t = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_transpose(dst=scores_psum_t, data=scores_buf) # TensorE
            scores_sbuf_t = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=scores_sbuf_t[:, :], src=scores_psum_t)    # ScalarE
            nl.store(scores_t[i_tile_kv, i_tile_q, :, :], scores_sbuf_t)

    # scores @ V, contract along seqlen_kv
    # d_head == P_MAX, no need to tile there
    for i_tile_q in nl.affine_range(seqlen_q // PMAX): # loop on stationary free
        attn_out_psum = nl.zeros((PMAX, PMAX), dtype=nl.float32, buffer=nl.psum)
        attn_out = nl.ndarray((PMAX, d_head),
                           dtype=nl.float32, buffer=nl.sbuf)
        for i_tile_kv in nl.affine_range(seqlen_kv // PMAX): # loop on contraction
            scores_sbuf_t = nl.load(scores_t[i_tile_kv, i_tile_q, :, :])
            v_sbuf_t = nl.load(v_t[i_tile_kv, :, :])
            nisa.nc_matmul(dst=attn_out_psum, stationary=scores_sbuf_t,
                                            moving=v_sbuf_t)
        nisa.tensor_copy(dst=attn_out, src=attn_out_psum)
        nl.store(dst=kernel_out[nl.ds(i_tile_q*PMAX, PMAX), :], value=attn_out[:,:])

    return kernel_out


####################################################################
# v4: Loop fusion
# combines QK matrix multiplication, all softmax steps, and V 
# multiplication to compute attention scores & output under one 
# common loop.
####################################################################
@nki.jit
def attn_fwd_v4(q, k, v):
    d_head, seqlen_q = q.shape
    seqlen_kv = seqlen_q

    PMAX = nl.tile_size.pmax
    FMAX_MOVING = nl.tile_size.gemm_moving_fmax
    FMAX_STATIONARY = nl.tile_size.gemm_stationary_fmax

    assert q.shape == k.shape == v.shape
    assert d_head == PMAX
    assert seqlen_q >= 512

    kernel_out = nl.ndarray((seqlen_q, d_head), dtype=q.dtype, buffer=nl.shared_hbm)

    # load inputs into SBUF:
    q_sbuf = nl.load(q)
    k_sbuf = nl.load(k)
    v_sbuf = nl.load(v)

    # v has the wrong layout
    v_sbuf_t = nl.ndarray((PMAX, seqlen_kv // PMAX, d_head), dtype=nl.float32, buffer=nl.sbuf)
    for i_tile_kv in nl.affine_range(seqlen_kv // PMAX):
        v_psum_t = nl.ndarray((PMAX, d_head), dtype=v_sbuf.dtype, buffer=nl.psum)
        nisa.nc_transpose(dst=v_psum_t, data=v_sbuf[:, nl.ds(i_tile_kv*PMAX, PMAX)]) # TensorE
        nisa.tensor_copy(dst=v_sbuf_t[:, i_tile_kv, :], src=v_psum_t)     # ScalarE

    num_kv_tiles = seqlen_kv // FMAX_MOVING

    # Tile along seqlen_q #
    for i_tile_q in nl.affine_range(seqlen_q // FMAX_STATIONARY): # loop on stationary_free
        # per i_tile_q we finish a partial block matrix for qk
        # Allocate fresh PSUM tiles each iteration to avoid accumulation
        qk_tiles = []
        for _i in range(num_kv_tiles):
            qk_tiles.append(nl.ndarray((PMAX, FMAX_MOVING), dtype=nl.float32, buffer=nl.psum))

        for i_tile_kv in range(num_kv_tiles): # loop on moving_free
            # Q @ K, contract along d_head #
            nisa.nc_matmul(dst=qk_tiles[i_tile_kv], 
                stationary=q_sbuf[0:PMAX, nl.ds(i_tile_q*FMAX_STATIONARY, FMAX_STATIONARY)],
                moving=k_sbuf[0:PMAX, nl.ds(i_tile_kv*FMAX_MOVING, FMAX_MOVING)])

        # Softmax #
        # reduce max along seqlen_k
        row_max = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)

        row_max_kv = nl.ndarray((PMAX, num_kv_tiles), dtype=nl.float32, buffer=nl.sbuf)
        for i_tile_kv in range(num_kv_tiles):
            nisa.tensor_reduce(dst=row_max_kv[:, nl.ds(i_tile_kv, 1)], op=nl.maximum, data=qk_tiles[i_tile_kv], axis=(1,))

        nisa.tensor_reduce(dst=row_max[:, :], op=nl.maximum, data=row_max_kv[:, :], axis=(1,))

        # subtract max from row
        norm_row = nl.ndarray((PMAX, seqlen_kv),
                            dtype=nl.float32, buffer=nl.sbuf)
        for i_tile_kv in range(num_kv_tiles):
            nisa.tensor_scalar(dst=norm_row[:, nl.ds(i_tile_kv*FMAX_MOVING, FMAX_MOVING)], 
                data=qk_tiles[i_tile_kv],
                op0=nl.subtract,
                operand0=row_max)

        # exponentiation
        exp_row = nl.ndarray((PMAX, seqlen_kv), dtype=nl.float32, buffer=nl.sbuf)
        for i_tile_kv in nl.affine_range(seqlen_kv // FMAX_MOVING):
            nisa.activation(dst=exp_row[:, nl.ds(i_tile_kv*FMAX_MOVING, FMAX_MOVING)], 
                op=nl.exp, data=norm_row[:, nl.ds(i_tile_kv*FMAX_MOVING, FMAX_MOVING)])

        # sum of exp results
        sum_row_kv = nl.ndarray((PMAX, seqlen_kv // FMAX_MOVING), dtype=nl.float32, buffer=nl.sbuf)
        for i_tile_kv in nl.affine_range(seqlen_kv // FMAX_MOVING):
            nisa.tensor_reduce(dst=sum_row_kv[:, nl.ds(i_tile_kv, 1)], 
                op=nl.add,
                data=exp_row[:, nl.ds(i_tile_kv*FMAX_MOVING, FMAX_MOVING)], axis=(1,))

        sum_row = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=sum_row, op=nl.add, data=sum_row_kv, axis=(1,))

        # reciprocal of sum_row, tile shape is [PMAX, 1]
        # has reciprocals of 128 rows at a time, akin to the block of
        # output each q-tile is responsible for.
        inverse_sum_row = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.reciprocal(dst=inverse_sum_row, data=sum_row)

        scores = nl.ndarray((PMAX, seqlen_kv), dtype=nl.float32, buffer=nl.sbuf)
        for i_tile_kv in nl.affine_range(seqlen_kv // FMAX_MOVING):
            nisa.tensor_scalar(dst=scores[:, nl.ds(i_tile_kv*FMAX_MOVING, FMAX_MOVING)], 
                data=exp_row[:, nl.ds(i_tile_kv*FMAX_MOVING, FMAX_MOVING)],
                op0=nl.multiply,
                operand0=inverse_sum_row)

        # scores has the wrong layout
        scores_sbuf_t = nl.ndarray((PMAX, seqlen_kv // PMAX, PMAX),
                                    dtype=nl.float32, buffer=nl.sbuf)
        for i_tile_kv in nl.affine_range(seqlen_kv // PMAX):
            scores_psum_t = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_transpose(dst=scores_psum_t, data=scores[:, nl.ds(i_tile_kv*PMAX, PMAX)]) # TensorE
            nisa.tensor_copy(dst=scores_sbuf_t[:, i_tile_kv, :], src=scores_psum_t)    # ScalarE

        # scores @ V, contract along seqlen_kv
        attn_out = nl.ndarray((PMAX, PMAX),
                            dtype=nl.float32, buffer=nl.sbuf)

        attn_out_psum = nl.zeros((PMAX, PMAX), dtype=nl.float32, buffer=nl.psum)

        for i_tile_kv in nl.affine_range(seqlen_kv // PMAX): # loop on contraction
            nisa.nc_matmul(dst=attn_out_psum, stationary=scores_sbuf_t[:, i_tile_kv, :],
                                            moving=v_sbuf_t[:, i_tile_kv, :])
        nisa.tensor_copy(dst=attn_out, src=attn_out_psum) # store output
        nl.store(dst=kernel_out[nl.ds(i_tile_q*PMAX, PMAX), :], value=attn_out[:, :])

    return kernel_out


####################################################################
# v5: softmax division delay
####################################################################
@nki.jit
def attn_fwd_v5(q, k, v):
    d_head, seqlen_q = q.shape
    seqlen_kv = seqlen_q

    PMAX = nl.tile_size.pmax
    FMAX_MOVING = nl.tile_size.gemm_moving_fmax
    FMAX_STATIONARY = nl.tile_size.gemm_stationary_fmax

    assert q.shape == k.shape == v.shape
    assert d_head == PMAX
    assert seqlen_q >= 512

    kernel_out = nl.ndarray((seqlen_q, d_head), dtype=q.dtype, buffer=nl.shared_hbm)

    # load inputs into SBUF:
    q_sbuf = nl.load(q)
    k_sbuf = nl.load(k)
    v_sbuf = nl.load(v)

    # v has the wrong layout
    v_sbuf_t = nl.ndarray((PMAX, seqlen_kv // PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf)
    for i_tile_kv in nl.affine_range(seqlen_kv // PMAX):
        v_psum_t = nl.ndarray((PMAX, d_head), dtype=v_sbuf.dtype, buffer=nl.psum)
        nisa.nc_transpose(dst=v_psum_t, data=v_sbuf[:, nl.ds(i_tile_kv*PMAX, PMAX)]) # TensorE
        nisa.tensor_copy(dst=v_sbuf_t[:, i_tile_kv, :], src=v_psum_t)     # ScalarE

    num_kv_tiles = seqlen_kv // FMAX_MOVING

    # Tile along seqlen_q #
    for i_tile_q in nl.affine_range(seqlen_q // FMAX_STATIONARY): # loop on stationary_free
        # Allocate fresh PSUM tiles each iteration to avoid accumulation
        qk_tiles = []
        for _i in range(num_kv_tiles):
            qk_tiles.append(nl.ndarray((PMAX, FMAX_MOVING), dtype=nl.float32, buffer=nl.psum))

        for i_tile_kv in range(num_kv_tiles): # loop on moving_free
            # Q @ K, contract along d_head #
            nisa.nc_matmul(dst=qk_tiles[i_tile_kv], 
                stationary=q_sbuf[0:PMAX, nl.ds(i_tile_q*FMAX_STATIONARY, FMAX_STATIONARY)],
                moving=k_sbuf[0:PMAX, nl.ds(i_tile_kv*FMAX_MOVING, FMAX_MOVING)])

        # Softmax #
        # reduce max along seqlen_k
        row_max = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)

        row_max_kv = nl.ndarray((PMAX, num_kv_tiles), dtype=nl.float32, buffer=nl.sbuf)
        for i_tile_kv in range(num_kv_tiles):
            nisa.tensor_reduce(dst=row_max_kv[:, nl.ds(i_tile_kv, 1)], op=nl.maximum, data=qk_tiles[i_tile_kv], axis=(1,))

        nisa.tensor_reduce(dst=row_max[:, :], op=nl.maximum, data=row_max_kv[:, :], axis=(1,))

        # subtract max from row
        norm_row = nl.ndarray((PMAX, seqlen_kv),
                            dtype=nl.float32, buffer=nl.sbuf)
        for i_tile_kv in range(num_kv_tiles):
            nisa.tensor_scalar(dst=norm_row[:, nl.ds(i_tile_kv*FMAX_MOVING, FMAX_MOVING)], 
                data=qk_tiles[i_tile_kv],
                op0=nl.subtract,
                operand0=row_max)

        # exponentiation
        exp_row = nl.ndarray((PMAX, seqlen_kv), dtype=nl.float32, buffer=nl.sbuf)
        for i_tile_kv in nl.affine_range(seqlen_kv // FMAX_MOVING):
            nisa.activation(dst=exp_row[:, nl.ds(i_tile_kv*FMAX_MOVING, FMAX_MOVING)], 
                op=nl.exp, data=norm_row[:, nl.ds(i_tile_kv*FMAX_MOVING, FMAX_MOVING)])

        # sum of exp results
        sum_row_kv = nl.ndarray((PMAX, seqlen_kv // FMAX_MOVING), dtype=nl.float32, buffer=nl.sbuf)
        for i_tile_kv in nl.affine_range(seqlen_kv // FMAX_MOVING):
            nisa.tensor_reduce(dst=sum_row_kv[:, nl.ds(i_tile_kv, 1)], 
                op=nl.add,
                data=exp_row[:, nl.ds(i_tile_kv*FMAX_MOVING, FMAX_MOVING)], axis=(1,))

        sum_row = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=sum_row, op=nl.add, data=sum_row_kv, axis=(1,))

        # reciprocal of sum_row, tile shape is [PMAX, 1]
        inverse_sum_row = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.reciprocal(dst=inverse_sum_row, data=sum_row)

        # CHANGE OF LOGIC COMPARED TO attn_fwd_v4, here we delay the division

        # scores has the wrong layout
        scores_sbuf_t = nl.ndarray((PMAX, seqlen_kv // PMAX, PMAX),
                                    dtype=nl.float32, buffer=nl.sbuf)
        for i_tile_kv in nl.affine_range(seqlen_kv // PMAX):
            scores_psum_t = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_transpose(dst=scores_psum_t, data=exp_row[:, nl.ds(i_tile_kv*PMAX, PMAX)]) # TensorE
            nisa.tensor_copy(dst=scores_sbuf_t[:, i_tile_kv, :], src=scores_psum_t)    # ScalarE

        # scores @ V, contract along seqlen_kv
        attn_out = nl.ndarray((PMAX, PMAX),
                            dtype=nl.float32, buffer=nl.sbuf)

        attn_out_psum = nl.zeros((PMAX, PMAX), dtype=nl.float32, buffer=nl.psum)

        for i_tile_kv in nl.affine_range(seqlen_kv // PMAX): # loop on contraction
            nisa.nc_matmul(dst=attn_out_psum, stationary=scores_sbuf_t[:, i_tile_kv, :],
                                            moving=v_sbuf_t[:, i_tile_kv, :])

        # notice how here the division is done on the final attention output
        # directly comparing to the previous implementation, we save on having to 
        # loop all the i_tile_kvs, meaning we do less divsion operations as our
        # attention block is already collapsed.
        nisa.tensor_scalar(dst=attn_out, data=attn_out_psum, op0=nl.multiply,
                                           operand0=inverse_sum_row)

        # store output
        nl.store(dst=kernel_out[nl.ds(i_tile_q*PMAX, PMAX), :], value=attn_out[:, :])

    return kernel_out

####################################################################
# v6: instruction combination on ScalarE
####################################################################
@nki.jit
def attn_fwd_v6(q, k, v):
    d_head, seqlen_q = q.shape
    seqlen_kv = seqlen_q

    PMAX = nl.tile_size.pmax
    FMAX_MOVING = nl.tile_size.gemm_moving_fmax
    FMAX_STATIONARY = nl.tile_size.gemm_stationary_fmax

    assert q.shape == k.shape == v.shape
    assert d_head == PMAX
    assert seqlen_q >= 512

    kernel_out = nl.ndarray((seqlen_q, d_head), dtype=q.dtype, buffer=nl.shared_hbm)

    # load inputs into SBUF:
    q_sbuf = nl.load(q)
    k_sbuf = nl.load(k)
    v_sbuf = nl.load(v)

    # v has the wrong layout
    v_sbuf_t = nl.ndarray((PMAX, seqlen_kv // PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf)
    for i_tile_kv in nl.affine_range(seqlen_kv // PMAX):
        v_psum_t = nl.ndarray((PMAX, d_head), dtype=v_sbuf.dtype, buffer=nl.psum)
        nisa.nc_transpose(dst=v_psum_t, data=v_sbuf[:, nl.ds(i_tile_kv*PMAX, PMAX)]) # TensorE
        nisa.tensor_copy(dst=v_sbuf_t[:, i_tile_kv, :], src=v_psum_t)     # ScalarE

    num_kv_tiles = seqlen_kv // FMAX_MOVING

    # Tile along seqlen_q #
    for i_tile_q in nl.affine_range(seqlen_q // FMAX_STATIONARY): # loop on stationary_free
        # Allocate fresh PSUM tiles each iteration to avoid accumulation
        qk_tiles = []
        for _i in range(num_kv_tiles):
            qk_tiles.append(nl.ndarray((PMAX, FMAX_MOVING), dtype=nl.float32, buffer=nl.psum))

        for i_tile_kv in range(num_kv_tiles): # loop on moving_free
            # Q @ K, contract along d_head #
            nisa.nc_matmul(dst=qk_tiles[i_tile_kv], 
                stationary=q_sbuf[0:PMAX, nl.ds(i_tile_q*FMAX_STATIONARY, FMAX_STATIONARY)],
                moving=k_sbuf[0:PMAX, nl.ds(i_tile_kv*FMAX_MOVING, FMAX_MOVING)])

        # Softmax #
        # reduce max along seqlen_k
        row_max = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)

        row_max_kv = nl.ndarray((PMAX, num_kv_tiles), dtype=nl.float32, buffer=nl.sbuf)
        for i_tile_kv in range(num_kv_tiles):
            nisa.tensor_reduce(dst=row_max_kv[:, nl.ds(i_tile_kv, 1)], op=nl.maximum, data=qk_tiles[i_tile_kv], axis=(1,))

        nisa.tensor_reduce(dst=row_max[:, :], op=nl.maximum, data=row_max_kv[:, :], axis=(1,), negate=True)

        # subtract max from row
        exp_row = nl.ndarray((PMAX, seqlen_kv),
                            dtype=nl.float32, buffer=nl.sbuf)
        sum_row_tiles = nl.ndarray((PMAX, num_kv_tiles), dtype=nl.float32, buffer=nl.sbuf)
        
        # We leverage scalar engine's hardware capability of applying reduce after activation
        # with no extra performance cost to compute the max_val subtraction and sum reduction 
        # in one step, saving on extra loops that were previously required.
        #
        # At the same time the vector engine is freed up from compute, giving it more idle time
        for i_tile_kv in range(num_kv_tiles):
            nisa.activation(dst=exp_row[:, nl.ds(i_tile_kv*FMAX_MOVING, FMAX_MOVING)], 
                op=nl.exp,
                data=qk_tiles[i_tile_kv],
                bias=row_max,
                reduce_op=nl.add,
                reduce_res=sum_row_tiles[:, nl.ds(i_tile_kv, 1)],
                reduce_cmd=nisa.reduce_cmd.reset_reduce
            )
        sum_row = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=sum_row, op=nl.add, data=sum_row_tiles, axis=(1,))

        # reciprocal of sum_row, tile shape is [PMAX, 1]
        inverse_sum_row = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.reciprocal(dst=inverse_sum_row, data=sum_row)

        # scores has the wrong layout
        scores_sbuf_t = nl.ndarray((PMAX, seqlen_kv // PMAX, PMAX),
                                    dtype=nl.float32, buffer=nl.sbuf)
        for i_tile_kv in nl.affine_range(seqlen_kv // PMAX):
            scores_psum_t = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_transpose(dst=scores_psum_t, data=exp_row[:, nl.ds(i_tile_kv*PMAX, PMAX)]) # TensorE
            nisa.tensor_copy(dst=scores_sbuf_t[:, i_tile_kv, :], src=scores_psum_t)    # ScalarE

        # scores @ V, contract along seqlen_kv
        attn_out = nl.ndarray((PMAX, PMAX),
                            dtype=nl.float32, buffer=nl.sbuf)

        attn_out_psum = nl.zeros((PMAX, PMAX), dtype=nl.float32, buffer=nl.psum)

        for i_tile_kv in nl.affine_range(seqlen_kv // PMAX): # loop on contraction
            nisa.nc_matmul(dst=attn_out_psum, stationary=scores_sbuf_t[:, i_tile_kv, :],
                                            moving=v_sbuf_t[:, i_tile_kv, :])

        nisa.tensor_scalar(dst=attn_out, data=attn_out_psum, op0=nl.multiply,
                                           operand0=inverse_sum_row)

        # store output
        nl.store(dst=kernel_out[nl.ds(i_tile_q*PMAX, PMAX), :], value=attn_out[:, :])

    return kernel_out


####################################################################
# v7: Downcast scores before transpose
# lower precision operations especially on transposes introduce
# higher performance in exchange of small precision loss.
# Furthermore, scalar engine has dtype conversion embedded,
# allowing some conversion cost to be pipelined away before
# going to the tensor engine for transposes.
####################################################################
@nki.jit
def attn_fwd_v7(q, k, v):
    d_head, seqlen_q = q.shape
    seqlen_kv = seqlen_q

    PMAX = nl.tile_size.pmax
    FMAX_MOVING = nl.tile_size.gemm_moving_fmax
    FMAX_STATIONARY = nl.tile_size.gemm_stationary_fmax

    assert q.shape == k.shape == v.shape
    assert d_head == PMAX
    assert seqlen_q >= 512

    kernel_out = nl.ndarray((seqlen_q, d_head), dtype=q.dtype, buffer=nl.shared_hbm)

    # load inputs into SBUF:
    q_sbuf = nl.load(q)
    k_sbuf = nl.load(k)
    v_sbuf = nl.load(v)

    # v has the wrong layout
    v_sbuf_t = nl.ndarray((PMAX, seqlen_kv // PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf)
    for i_tile_kv in nl.affine_range(seqlen_kv // PMAX):
        v_psum_t = nl.ndarray((PMAX, d_head), dtype=v_sbuf.dtype, buffer=nl.psum)
        nisa.nc_transpose(dst=v_psum_t, data=v_sbuf[:, nl.ds(i_tile_kv*PMAX, PMAX)]) # TensorE
        nisa.tensor_copy(dst=v_sbuf_t[:, i_tile_kv, :], src=v_psum_t)     # ScalarE

    num_kv_tiles = seqlen_kv // FMAX_MOVING

    # Tile along seqlen_q #
    for i_tile_q in nl.affine_range(seqlen_q // FMAX_STATIONARY): # loop on stationary_free
        # Allocate fresh PSUM tiles each iteration to avoid accumulation
        qk_tiles = []
        for _i in range(num_kv_tiles):
            qk_tiles.append(nl.ndarray((PMAX, FMAX_MOVING), dtype=nl.float32, buffer=nl.psum))

        for i_tile_kv in range(num_kv_tiles): # loop on moving_free
            # Q @ K, contract along d_head #
            nisa.nc_matmul(dst=qk_tiles[i_tile_kv], 
                stationary=q_sbuf[0:PMAX, nl.ds(i_tile_q*FMAX_STATIONARY, FMAX_STATIONARY)],
                moving=k_sbuf[0:PMAX, nl.ds(i_tile_kv*FMAX_MOVING, FMAX_MOVING)])

        # Softmax #
        # reduce max along seqlen_k
        row_max = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)

        row_max_kv = nl.ndarray((PMAX, num_kv_tiles), dtype=nl.float32, buffer=nl.sbuf)
        for i_tile_kv in range(num_kv_tiles):
            nisa.tensor_reduce(dst=row_max_kv[:, nl.ds(i_tile_kv, 1)], op=nl.maximum, data=qk_tiles[i_tile_kv], axis=(1,))

        nisa.tensor_reduce(dst=row_max[:, :], op=nl.maximum, data=row_max_kv[:, :], axis=(1,), negate=True)

        # subtract max from row
        exp_row = nl.ndarray((PMAX, seqlen_kv),
                            dtype=nl.bfloat16, buffer=nl.sbuf)
        sum_row_tiles = nl.ndarray((PMAX, num_kv_tiles), dtype=nl.float32, buffer=nl.sbuf)

        for i_tile_kv in range(num_kv_tiles):
            nisa.activation(dst=exp_row[:, nl.ds(i_tile_kv*FMAX_MOVING, FMAX_MOVING)], 
                op=nl.exp,
                data=qk_tiles[i_tile_kv],
                bias=row_max,
                reduce_op=nl.add,
                reduce_res=sum_row_tiles[:, nl.ds(i_tile_kv, 1)],
                reduce_cmd=nisa.reduce_cmd.reset_reduce
            )
        sum_row = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=sum_row, op=nl.add, data=sum_row_tiles, axis=(1,))

        # reciprocal of sum_row, tile shape is [PMAX, 1]
        inverse_sum_row = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.reciprocal(dst=inverse_sum_row, data=sum_row)

        # scores has the wrong layout
        scores_sbuf_t = nl.ndarray((PMAX, seqlen_kv // PMAX, PMAX),
                                    dtype=nl.bfloat16, buffer=nl.sbuf)
        for i_tile_kv in nl.affine_range(seqlen_kv // PMAX):
            scores_psum_t = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.psum)
            nisa.nc_transpose(dst=scores_psum_t, data=exp_row[:, nl.ds(i_tile_kv*PMAX, PMAX)]) # TensorE
            nisa.tensor_copy(dst=scores_sbuf_t[:, i_tile_kv, :], src=scores_psum_t)    # ScalarE

        # scores @ V, contract along seqlen_kv
        attn_out = nl.ndarray((PMAX, PMAX),
                            dtype=nl.float32, buffer=nl.sbuf)

        attn_out_psum = nl.zeros((PMAX, PMAX), dtype=nl.float32, buffer=nl.psum)

        for i_tile_kv in nl.affine_range(seqlen_kv // PMAX): # loop on contraction
            nisa.nc_matmul(dst=attn_out_psum, stationary=scores_sbuf_t[:, i_tile_kv, :],
                                            moving=v_sbuf_t[:, i_tile_kv, :])

        nisa.tensor_scalar(dst=attn_out, data=attn_out_psum, op0=nl.multiply,
                                           operand0=inverse_sum_row)

        # store output
        nl.store(dst=kernel_out[nl.ds(i_tile_q*PMAX, PMAX), :], value=attn_out[:, :])

    return kernel_out


####################################################################
# v8: Use tensor_scalar_reduce on VectorE
# In short, this evicts PSUM earlier allowing other Q@K tiles
# to potentially be computed, freeing up the tensor engine to do
# compute. This does lead to a slowdown compared to the v7 kernel, 
# which is the fastest attention kernel we have thus far, but it 
# sets us up for software-pipelining and manual allocation, which
# should outweight the cost penalty.
####################################################################
@nki.jit
def attn_fwd_v8(q, k, v):
    d_head, seqlen_q = q.shape
    seqlen_kv = seqlen_q

    PMAX = nl.tile_size.pmax
    FMAX_MOVING = nl.tile_size.gemm_moving_fmax
    FMAX_STATIONARY = nl.tile_size.gemm_stationary_fmax

    assert q.shape == k.shape == v.shape
    assert d_head == PMAX
    assert seqlen_q >= 512

    kernel_out = nl.ndarray((seqlen_q, d_head), dtype=q.dtype, buffer=nl.shared_hbm)

    # load inputs into SBUF
    q_sbuf = nl.load(q)
    k_sbuf = nl.load(k)
    v_sbuf = nl.load(v)

    # v has the wrong layout
    v_sbuf_t = nl.ndarray((PMAX, seqlen_kv // PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf)
    for i_tile_kv in nl.affine_range(seqlen_kv // PMAX):
        v_psum_t = nl.ndarray((PMAX, d_head), dtype=v_sbuf.dtype, buffer=nl.psum)
        nisa.nc_transpose(dst=v_psum_t, data=v_sbuf[:, nl.ds(i_tile_kv*PMAX, PMAX)]) # TensorE
        nisa.tensor_copy(dst=v_sbuf_t[:, i_tile_kv, :], src=v_psum_t)     # ScalarE

    num_kv_tiles = seqlen_kv // FMAX_MOVING

    # Tile along seqlen_q #
    for i_tile_q in nl.affine_range(seqlen_q // FMAX_STATIONARY): # loop on stationary_free
        # Allocate fresh PSUM tiles each iteration to avoid accumulation
        qk_tiles = []
        for _i in range(num_kv_tiles):
            qk_tiles.append(nl.ndarray((PMAX, FMAX_MOVING), dtype=nl.float32, buffer=nl.psum))

        for i_tile_kv in range(num_kv_tiles): # loop on moving_free
            # Q @ K, contract along d_head #
            nisa.nc_matmul(dst=qk_tiles[i_tile_kv], 
                stationary=q_sbuf[0:PMAX, nl.ds(i_tile_q*FMAX_STATIONARY, FMAX_STATIONARY)],
                moving=k_sbuf[0:PMAX, nl.ds(i_tile_kv*FMAX_MOVING, FMAX_MOVING)])

        # Softmax #
        # reduce max along seqlen_k
        qk_sbuf = nl.ndarray((PMAX, num_kv_tiles, FMAX_MOVING), dtype=nl.float32, buffer=nl.sbuf)
        row_max = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        row_max_kv = nl.ndarray((PMAX, num_kv_tiles), dtype=nl.float32, buffer=nl.sbuf)

        for i_tile_kv in range(num_kv_tiles):
            # previously the entire qk_sbuf row would be processed at once, so PSUM would be occupied for longer
            # here PSUM gets evicted a bit earlier, allowing us to queue the tensor engine earlier as well.
            qk_tile_sb = nl.ndarray((PMAX, FMAX_MOVING), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=qk_tile_sb, src=qk_tiles[i_tile_kv])
            nisa.tensor_scalar_reduce(dst=qk_sbuf[:, i_tile_kv, :], data=qk_tile_sb, op0=nl.multiply, operand0=1.0,
                                                                    reduce_op=nl.maximum, reduce_res=row_max_kv[:, nl.ds(i_tile_kv, 1)])
                                                                    
        nisa.tensor_reduce(dst=row_max[:, :], op=nl.maximum, data=row_max_kv[:, :], axis=(1,), negate=True)

        # subtract max from row
        exp_row = nl.ndarray((PMAX, seqlen_kv),
                            dtype=nl.bfloat16, buffer=nl.sbuf)
        sum_row_tiles = nl.ndarray((PMAX, num_kv_tiles), dtype=nl.float32, buffer=nl.sbuf)

        for i_tile_kv in range(num_kv_tiles):
            nisa.activation(dst=exp_row[:, nl.ds(i_tile_kv*FMAX_MOVING, FMAX_MOVING)], 
                op=nl.exp,
                data=qk_tiles[i_tile_kv],
                bias=row_max,
                reduce_op=nl.add,
                reduce_res=sum_row_tiles[:, nl.ds(i_tile_kv, 1)],
                reduce_cmd=nisa.reduce_cmd.reset_reduce
            )
        sum_row = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=sum_row, op=nl.add, data=sum_row_tiles, axis=(1,))

        # reciprocal of sum_row, tile shape is [PMAX, 1]
        inverse_sum_row = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.reciprocal(dst=inverse_sum_row, data=sum_row)

        # scores has the wrong layout
        scores_sbuf_t = nl.ndarray((PMAX, seqlen_kv // PMAX, PMAX),
                                    dtype=nl.bfloat16, buffer=nl.sbuf)
        for i_tile_kv in nl.affine_range(seqlen_kv // PMAX):
            scores_psum_t = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.psum)
            nisa.nc_transpose(dst=scores_psum_t, data=exp_row[:, nl.ds(i_tile_kv*PMAX, PMAX)]) # TensorE
            nisa.tensor_copy(dst=scores_sbuf_t[:, i_tile_kv, :], src=scores_psum_t)    # ScalarE

        # scores @ V, contract along seqlen_kv
        attn_out = nl.ndarray((PMAX, PMAX),
                            dtype=nl.float32, buffer=nl.sbuf)

        attn_out_psum = nl.zeros((PMAX, PMAX), dtype=nl.float32, buffer=nl.psum)

        for i_tile_kv in nl.affine_range(seqlen_kv // PMAX): # loop on contraction
            nisa.nc_matmul(dst=attn_out_psum, stationary=scores_sbuf_t[:, i_tile_kv, :],
                                            moving=v_sbuf_t[:, i_tile_kv, :])

        nisa.tensor_scalar(dst=attn_out, data=attn_out_psum, op0=nl.multiply,
                                           operand0=inverse_sum_row)

        # store output
        nl.store(dst=kernel_out[nl.ds(i_tile_q*PMAX, PMAX), :], value=attn_out[:, :])

    return kernel_out

####################################################################
# v8a_2: refactor v8 to prepare for direct allocation
# and software pipelining
####################################################################
@nki.jit
def attn_fwd_v8a(q, k, v):
    d_head, seqlen_q = q.shape
    seqlen_kv = seqlen_q

    PMAX = nl.tile_size.pmax
    FMAX_MOVING = nl.tile_size.gemm_moving_fmax

    assert q.shape == k.shape == v.shape
    assert d_head == PMAX
    assert seqlen_q >= 512

    kernel_out = nl.ndarray((seqlen_q, d_head), dtype=nl.bfloat16, buffer=nl.shared_hbm)

    q_sbuf = nl.load(q)
    k_sbuf = nl.load(k)
    v_sbuf = nl.load(v)

    # v has the wrong layout
    v_sbuf_t = nl.ndarray((PMAX, seqlen_kv // PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf)
    for i_tile_kv in nl.affine_range(seqlen_kv // PMAX):
        v_psum_t = nl.ndarray((PMAX, PMAX), dtype=v_sbuf.dtype, buffer=nl.psum)
        nisa.nc_transpose(dst=v_psum_t, data=v_sbuf[:, nl.ds(i_tile_kv*PMAX, PMAX)])
        nisa.tensor_copy(dst=v_sbuf_t[:, i_tile_kv, :], src=v_psum_t)

    num_tile_q = seqlen_q // PMAX
    num_kv_tiles = seqlen_kv // FMAX_MOVING

    # Tile along seqlen_q #
    for i_tile_q in nl.sequential_range(num_tile_q):
        # --- qk_max ---
        qk_tiles = []
        for _i in range(num_kv_tiles):
            qk_tiles.append(nl.ndarray((PMAX, FMAX_MOVING), dtype=nl.float32, buffer=nl.psum))

        row_max_kv = nl.ndarray((PMAX, num_kv_tiles), dtype=nl.float32, buffer=nl.sbuf)
        qk_sbuf_tiles = nl.ndarray((PMAX, num_kv_tiles, FMAX_MOVING), dtype=nl.float32, buffer=nl.sbuf)
        for i_tile_kv in range(num_kv_tiles):
            nisa.nc_matmul(dst=qk_tiles[i_tile_kv],
                stationary=q_sbuf[0:PMAX, nl.ds(i_tile_q*PMAX, PMAX)],
                moving=k_sbuf[0:PMAX, nl.ds(i_tile_kv*FMAX_MOVING, FMAX_MOVING)])
            qk_sb = nl.ndarray((PMAX, FMAX_MOVING), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=qk_sb, src=qk_tiles[i_tile_kv])
            nisa.tensor_scalar_reduce(dst=qk_sbuf_tiles[:, i_tile_kv, :], data=qk_sb,
                                      op0=nl.multiply, operand0=1.0,
                                      reduce_op=nl.maximum,
                                      reduce_res=row_max_kv[:, nl.ds(i_tile_kv, 1)])
        row_max = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=row_max, op=nl.maximum, data=row_max_kv, axis=(1,), negate=True)

        # --- exp_row_sum ---
        exp_row = nl.ndarray((PMAX, seqlen_kv), dtype=nl.bfloat16, buffer=nl.sbuf)
        sum_row_tiles = nl.ndarray((PMAX, num_kv_tiles), dtype=nl.float32, buffer=nl.sbuf)
        for i_tile_kv in nl.affine_range(seqlen_kv // FMAX_MOVING):
            nisa.activation(dst=exp_row[:, nl.ds(i_tile_kv*FMAX_MOVING, FMAX_MOVING)],
                op=nl.exp,
                data=qk_sbuf_tiles[:, i_tile_kv, :],
                bias=row_max,
                reduce_op=nl.add,
                reduce_res=sum_row_tiles[:, nl.ds(i_tile_kv, 1)],
                reduce_cmd=nisa.reduce_cmd.reset_reduce)

        # --- transpose_scores ---
        scores_sbuf_t = nl.ndarray((PMAX, seqlen_kv // PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf)
        for i_tile_kv in nl.affine_range(seqlen_kv // PMAX):
            scores_psum_t = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.psum)
            nisa.nc_transpose(dst=scores_psum_t, data=exp_row[:, nl.ds(i_tile_kv*PMAX, PMAX)])
            nisa.tensor_copy(dst=scores_sbuf_t[:, i_tile_kv, :], src=scores_psum_t)

        # --- pv_matmul ---
        attn_out_psum = nl.zeros((PMAX, PMAX), dtype=nl.float32, buffer=nl.psum)
        for i_tile_kv in nl.affine_range(seqlen_kv // PMAX):
            nisa.nc_matmul(dst=attn_out_psum, stationary=scores_sbuf_t[:, i_tile_kv, :],
                                              moving=v_sbuf_t[:, i_tile_kv, :])
        attn_out_sbuf = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=attn_out_sbuf, src=attn_out_psum)

        # --- write_back ---
        sum_row = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=sum_row, op=nl.add, data=sum_row_tiles, axis=(1,))
        inverse_sum_row = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.reciprocal(dst=inverse_sum_row, data=sum_row)
        attn_out = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=attn_out, data=attn_out_sbuf, op0=nl.multiply,
                           operand0=inverse_sum_row)
        nl.store(dst=kernel_out[nl.ds(i_tile_q*PMAX, PMAX), :], value=attn_out)

    return kernel_out
