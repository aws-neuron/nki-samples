"""
Batch-Invariant RMSNorm Kernel

This kernel demonstrates batch invariance in RMSNorm by controlling the
hidden-dimension tiling strategy.

NKI version: 0.3.0 (Beta 3)
"""

import math

import nki
import nki.isa as nisa
import nki.language as nl
import nki.typing as nt


@nki.jit
def nki_rmsnorm_kernel_isa(a, g, deterministic=True):
    """
    RMSNorm with batch invariance parameter.

    Computes: out[i] = a[i] / rms(a[i]) * g, where rms(x) = sqrt(mean(x^2))

    Args:
        a: Input tensor of shape [num_rows, hidden_dim]
        g: Weight tensor of shape [hidden_dim] or [1, hidden_dim]
        deterministic: If True, uses fixed HIDDEN_TILE=128, producing identical
                       results across different batch sizes / accumulation counts.
                       If False, uses HIDDEN_TILE=64.

    Returns:
        out_tensor: Normalized output of shape [num_rows, hidden_dim], same dtype as inputs

    Notes:
        Internal sum-of-squares accumulation uses float32 regardless of input dtype.
        The ONLY difference between modes is HIDDEN_TILE size, which changes the
        number of partial sum-of-squares accumulations.
        With bfloat16 inputs this difference vanishes (invariant); with float32 it does not.
    """
    out_tensor = nl.ndarray(a.shape, dtype=a.dtype, buffer=nl.shared_hbm)

    num_rows, hidden_dim = a.shape[0], a.shape[1]
    BATCH_TILE = 128
    HIDDEN_TILE = 128 if deterministic else 64

    g = g.reshape((1, hidden_dim))

    # ones_vec and zero_bias stay float32 (used in float32 compute paths)
    ones_vec = nl.ndarray((1, BATCH_TILE), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=ones_vec, value=1.0)

    zero_bias = nl.ndarray((BATCH_TILE, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=zero_bias, value=0.0)

    for i in nl.affine_range(math.ceil(num_rows / BATCH_TILE)):
        b_start = i * BATCH_TILE
        b_end = min(num_rows, b_start + BATCH_TILE)

        # sum_sq accumulates in float32 for precision
        sum_sq = nl.ndarray((BATCH_TILE, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=sum_sq, value=0.0)

        # Pass 1: Accumulate sum of squares over hidden_dim tiles
        for h in nl.affine_range(math.ceil(hidden_dim / HIDDEN_TILE)):
            h_start = h * HIDDEN_TILE
            h_end = min(hidden_dim, h_start + HIDDEN_TILE)

            x = nl.ndarray((BATCH_TILE, HIDDEN_TILE), dtype=a.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=x, src=a[b_start:b_end, h_start:h_end])

            # x_sq and tile_sum in float32 — activation_reduce upcasts input
            x_sq = nl.ndarray((BATCH_TILE, HIDDEN_TILE), dtype=nl.float32, buffer=nl.sbuf)
            tile_sum = nl.ndarray((BATCH_TILE, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation_reduce(
                dst=x_sq,
                op=nl.square,
                data=x,
                reduce_op=nl.add,
                reduce_res=tile_sum,
                bias=zero_bias,
                scale=1.0,
            )

            nisa.tensor_tensor(dst=sum_sq, data1=sum_sq, data2=tile_sum, op=nl.add)

        # rms_inv in float32
        rms_inv = nl.ndarray((BATCH_TILE, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(
            dst=rms_inv,
            op=nl.rsqrt,
            data=sum_sq,
            scale=1.0 / hidden_dim,
            bias=zero_bias,
        )

        # Pass 2: Normalize and apply weight, output in input dtype
        for h in nl.affine_range(math.ceil(hidden_dim / HIDDEN_TILE)):
            h_start = h * HIDDEN_TILE
            h_end = min(hidden_dim, h_start + HIDDEN_TILE)

            x = nl.ndarray((BATCH_TILE, HIDDEN_TILE), dtype=a.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=x, src=a[b_start:b_end, h_start:h_end])

            # Load g in float32 for the broadcast matmul
            g_tile = nl.ndarray((1, HIDDEN_TILE), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=g_tile, src=g[0:1, h_start:h_end])

            g_bcast = nl.ndarray((BATCH_TILE, HIDDEN_TILE), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=g_bcast, stationary=ones_vec, moving=g_tile)

            x_out = nl.ndarray((BATCH_TILE, HIDDEN_TILE), dtype=a.dtype, buffer=nl.sbuf)
            nisa.scalar_tensor_tensor(
                dst=x_out,
                data=x,
                op0=nl.multiply,
                operand0=rms_inv,
                op1=nl.multiply,
                operand1=g_bcast,
            )

            nisa.dma_copy(
                dst=out_tensor[b_start:b_end, h_start:h_end],
                src=x_out,
            )

    return out_tensor
