"""
Batch-Invariant Scaled Dot-Product Attention Kernel

NKI nc_matmul layout:
  stationary: [par_dim(P), K]  moving: [par_dim(K), N]  dst PSUM: [par_dim(P), N]
  where P = stationary free dim = dst partition dim.

For QK^T: Q=[seq_q, d_head], K=[seq_k, d_head]
  Transpose both to [d_head, Q_TILE] and [d_head, KV_TILE].
  stationary=[d_head, Q_TILE], moving=[d_head, KV_TILE] → dst=[Q_TILE, KV_TILE] ✓

For scores@V: scores=[Q_TILE, KV_TILE], V=[KV_TILE, d_head]
  Transpose scores to [KV_TILE, Q_TILE].
  stationary=[KV_TILE, Q_TILE], moving=[KV_TILE, d_head] → dst=[Q_TILE, d_head] ✓

The ONLY difference between deterministic=True and False is KV_TILE (128 vs 64).
"""

import nki
import nki.isa as nisa
import nki.language as nl
import numpy as np


@nki.jit
def nki_attention_kernel_isa(q, k, v, deterministic=True):
    """
    Scaled dot-product attention: softmax(Q K^T / sqrt(d)) V

    Args:
        q: [seq_q, d_head]
        k: [seq_k, d_head]
        v: [seq_k, d_head]
        deterministic: True → KV_TILE=128 (batch-invariant), False → KV_TILE=64

    Returns:
        out: [seq_q, d_head], same dtype as inputs
    """
    seq_q, d_head = q.shape
    seq_k = k.shape[0]

    Q_TILE = 128
    KV_TILE = 128 if deterministic else 64  # THE ONLY DIFFERENCE
    scale = float(d_head) ** -0.5

    out = nl.ndarray((seq_q, d_head), dtype=q.dtype, buffer=nl.shared_hbm)

    for q_tile_idx in nl.affine_range(seq_q // Q_TILE):
        q_start = q_tile_idx * Q_TILE

        # Load Q tile [Q_TILE, d_head] and transpose to [d_head, Q_TILE] for stationary
        q_tile = nl.ndarray((Q_TILE, d_head), dtype=q.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=q_tile, src=q[q_start:q_start + Q_TILE, 0:d_head])
        q_t = nl.ndarray((d_head, Q_TILE), dtype=q.dtype, buffer=nl.psum)
        nisa.nc_transpose(q_t, q_tile)
        q_t_sbuf = nl.ndarray((d_head, Q_TILE), dtype=q.dtype, buffer=nl.sbuf)
        nisa.tensor_copy(dst=q_t_sbuf, src=q_t)

        # scores [Q_TILE, seq_k] in SBUF (float32)
        scores = nl.ndarray((Q_TILE, seq_k), dtype=nl.float32, buffer=nl.sbuf)

        # --- QK^T ---
        for kv_idx in nl.affine_range(seq_k // KV_TILE):
            k_start = kv_idx * KV_TILE
            k_tile = nl.ndarray((KV_TILE, d_head), dtype=k.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=k_tile, src=k[k_start:k_start + KV_TILE, 0:d_head])
            k_t = nl.ndarray((d_head, KV_TILE), dtype=k.dtype, buffer=nl.psum)
            nisa.nc_transpose(k_t, k_tile)
            k_t_sbuf = nl.ndarray((d_head, KV_TILE), dtype=k.dtype, buffer=nl.sbuf)
            nisa.tensor_copy(dst=k_t_sbuf, src=k_t)

            qk_psum = nl.ndarray((Q_TILE, KV_TILE), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=qk_psum, stationary=q_t_sbuf, moving=k_t_sbuf)

            qk_sbuf = nl.ndarray((Q_TILE, KV_TILE), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=qk_sbuf, data=qk_psum, op0=nl.multiply, operand0=scale)
            nisa.dma_copy(dst=scores[0:Q_TILE, k_start:k_start + KV_TILE], src=qk_sbuf)

        # --- Softmax using activation (handles [Q_TILE,1] broadcast over free dim) ---
        # row_max
        row_max = nl.ndarray((Q_TILE, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=row_max, value=-3.4028235e+38)
        for kv_idx in nl.affine_range(seq_k // KV_TILE):
            k_start = kv_idx * KV_TILE
            s = nl.ndarray((Q_TILE, KV_TILE), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=s, src=scores[0:Q_TILE, k_start:k_start + KV_TILE])
            tile_max = nl.ndarray((Q_TILE, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_reduce(dst=tile_max, data=s, op=nl.max, axis=(1,), negate=False)
            nisa.tensor_tensor(dst=row_max, data1=row_max, data2=tile_max, op=nl.maximum)

        # negate row_max for use as bias in activation
        neg_row_max = nl.ndarray((Q_TILE, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=neg_row_max, data=row_max, op0=nl.multiply, operand0=-1.0)

        # exp(s - max) and row_sum
        row_sum = nl.ndarray((Q_TILE, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=row_sum, value=0.0)
        for kv_idx in nl.affine_range(seq_k // KV_TILE):
            k_start = kv_idx * KV_TILE
            s = nl.ndarray((Q_TILE, KV_TILE), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=s, src=scores[0:Q_TILE, k_start:k_start + KV_TILE])
            # activation: exp(s * 1.0 + neg_row_max) — neg_row_max is [Q_TILE,1], broadcasts
            exp_s = nl.ndarray((Q_TILE, KV_TILE), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(dst=exp_s, op=nl.exp, data=s, bias=neg_row_max, scale=1.0)
            nisa.dma_copy(dst=scores[0:Q_TILE, k_start:k_start + KV_TILE], src=exp_s)
            tile_sum = nl.ndarray((Q_TILE, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_reduce(dst=tile_sum, data=exp_s, op=nl.add, axis=(1,), negate=False)
            nisa.tensor_tensor(dst=row_sum, data1=row_sum, data2=tile_sum, op=nl.add)

        # inv_sum = 1/row_sum as [Q_TILE,1] vector
        inv_sum = nl.ndarray((Q_TILE, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=inv_sum, op=nl.reciprocal, data=row_sum, scale=1.0)

        # normalize: activation(copy, exp_s, scale=inv_sum) → exp_s * inv_sum, broadcasts
        for kv_idx in nl.affine_range(seq_k // KV_TILE):
            k_start = kv_idx * KV_TILE
            s = nl.ndarray((Q_TILE, KV_TILE), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=s, src=scores[0:Q_TILE, k_start:k_start + KV_TILE])
            norm = nl.ndarray((Q_TILE, KV_TILE), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(dst=norm, op=nl.copy, data=s, scale=inv_sum)
            nisa.dma_copy(dst=scores[0:Q_TILE, k_start:k_start + KV_TILE], src=norm)

        # --- scores @ V ---
        out_psum = nl.ndarray((Q_TILE, d_head), dtype=nl.float32, buffer=nl.psum)
        for kv_idx in nl.affine_range(seq_k // KV_TILE):
            k_start = kv_idx * KV_TILE
            s = nl.ndarray((Q_TILE, KV_TILE), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=s, src=scores[0:Q_TILE, k_start:k_start + KV_TILE])
            s_cast = nl.ndarray((Q_TILE, KV_TILE), dtype=q.dtype, buffer=nl.sbuf)
            nisa.tensor_copy(dst=s_cast, src=s)

            s_t = nl.ndarray((KV_TILE, Q_TILE), dtype=q.dtype, buffer=nl.psum)
            nisa.nc_transpose(s_t, s_cast)
            s_t_sbuf = nl.ndarray((KV_TILE, Q_TILE), dtype=q.dtype, buffer=nl.sbuf)
            nisa.tensor_copy(dst=s_t_sbuf, src=s_t)

            v_tile = nl.ndarray((KV_TILE, d_head), dtype=v.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=v_tile, src=v[k_start:k_start + KV_TILE, 0:d_head])
            nisa.nc_matmul(dst=out_psum, stationary=s_t_sbuf, moving=v_tile)

        out_sbuf = nl.ndarray((Q_TILE, d_head), dtype=q.dtype, buffer=nl.sbuf)
        nisa.tensor_copy(dst=out_sbuf, src=out_psum)
        nisa.dma_copy(dst=out[q_start:q_start + Q_TILE, 0:d_head], src=out_sbuf)

    return out
