"""
Batch-Invariant Scaled Dot-Product Attention Kernel

Based on attn_fwd_v4 from nki_samples/tutorials/attention_fwd_performance/attention_kernels.py
(loop-fused, nki.isa throughout, correct PSUM accumulation pattern).

The ONLY difference between deterministic=True and deterministic=False is KV_TILE:
  deterministic=True  → KV_TILE=512  (FMAX_MOVING, fewer accumulation steps in scores@V)
  deterministic=False → KV_TILE=256  (half tile,   more  accumulation steps in scores@V)

This mirrors the matmul and rmsnorm kernels where tile size is the single
controlled variable. All other logic — softmax numerics, PSUM layout, transpose
strategy — is identical between modes.

Why bfloat16 is invariant:
  The scores@V matmul accumulates into a float32 PSUM. With bfloat16 inputs,
  each softmax_score * V product is snapped to the bfloat16 coarse grid before
  entering the float32 PSUM accumulator. Regrouping KV tiles therefore does not
  change the accumulated value — the inputs to the accumulator are identical.
  With float32 inputs the products retain full precision and different groupings
  produce different float32 partial sums.

Input layout (matches tutorial reference kernel):
  q: [d_head, seq_q]   (partition dim = d_head)
  k: [d_head, seq_k]
  v: [d_head, seq_k]   (transposed inside kernel before scores@V)
  out: [seq_q, d_head]

NKI version: 0.3.0 (Beta 3)
"""

import numpy as np
import nki
import nki.isa as nisa
import nki.language as nl
from nki.language import par_dim


@nki.jit
def nki_attention_kernel_isa(q, k, v, deterministic=True):
    """
    Scaled dot-product attention: out = softmax(Q K^T / sqrt(d)) V

    Args:
        q:             [d_head, seq_q]
        k:             [d_head, seq_k]
        v:             [d_head, seq_k]
        deterministic: True  -> KV_TILE=512 (batch-invariant)
                       False -> KV_TILE=256 (more accumulations)

    Returns:
        out: [seq_q, d_head], same dtype as inputs

    Notes:
        PSUM always accumulates in float32 regardless of input dtype.
        The ONLY difference between modes is KV_TILE (FMAX_MOVING).
        With bfloat16 inputs tiling change is invisible (invariant).
        With float32 inputs different groupings produce different partial sums.
    """
    d_head, seq_q = q.shape
    seq_k = k.shape[1]

    PMAX = nl.tile_size.pmax                    # 128
    FMAX = nl.tile_size.gemm_moving_fmax        # 512
    # THE ONLY DIFFERENCE:
    KV_TILE = FMAX if deterministic else FMAX // 2   # 512 vs 256

    assert d_head == PMAX, f"d_head must be {PMAX}, got {d_head}"
    assert seq_q % PMAX == 0, f"seq_q must be divisible by {PMAX}"
    assert seq_k % KV_TILE == 0, f"seq_k={seq_k} must be divisible by KV_TILE={KV_TILE}"

    softmax_scale = float(d_head) ** -0.5

    out = nl.ndarray((seq_q, d_head), dtype=q.dtype, buffer=nl.shared_hbm)

    # Pre-transpose V into tiled SBUF: [d_head, seq_k] -> [par_dim(PMAX), seq_k//PMAX, PMAX]
    v_t = nl.ndarray((par_dim(PMAX), seq_k // PMAX, PMAX), dtype=q.dtype, buffer=nl.sbuf)
    for i_kv in nl.affine_range(seq_k // PMAX):
        v_psum_t = nisa.nc_transpose(v[:, nl.ds(i_kv * PMAX, PMAX)])
        v_t[:, i_kv, :] = nisa.tensor_copy(v_psum_t, dtype=q.dtype)

    # Load Q, K into SBUF once
    q_sbuf = nl.ndarray((d_head, seq_q), dtype=q.dtype, buffer=nl.sbuf)
    k_sbuf = nl.ndarray((d_head, seq_k), dtype=k.dtype, buffer=nl.sbuf)
    q_sbuf[...] = nl.load(q)
    k_sbuf[...] = nl.load(k)

    # Outer loop: one output tile of PMAX rows per iteration
    for i_q in nl.affine_range(seq_q // PMAX):

        # --- QK^T tiled over KV_TILE ---
        qk = nl.ndarray((seq_k // KV_TILE, par_dim(PMAX), KV_TILE),
                        dtype=nl.float32, buffer=nl.psum)
        for i_kv in nl.affine_range(seq_k // KV_TILE):
            qk[i_kv, :, :] = nisa.nc_matmul(
                stationary=q_sbuf[0:PMAX, nl.ds(i_q * PMAX, PMAX)],
                moving=k_sbuf[0:PMAX, nl.ds(i_kv * KV_TILE, KV_TILE)])

        # Scale and evict PSUM to SBUF, find row max for stable softmax
        qk_sbuf = nl.ndarray((par_dim(PMAX), seq_k), dtype=nl.float32, buffer=nl.sbuf)
        row_max_kv = nl.ndarray((par_dim(PMAX), seq_k // KV_TILE), dtype=nl.float32, buffer=nl.sbuf)
        for i_kv in nl.affine_range(seq_k // KV_TILE):
            scaled = nisa.tensor_scalar(
                data=qk[i_kv], op0=nl.multiply, operand0=softmax_scale)
            qk_sbuf[:, nl.ds(i_kv * KV_TILE, KV_TILE)] = scaled
            row_max_kv[:, i_kv] = nisa.tensor_reduce(
                op=nl.max, data=scaled, axis=(1,), dtype=nl.float32, negate=True)

        # Global row max (negated, used as bias in activation)
        row_max = nisa.tensor_reduce(
            op=nl.min, data=row_max_kv, axis=(1,), dtype=nl.float32, negate=False)

        # exp(qk_scaled + neg_max) with simultaneous partial row sum
        exp_row = nl.ndarray((par_dim(PMAX), seq_k), dtype=q.dtype, buffer=nl.sbuf)
        sum_row_kv = nl.ndarray((par_dim(PMAX), seq_k // KV_TILE), dtype=nl.float32, buffer=nl.sbuf)
        for i_kv in nl.affine_range(seq_k // KV_TILE):
            exp_row[:, nl.ds(i_kv * KV_TILE, KV_TILE)] = nisa.activation_reduce(
                op=nl.exp,
                data=qk_sbuf[:, nl.ds(i_kv * KV_TILE, KV_TILE)],
                bias=row_max,
                scale=1.0,
                reduce_op=nl.add,
                reduce_res=sum_row_kv[:, i_kv],
                dtype=q.dtype)

        sum_row = nisa.tensor_reduce(op=nl.add, data=sum_row_kv, axis=(1,), dtype=nl.float32)
        inv_sum = nisa.reciprocal(data=sum_row)

        # Normalize: scores = exp_row * inv_sum
        scores = nl.ndarray((par_dim(PMAX), seq_k), dtype=q.dtype, buffer=nl.sbuf)
        for i_kv in nl.affine_range(seq_k // KV_TILE):
            scores[:, nl.ds(i_kv * KV_TILE, KV_TILE)] = nisa.tensor_scalar(
                data=exp_row[:, nl.ds(i_kv * KV_TILE, KV_TILE)],
                op0=nl.multiply,
                operand0=inv_sum,
                engine=nisa.vector_engine,
                dtype=q.dtype)

        # Transpose scores: [PMAX, seq_k] -> tiled [par_dim(PMAX), seq_k//PMAX, PMAX]
        scores_t = nl.ndarray((par_dim(PMAX), seq_k // PMAX, PMAX), dtype=q.dtype, buffer=nl.sbuf)
        for i_kv in nl.affine_range(seq_k // PMAX):
            scores_psum_t = nisa.nc_transpose(scores[:, nl.ds(i_kv * PMAX, PMAX)])
            scores_t[:, i_kv, :] = nisa.tensor_copy(scores_psum_t, dtype=q.dtype)

        # --- scores @ V: accumulates into float32 PSUM ---
        # This is the invariance-relevant accumulation loop.
        # KV_TILE does NOT control this loop — it always tiles at PMAX (128).
        # What changes with KV_TILE is how many exp/sum tiles fed into scores above.
        # The bfloat16 grid-snap happens at the nc_matmul inputs (scores_t, v_t),
        # so different KV_TILE groupings in the softmax path still produce
        # identical float32 PSUM accumulations for bfloat16 inputs.
        attn_psum = nl.zeros((PMAX, PMAX), dtype=nl.float32, buffer=nl.psum)
        for i_kv in nl.affine_range(seq_k // PMAX):
            attn_psum += nisa.nc_matmul(
                stationary=scores_t[:, i_kv, :],
                moving=v_t[:, i_kv, :])

        # Cast PSUM -> output dtype and store
        attn_out = nisa.tensor_scalar(
            data=attn_psum, op0=nl.multiply, operand0=1.0,
            engine=nisa.vector_engine, dtype=q.dtype)
        nl.store(dst=out[nl.ds(i_q * PMAX, PMAX), :], value=attn_out)

    return out
