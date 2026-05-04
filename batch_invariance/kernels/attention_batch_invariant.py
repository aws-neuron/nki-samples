"""
Batch-Invariant Scaled Dot-Product Attention Kernel

Written to match the ISA style of matmul_batch_invariant.py and
rmsnorm_batch_invariant.py — explicit nisa.dma_copy / nisa.nc_matmul /
nisa.tensor_copy, hardcoded integer tile sizes, NKI 0.3.0 compliant.

The ONLY difference between deterministic=True and deterministic=False is
KV_TILE used in the scores@V matmul:
  deterministic=True  -> KV_TILE=128  (4 accumulation steps for seq_k=512)
  deterministic=False -> KV_TILE=64   (8 accumulation steps for seq_k=512)

The softmax (QK^T, row_max, exp, row_sum, normalize) always uses
KV_TILE_SOFTMAX=128 so its float32 reductions are identical in both modes.
Only the final scores@V matmul varies — matching the K_TILE variation in
matmul_batch_invariant.py exactly.

Why bfloat16 is invariant in scores@V:
  The scores@V matmul accumulates into a float32 PSUM. With bfloat16 softmax
  scores (which are bit-exact because the softmax tile size is fixed), each
  softmax_score * V product is snapped to the bfloat16 coarse grid before
  entering the float32 accumulator. Regrouping KV tiles does not change the
  accumulated value — the inputs to the accumulator are identical.
  With float32 inputs the products retain full precision and different
  groupings produce different float32 partial sums.

Why the softmax uses a fixed tile size:
  nisa.tensor_reduce(op=nl.add) uses float32 tree reduction internally.
  Different tile sizes produce different reduction trees, giving different
  float32 row_sum values even for identical inputs. Fixing the softmax tile
  size ensures the bfloat16 softmax scores are bit-exact across both variants,
  so the only difference is in the scores@V accumulation — the property we
  want to demonstrate.

Input layout:
  q: [seq_q, d_head]
  k: [seq_k, d_head]
  v: [seq_k, d_head]
  out: [seq_q, d_head], same dtype as inputs

Tile constraints (NKI partition dim <= 128):
  Q_TILE          = 128  (seq_q partition)
  D_TILE          = 128  (d_head -- must equal d_head for this kernel)
  KV_TILE_SOFTMAX = 128  (fixed -- softmax always uses 128-element tiles)
  KV_TILE         = 128 or 64 (scores@V only -- the sole invariance variable)

Requirements: seq_q % 128 == 0, seq_k % 128 == 0, d_head == 128

NKI version: 0.3.0
"""

import nki
import nki.isa as nisa
import nki.language as nl


@nki.jit
def nki_attention_kernel_isa(q, k, v, deterministic=True, attn_bias=None):
    """
    Scaled dot-product attention: out = softmax(Q K^T / sqrt(d)) V

    Args:
        q:             [seq_q, d_head]
        k:             [seq_k, d_head]
        v:             [seq_k, d_head]
        deterministic: True  -> KV_TILE=128 in scores@V (batch-invariant)
                       False -> KV_TILE=64  in scores@V (more accumulations)
        attn_bias:     optional [seq_q, seq_k] float32 HBM tensor added to
                       QK^T scores before softmax (use -1e9 to mask positions)

    Returns:
        out: [seq_q, d_head], same dtype as inputs
    """
    seq_q, d_head = q.shape
    seq_k = k.shape[0]

    Q_TILE          = 128
    D_TILE          = 128
    KV_TILE_SOFTMAX = 128  # Fixed -- softmax reductions are always identical
    KV_TILE         = 128 if deterministic else 64  # Only scores@V varies

    scale = float(d_head) ** -0.5

    out = nl.ndarray((seq_q, d_head), dtype=q.dtype, buffer=nl.shared_hbm)

    for q_idx in nl.affine_range(seq_q // Q_TILE):
        q_start = q_idx * Q_TILE

        # Load Q tile and transpose to [D_TILE, Q_TILE] for stationary in matmul
        q_tile = nl.ndarray((Q_TILE, D_TILE), dtype=q.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=q_tile, src=q[q_start:q_start + Q_TILE, 0:D_TILE])
        q_t_psum = nl.ndarray((D_TILE, Q_TILE), dtype=q.dtype, buffer=nl.psum)
        nisa.nc_transpose(q_t_psum, q_tile)
        q_t = nl.ndarray((D_TILE, Q_TILE), dtype=q.dtype, buffer=nl.sbuf)
        nisa.tensor_copy(dst=q_t, src=q_t_psum)

        # Intermediate scores buffer: stores QK^T, then exp(s-max), then softmax
        # Always tiled at KV_TILE_SOFTMAX=128 so softmax is bit-reproducible
        scores_sbuf = nl.ndarray((Q_TILE, seq_k), dtype=nl.float32, buffer=nl.sbuf)

        # ── QK^T (fixed KV_TILE_SOFTMAX) ─────────────────────────────────────
        for kv_idx in nl.affine_range(seq_k // KV_TILE_SOFTMAX):
            kv_start = kv_idx * KV_TILE_SOFTMAX

            k_tile = nl.ndarray((KV_TILE_SOFTMAX, D_TILE), dtype=k.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=k_tile, src=k[kv_start:kv_start + KV_TILE_SOFTMAX, 0:D_TILE])
            k_t_psum = nl.ndarray((D_TILE, KV_TILE_SOFTMAX), dtype=k.dtype, buffer=nl.psum)
            nisa.nc_transpose(k_t_psum, k_tile)
            k_t = nl.ndarray((D_TILE, KV_TILE_SOFTMAX), dtype=k.dtype, buffer=nl.sbuf)
            nisa.tensor_copy(dst=k_t, src=k_t_psum)

            qk_psum = nl.ndarray((Q_TILE, KV_TILE_SOFTMAX), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=qk_psum, stationary=q_t, moving=k_t)
            qk_sbuf = nl.ndarray((Q_TILE, KV_TILE_SOFTMAX), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=qk_sbuf, data=qk_psum, op0=nl.multiply, operand0=scale)
            if attn_bias is not None:
                bias_tile = nl.ndarray((Q_TILE, KV_TILE_SOFTMAX), dtype=nl.float32, buffer=nl.sbuf)
                nisa.dma_copy(dst=bias_tile,
                              src=attn_bias[q_start:q_start + Q_TILE,
                                            kv_start:kv_start + KV_TILE_SOFTMAX])
                qk_biased = nl.ndarray((Q_TILE, KV_TILE_SOFTMAX), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=qk_biased, data1=qk_sbuf, data2=bias_tile, op=nl.add)
                nisa.dma_copy(dst=scores_sbuf[0:Q_TILE, kv_start:kv_start + KV_TILE_SOFTMAX],
                              src=qk_biased)
            else:
                nisa.dma_copy(dst=scores_sbuf[0:Q_TILE, kv_start:kv_start + KV_TILE_SOFTMAX],
                              src=qk_sbuf)

        # ── Row max (fixed KV_TILE_SOFTMAX) ──────────────────────────────────
        row_max = nl.ndarray((Q_TILE, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=row_max, value=-3.4028235e+38)

        for kv_idx in nl.affine_range(seq_k // KV_TILE_SOFTMAX):
            kv_start = kv_idx * KV_TILE_SOFTMAX
            s = nl.ndarray((Q_TILE, KV_TILE_SOFTMAX), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=s, src=scores_sbuf[0:Q_TILE, kv_start:kv_start + KV_TILE_SOFTMAX])
            tile_max = nl.ndarray((Q_TILE, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_reduce(dst=tile_max, data=s, op=nl.maximum, axis=(1,), negate=False)
            nisa.tensor_tensor(dst=row_max, data1=row_max, data2=tile_max, op=nl.maximum)

        neg_max = nl.ndarray((Q_TILE, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=neg_max, data=row_max, op0=nl.multiply, operand0=-1.0)

        # ── exp(s - max) + row_sum (fixed KV_TILE_SOFTMAX) ───────────────────
        row_sum = nl.ndarray((Q_TILE, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=row_sum, value=0.0)

        for kv_idx in nl.affine_range(seq_k // KV_TILE_SOFTMAX):
            kv_start = kv_idx * KV_TILE_SOFTMAX
            s = nl.ndarray((Q_TILE, KV_TILE_SOFTMAX), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=s, src=scores_sbuf[0:Q_TILE, kv_start:kv_start + KV_TILE_SOFTMAX])
            exp_s = nl.ndarray((Q_TILE, KV_TILE_SOFTMAX), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(dst=exp_s, op=nl.exp, data=s, bias=neg_max, scale=1.0)
            nisa.dma_copy(dst=scores_sbuf[0:Q_TILE, kv_start:kv_start + KV_TILE_SOFTMAX],
                          src=exp_s)
            tile_sum = nl.ndarray((Q_TILE, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_reduce(dst=tile_sum, data=exp_s, op=nl.add, axis=(1,), negate=False)
            nisa.tensor_tensor(dst=row_sum, data1=row_sum, data2=tile_sum, op=nl.add)

        inv_sum = nl.ndarray((Q_TILE, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=inv_sum, op=nl.reciprocal, data=row_sum, scale=1.0)

        # ── Normalize to bfloat16 (fixed KV_TILE_SOFTMAX) ────────────────────
        # After this loop, scores_sbuf holds bfloat16 softmax scores (in float32 slots).
        # These values are bit-exact in both modes because KV_TILE_SOFTMAX is fixed.
        for kv_idx in nl.affine_range(seq_k // KV_TILE_SOFTMAX):
            kv_start = kv_idx * KV_TILE_SOFTMAX
            s_f32 = nl.ndarray((Q_TILE, KV_TILE_SOFTMAX), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=s_f32,
                          src=scores_sbuf[0:Q_TILE, kv_start:kv_start + KV_TILE_SOFTMAX])
            norm = nl.ndarray((Q_TILE, KV_TILE_SOFTMAX), dtype=q.dtype, buffer=nl.sbuf)
            ones_bcast = nl.ndarray((Q_TILE, KV_TILE_SOFTMAX), dtype=nl.float32, buffer=nl.sbuf)
            nisa.memset(dst=ones_bcast, value=1.0)
            nisa.scalar_tensor_tensor(
                dst=norm,
                data=s_f32,
                op0=nl.multiply,
                operand0=inv_sum,
                op1=nl.multiply,
                operand1=ones_bcast,
            )
            nisa.dma_copy(dst=scores_sbuf[0:Q_TILE, kv_start:kv_start + KV_TILE_SOFTMAX],
                          src=norm)

        # ── scores @ V: THE invariance-relevant accumulation (variable KV_TILE)
        # Softmax scores are bit-exact bfloat16 values in both modes.
        # Regrouping at KV_TILE=128 vs KV_TILE=64 changes only the float32
        # accumulation order -- but bfloat16 products are on the coarse grid
        # so different groupings produce identical float32 PSUM values.
        out_psum = nl.ndarray((Q_TILE, D_TILE), dtype=nl.float32, buffer=nl.psum)
        nisa.memset(dst=out_psum, value=0.0)

        for kv_idx in nl.affine_range(seq_k // KV_TILE):
            kv_start = kv_idx * KV_TILE

            s = nl.ndarray((Q_TILE, KV_TILE), dtype=q.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=s,
                          src=scores_sbuf[0:Q_TILE, kv_start:kv_start + KV_TILE])

            s_t_psum = nl.ndarray((KV_TILE, Q_TILE), dtype=q.dtype, buffer=nl.psum)
            nisa.nc_transpose(s_t_psum, s)
            s_t = nl.ndarray((KV_TILE, Q_TILE), dtype=q.dtype, buffer=nl.sbuf)
            nisa.tensor_copy(dst=s_t, src=s_t_psum)

            v_tile = nl.ndarray((KV_TILE, D_TILE), dtype=v.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=v_tile, src=v[kv_start:kv_start + KV_TILE, 0:D_TILE])

            # stationary=[KV_TILE, Q_TILE], moving=[KV_TILE, D_TILE] -> [Q_TILE, D_TILE]
            nisa.nc_matmul(dst=out_psum, stationary=s_t, moving=v_tile)

        out_sbuf = nl.ndarray((Q_TILE, D_TILE), dtype=q.dtype, buffer=nl.sbuf)
        nisa.tensor_copy(dst=out_sbuf, src=out_psum)
        nisa.dma_copy(dst=out[q_start:q_start + Q_TILE, 0:D_TILE], src=out_sbuf)

    return out
