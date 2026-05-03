"""
Batch-Invariant Scaled Dot-Product Attention Kernel

Written to match the ISA style of matmul_batch_invariant.py and
rmsnorm_batch_invariant.py — explicit nisa.dma_copy / nisa.nc_matmul /
nisa.tensor_copy, hardcoded integer tile sizes, NKI 0.3.0 compliant.

The ONLY difference between deterministic=True and deterministic=False is KV_TILE:
  deterministic=True  -> KV_TILE=128  (fewer accumulation steps in scores@V)
  deterministic=False -> KV_TILE=64   (more  accumulation steps in scores@V)

Mirrors matmul (K_TILE=128 vs 64) and rmsnorm (HIDDEN_TILE=128 vs 64).

Why bfloat16 is invariant:
  The scores@V matmul accumulates into a float32 PSUM. With bfloat16 inputs,
  each softmax_score * V product is snapped to the bfloat16 coarse grid before
  entering the float32 accumulator. Regrouping KV tiles does not change the
  accumulated value — the inputs to the accumulator are identical.
  With float32 inputs the products retain full precision and different groupings
  produce different float32 partial sums.

Input layout:
  q: [seq_q, d_head]
  k: [seq_k, d_head]
  v: [seq_k, d_head]
  out: [seq_q, d_head], same dtype as inputs

Tile constraints (NKI partition dim <= 128):
  Q_TILE  = 128  (seq_q partition)
  D_TILE  = 128  (d_head — must equal d_head for this kernel)
  KV_TILE = 128 or 64 (the sole invariance variable)

NKI version: 0.3.0
"""

import nki
import nki.isa as nisa
import nki.language as nl


@nki.jit
def nki_attention_kernel_isa(q, k, v, deterministic=True):
    """
    Scaled dot-product attention: out = softmax(Q K^T / sqrt(d)) V

    Args:
        q:             [seq_q, d_head]
        k:             [seq_k, d_head]
        v:             [seq_k, d_head]
        deterministic: True  -> KV_TILE=128 (batch-invariant)
                       False -> KV_TILE=64  (more accumulations)

    Returns:
        out: [seq_q, d_head], same dtype as inputs
    """
    seq_q, d_head = q.shape
    seq_k = k.shape[0]

    Q_TILE  = 128
    D_TILE  = 128
    # THE ONLY DIFFERENCE — mirrors K_TILE in matmul kernel:
    KV_TILE = 128 if deterministic else 64

    assert d_head == D_TILE, f"d_head must be {D_TILE}, got {d_head}"
    assert seq_q % Q_TILE  == 0, f"seq_q={seq_q} must be divisible by {Q_TILE}"
    assert seq_k % KV_TILE == 0, f"seq_k={seq_k} must be divisible by KV_TILE={KV_TILE}"

    scale = float(d_head) ** -0.5

    out = nl.ndarray((seq_q, d_head), dtype=q.dtype, buffer=nl.shared_hbm)

    for q_idx in nl.affine_range(seq_q // Q_TILE):
        q_start = q_idx * Q_TILE

        # Load Q tile [Q_TILE, D_TILE], transpose to [D_TILE, Q_TILE] for stationary
        q_tile = nl.ndarray((Q_TILE, D_TILE), dtype=q.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=q_tile, src=q[q_start:q_start + Q_TILE, 0:D_TILE])

        q_t_psum = nl.ndarray((D_TILE, Q_TILE), dtype=q.dtype, buffer=nl.psum)
        nisa.nc_transpose(q_t_psum, q_tile)
        # NKI 0.3.0: tensor_copy PSUM->SBUF before any dma_copy
        q_t = nl.ndarray((D_TILE, Q_TILE), dtype=q.dtype, buffer=nl.sbuf)
        nisa.tensor_copy(dst=q_t, src=q_t_psum)

        # ── QK^T: [Q_TILE, seq_k] tiled over KV_TILE ─────────────────────────
        scores_sbuf = nl.ndarray((Q_TILE, seq_k), dtype=nl.float32, buffer=nl.sbuf)

        for kv_idx in nl.affine_range(seq_k // KV_TILE):
            kv_start = kv_idx * KV_TILE

            k_tile = nl.ndarray((KV_TILE, D_TILE), dtype=k.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=k_tile, src=k[kv_start:kv_start + KV_TILE, 0:D_TILE])

            k_t_psum = nl.ndarray((D_TILE, KV_TILE), dtype=k.dtype, buffer=nl.psum)
            nisa.nc_transpose(k_t_psum, k_tile)
            k_t = nl.ndarray((D_TILE, KV_TILE), dtype=k.dtype, buffer=nl.sbuf)
            nisa.tensor_copy(dst=k_t, src=k_t_psum)

            qk_psum = nl.ndarray((Q_TILE, KV_TILE), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=qk_psum, stationary=q_t, moving=k_t)

            # Scale and evict PSUM->SBUF (NKI 0.3.0: no dma_copy from PSUM directly)
            qk_sbuf = nl.ndarray((Q_TILE, KV_TILE), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=qk_sbuf, data=qk_psum,
                               op0=nl.multiply, operand0=scale)
            nisa.dma_copy(dst=scores_sbuf[0:Q_TILE, kv_start:kv_start + KV_TILE],
                          src=qk_sbuf)

        # ── Softmax ───────────────────────────────────────────────────────────

        # Row max
        row_max = nl.ndarray((Q_TILE, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=row_max, value=-3.4028235e+38)

        for kv_idx in nl.affine_range(seq_k // KV_TILE):
            kv_start = kv_idx * KV_TILE
            s = nl.ndarray((Q_TILE, KV_TILE), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=s,
                          src=scores_sbuf[0:Q_TILE, kv_start:kv_start + KV_TILE])
            tile_max = nl.ndarray((Q_TILE, 1), dtype=nl.float32, buffer=nl.sbuf)
            # axis=1 correct for 2D [Q_TILE, KV_TILE] — reduces free dim
            nisa.tensor_reduce(dst=tile_max, data=s,
                               op=nl.max, axis=(1,), negate=False)
            nisa.tensor_tensor(dst=row_max, data1=row_max, data2=tile_max, op=nl.maximum)

        neg_max = nl.ndarray((Q_TILE, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=neg_max, data=row_max, op0=nl.multiply, operand0=-1.0)

        # exp(s - max) + row_sum
        row_sum = nl.ndarray((Q_TILE, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=row_sum, value=0.0)

        for kv_idx in nl.affine_range(seq_k // KV_TILE):
            kv_start = kv_idx * KV_TILE
            s = nl.ndarray((Q_TILE, KV_TILE), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=s,
                          src=scores_sbuf[0:Q_TILE, kv_start:kv_start + KV_TILE])
            exp_s = nl.ndarray((Q_TILE, KV_TILE), dtype=nl.float32, buffer=nl.sbuf)
            # bias=neg_max broadcasts [Q_TILE,1] over free dim — correct usage
            nisa.activation(dst=exp_s, op=nl.exp, data=s, bias=neg_max, scale=1.0)
            nisa.dma_copy(dst=scores_sbuf[0:Q_TILE, kv_start:kv_start + KV_TILE],
                          src=exp_s)
            tile_sum = nl.ndarray((Q_TILE, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_reduce(dst=tile_sum, data=exp_s,
                               op=nl.add, axis=(1,), negate=False)
            nisa.tensor_tensor(dst=row_sum, data1=row_sum, data2=tile_sum, op=nl.add)

        # inv_sum = 1 / row_sum  [Q_TILE, 1]
        inv_sum = nl.ndarray((Q_TILE, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=inv_sum, op=nl.reciprocal, data=row_sum, scale=1.0)

        # Normalize + cast to input dtype: scores = exp_s * inv_sum
        # Use tensor_scalar with scalar operand — inv_sum is [Q_TILE,1] so we
        # use scalar_tensor_tensor to broadcast correctly (same as rmsnorm kernel)
        for kv_idx in nl.affine_range(seq_k // KV_TILE):
            kv_start = kv_idx * KV_TILE
            s_f32 = nl.ndarray((Q_TILE, KV_TILE), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=s_f32,
                          src=scores_sbuf[0:Q_TILE, kv_start:kv_start + KV_TILE])
            norm = nl.ndarray((Q_TILE, KV_TILE), dtype=q.dtype, buffer=nl.sbuf)
            # scalar_tensor_tensor: dst = data * operand0, broadcasts [Q_TILE,1]
            nisa.scalar_tensor_tensor(
                dst=norm,
                data=s_f32,
                op0=nl.multiply,
                operand0=inv_sum,
            )
            nisa.dma_copy(dst=scores_sbuf[0:Q_TILE, kv_start:kv_start + KV_TILE],
                          src=norm)

        # ── scores @ V: THE invariance-relevant accumulation ──────────────────
        # Tiles at KV_TILE. bfloat16 scores are already on the coarse grid,
        # so different KV_TILE groupings produce identical float32 PSUM values.
        out_psum = nl.ndarray((Q_TILE, D_TILE), dtype=nl.float32, buffer=nl.psum)
        nisa.memset(dst=out_psum, value=0.0)

        for kv_idx in nl.affine_range(seq_k // KV_TILE):
            kv_start = kv_idx * KV_TILE

            s = nl.ndarray((Q_TILE, KV_TILE), dtype=q.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=s,
                          src=scores_sbuf[0:Q_TILE, kv_start:kv_start + KV_TILE])

            # Transpose scores [Q_TILE, KV_TILE] -> [KV_TILE, Q_TILE] for stationary
            s_t_psum = nl.ndarray((KV_TILE, Q_TILE), dtype=q.dtype, buffer=nl.psum)
            nisa.nc_transpose(s_t_psum, s)
            # NKI 0.3.0: tensor_copy PSUM->SBUF before use as stationary
            s_t = nl.ndarray((KV_TILE, Q_TILE), dtype=q.dtype, buffer=nl.sbuf)
            nisa.tensor_copy(dst=s_t, src=s_t_psum)

            v_tile = nl.ndarray((KV_TILE, D_TILE), dtype=v.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=v_tile, src=v[kv_start:kv_start + KV_TILE, 0:D_TILE])

            # stationary=[KV_TILE, Q_TILE], moving=[KV_TILE, D_TILE] -> dst [Q_TILE, D_TILE]
            nisa.nc_matmul(dst=out_psum, stationary=s_t, moving=v_tile)

        # tensor_copy PSUM->SBUF (NKI 0.3.0 requirement), then dma_copy to HBM
        out_sbuf = nl.ndarray((Q_TILE, D_TILE), dtype=q.dtype, buffer=nl.sbuf)
        nisa.tensor_copy(dst=out_sbuf, src=out_psum)
        nisa.dma_copy(dst=out[q_start:q_start + Q_TILE, 0:D_TILE], src=out_sbuf)

    return out
