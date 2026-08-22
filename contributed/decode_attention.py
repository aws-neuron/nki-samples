"""
Decode (flash-decoding) attention kernel.

Decode step of autoregressive attention: a single query position (seqlen_q = 1) attending over a cached K/V of length seqlen_kv. 
This is the memory-bound complement to the compute-bound prefill kernel in `pipelined_attention.py`.

Two kernels live here. 
`decode_attention_fwd` is the simplest correct version: one head, one KV tile, seqlen_kv <= 128. 
`decode_attention_gqa_fwd` lifts that length limit with KV tiling and an online softmax, and 
adds grouped-query attention (GQA) so query heads sharing a KV head also share its K/V loads.

Author: Varun (varuntej.dev@gmail.com)

WARNING: These kernels:
   - Are validated against a NumPy reference via nki.simulate_kernel,
     not yet on Neuron hardware
   - Have not been tested across all input configurations
   - Carry no compatibility guarantees
   - May change without prior notice

Status:
   - [A] done: single-head, single-tile decode (MHA, seqlen_kv <= 128)
   - [B] done: KV tiling + online softmax + GQA (decode_attention_gqa_fwd)
   - [C] planned: flash-decoding split-KV for long context

"""
import math
import numpy as np

import nki
# nisa - Neuron Instruction Set Architecture. This is the low-level API to Neuron hardware.
import nki.isa as nisa
import nki.language as nl

# =====================================================================
# Milestone A: single-head, single-tile decode (MHA).
# Adapted from `attn_fwd_v1` in the attention_fwd_performance tutorial,
# specialized to seqlen_q = 1 and with the softmax scale applied.

@nki.jit
def decode_attention_fwd(q, k, v, softmax_scale=None):
    """
    Bird's Eye View: The model has already processed the prompt; Keys/Values are cached in HBM.
    Now this is the kernel for generating tokens one at a time.
    This kernel computes the attention for one new token, attending over the entire cached KV.
    
    IO tensor layouts (d on the partition axis, matching attn_fwd_v1):
      - q: (d, seqlen_q)     with seqlen_q == 1   (one new query vector)
      - k: (d, seqlen_kv)                         (cached keys, d-major)
      - v: (d, seqlen_kv)                         (cached values, d-major)
      - returns o: (seqlen_q, d) == (1, d)

    Compile-time constant: softmax_scale (defaults to 1/sqrt(d)).

    Assumptions (Milestone A):
      - d <= 128          (head dim fits the partition axis)
      - seqlen_q == 1     (decode: a single query position)
      - seqlen_kv <= 128  (single tile; the P@V contraction axis must fit the 128-wide partition dimension. 
                           Lifting this is Milestone B: KV tiling + online softmax.)
    """
    d, seqlen_q = q.shape
    d_k, seqlen_kv = k.shape
    d_v, seqlen_kv_v = v.shape

    assert d == d_k == d_v, "q, k, v must share head dim d"
    assert seqlen_kv == seqlen_kv_v, "k and v must share seqlen_kv"
    assert d <= 128, "head dim d must fit the 128-wide partition axis"
    assert seqlen_q == 1, "decode kernel expects a single query position"
    assert seqlen_kv <= 128, "Milestone A is single-tile; tile KV in Milestone B"

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(d)

    out = nl.ndarray((seqlen_q, d), dtype=q.dtype, buffer=nl.shared_hbm)

    # --- load inputs (q, k, v) from HBM -> copy to SBUF ---
    q_sbuf = nl.load(q)   # [d, 1] one new query vector
    k_sbuf = nl.load(k)   # [d, seqlen_kv]  cached keys
    v_sbuf = nl.load(v)   # [d, seqlen_kv]  cached values

    # --- logits: s = scale * (qᵀ @ k), contract over d (the partition axis) ---
    # matmul lands in PSUM (the only exit door from the tensor engine).
    qk_psum = nl.matmul(q_sbuf, k_sbuf, transpose_x=True)   # [seqlen_q, seqlen_kv]

    # The vector/scalar engines that run the softmax *can* read PSUM, but PSUM is tiny 
    # and is meant to hold tensor-engine matmul outputs, so the recommended practice is 
    # to evict to SBUF as soon as possible and free the bank for the next matmul. 
    # nc_matmul already accumulates in fp32; keeping it fp32 here keeps the softmax numerically stable.
    qk_sbuf = nl.ndarray(qk_psum.shape, dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(qk_sbuf, qk_psum)

    # (seqlen_q, seqlen_kv) = (1, seqlen_kv); tensor_scalar writes the scaled tile
    qk_scaled = nl.ndarray(qk_sbuf.shape, dtype=qk_sbuf.dtype, buffer=nl.sbuf)
    nisa.tensor_scalar(qk_scaled, qk_sbuf, op0=nl.multiply, operand0=softmax_scale)

    # softmax over seqlen_kv (the cached tokens). Reduce along axis=1 with keepdims
    # collapses seqlen_kv -> 1, so row_max has shape (seqlen_q, 1) = (1, 1).
    row_max = nl.max(qk_scaled, axis=1, keepdims=True)      # find max (stability)
    norm = nl.ndarray(qk_scaled.shape, dtype=qk_scaled.dtype, buffer=nl.sbuf)
    nisa.tensor_scalar(norm, qk_scaled, op0=nl.subtract, operand0=row_max)   # subtract max

    # softmax(x) = exp(x) / Σexp(x); scores = softmax(qk_scaled)
    exp_row = nl.exp(norm)                                 # exponentiate [seqlen_q, seqlen_kv]
    sum_row = nl.sum(exp_row, axis=1, keepdims=True)       # denominator [seqlen_q, 1]
    inv_sum = nl.reciprocal(sum_row)                 # 1 / denominator

    scores = nl.ndarray(exp_row.shape, dtype=exp_row.dtype, buffer=nl.sbuf)
    nisa.tensor_scalar(scores, exp_row, op0=nl.multiply, operand0=inv_sum)

    # output = Σⱼ scoreⱼ · vⱼ
    v_t_psum = nl.transpose(v_sbuf)           # (d, N) -> (seqlen_kv, d) = [N, d]

    # nl.transpose runs on the Tensor Engine, so its result lands in PSUM. 
    # nc_matmul must read its inputs from SBUF, so we evacuate the transposed result 
    # from PSUM to SBUF before the final matmul. Hence, tensor_copy.
    v_t = nl.ndarray(v_t_psum.shape, dtype=v_sbuf.dtype, buffer=nl.sbuf)
    nisa.tensor_copy(v_t, v_t_psum)

    scores_t_psum = nl.transpose(scores)           # [seqlen_kv, seqlen_q]
    scores_t = nl.ndarray(scores_t_psum.shape, dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(scores_t, scores_t_psum)

    attn_psum = nl.matmul(scores_t, v_t, transpose_x=True)        # [seqlen_q, d] = (1, d)
    attn_sbuf = nl.ndarray(attn_psum.shape, dtype=q.dtype, buffer=nl.sbuf)
    nisa.tensor_copy(attn_sbuf, attn_psum)      # PSUM -> SBUF

    nl.store(out, value=attn_sbuf)      # copy output from SBUF -> HBM
    return out


# =====================================================================
# Milestone B: KV tiling + online softmax + grouped-query attention (GQA).
# =====================================================================
# Single batch element. Builds directly on Milestone A:
#   * same QK -> scale -> softmax -> PV pipeline, but
#   * seqlen_kv is streamed in tiles of TILE_KV with a running online-softmax
#     state (m, l, acc) carried across tiles, so we never need all logits at
#     once (this is what lifts A's 'seqlen_kv <= 128' wall), and
#   * 'group' query heads share ONE KV head -> load K/V once per group (GQA win).

TILE_KV = 128   # KV chunk width. Must be <= 128: it becomes the partition axis of the P@V matmul


@nki.jit
def decode_attention_gqa_fwd(q, k, v, n_q_heads, n_kv_heads, softmax_scale=None):
    """
    GQA decode attention, single batch element, online softmax over KV tiles.

    IO tensor layouts (d on the partition axis):
      - q: (d, n_q_heads)
      - k: (n_kv_heads, d, seqlen_kv)
      - v: (n_kv_heads, d, seqlen_kv)
      - returns o: (n_q_heads, d)

    Compile-time constants: n_q_heads, n_kv_heads, softmax_scale (default 1/sqrt(d)).

    Assumptions (Milestone B v1):
      - d <= 128
      - n_q_heads % n_kv_heads == 0       (group = n_q_heads // n_kv_heads)
      - seqlen_kv % TILE_KV == 0          (no padding yet -> Future Work)
    """
    d, n_q = q.shape
    n_kv, d_k, seqlen_kv = k.shape
    n_kv_v, d_v, seqlen_kv_v = v.shape

    assert d == d_k == d_v, "q, k, v must share head dim d"
    assert n_q == n_q_heads, "q head count must match n_q_heads"
    assert n_kv == n_kv_v == n_kv_heads, "k, v head count must match n_kv_heads"
    assert seqlen_kv == seqlen_kv_v, "k and v must share seqlen_kv"
    assert d <= 128, "head dim d must fit the 128-wide partition axis"
    assert n_q_heads % n_kv_heads == 0, "n_q_heads must be a multiple of n_kv_heads"
    assert seqlen_kv % TILE_KV == 0, "seqlen_kv must be a multiple of TILE_KV (v1)"

    group = n_q_heads // n_kv_heads
    num_tiles = seqlen_kv // TILE_KV

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(d)

    out = nl.ndarray((n_q_heads, d), dtype=q.dtype, buffer=nl.shared_hbm)

    # One KV head at a time; its 'group' query heads ride together on the free axis 
    # so the shared K/V tile is loaded ONCE per group.
    for i_kv in nl.affine_range(n_kv_heads):
        # grouping the query heads: slice grabs the group w.r.t. n_kv_heads, then load from HBM -> SBUF.
        q_group = nl.load(q[:, i_kv * group:(i_kv + 1) * group])

        # running online-softmax state for the 'group' rows (lives across tiles)
        # SOFTMAX = final o = (Σ exp(logit - m)·v) /  Σ exp(logit - m) 
        # Keeping numerator and denominator UNNORMALIZED here and divide by l once at the end. 
        m_state = nl.full((group, 1), -np.inf, dtype=nl.float32, buffer=nl.sbuf)   # running max per query head, initialized to -inf
        acc = nl.zeros((group, d), dtype=nl.float32, buffer=nl.sbuf)              # acc = running Σ exp(logit - m)·v (numerator)
        
        l_state = nl.zeros((group, 1), dtype=nl.float32, buffer=nl.sbuf)        # running sum of exp(logits - m) -> denominator (normalizer)

        # sequential_range cuz tile i_t depends on tile i_t-1's (m, l, acc).
        for i_t in nl.sequential_range(num_tiles):
            kv_lo = i_t * TILE_KV

            k_tile = nl.load(k[i_kv, :, kv_lo:kv_lo + TILE_KV])   # [d, TILE_KV] since NKI wants d on the partition axis
            v_tile = nl.load(v[i_kv, :, kv_lo:kv_lo + TILE_KV])   # [d, TILE_KV]

            # logits  qk = scale * (q_groupᵀ @ k_tile)
            # contract over d (the partition axis) -> [group, TILE_KV]
            qk_psum = nl.matmul(q_group, k_tile, transpose_x=True)    # PSUM [group, TILE_KV]
            qk_unscaled = nl.ndarray(qk_psum.shape, dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(qk_unscaled, qk_psum)                    # PSUM -> SBUF
            qk = nl.ndarray(qk_unscaled.shape, dtype=qk_unscaled.dtype, buffer=nl.sbuf)
            nisa.tensor_scalar(qk, qk_unscaled, op0=nl.multiply, operand0=softmax_scale)

            # online-softmax update
            tile_max = nl.max(qk, axis=1, keepdims=True)             # [group, 1] max logit in THIS tile per query head
            new_m = nl.maximum(m_state, tile_max)                    # [group, 1] update running max

            # acc and l_state were built using m_old as the reference point, every exp was exp(logit - m_old).
            # The new tile uses m_new. Two different reference points can't be added directly.
            # rebase_factor = exp(m_old - m_new) converts the old running state into m_new's units by:
            #   exp(logit - m_old) * exp(m_old - m_new) => exp(logit - m_new)
            # Always in (0, 1] because m_new >= m_old, so the exponent is always <= 0.
            # First tile: m_old = -inf -> rebase_factor = 0 (wipes the empty state cleanly).
            rebase_exp_in = nl.ndarray(m_state.shape, dtype=m_state.dtype, buffer=nl.sbuf)
            nisa.tensor_scalar(rebase_exp_in, m_state, op0=nl.subtract, operand0=new_m)
            rebase_factor = nl.exp(rebase_exp_in)

            # p = exp(qk - new_m); new_m (a per-row scalar) broadcasts on the free axis
            norm = nl.ndarray(qk.shape, dtype=qk.dtype, buffer=nl.sbuf)
            nisa.tensor_scalar(norm, qk, op0=nl.subtract, operand0=new_m)
            p = nl.exp(norm)                                         # [group, TILE_KV]
            tile_l = nl.sum(p, axis=1, keepdims=True)               # [group,1]

            # l_state = l_state*rebase_factor + tile_l    (the denominator)
            l_scaled = nl.ndarray(l_state.shape, dtype=l_state.dtype, buffer=nl.sbuf)
            nisa.tensor_scalar(l_scaled, l_state, op0=nl.multiply, operand0=rebase_factor)
            new_l = nl.add(l_scaled, tile_l)                        # [group,1]

            # P @ V, contracting over TILE_KV
            # the contraction axis must sit on partition, so transpose both
            # operands to [TILE_KV, *] and evacuate (PSUM can't feed a matmul).
            p_t_psum = nl.transpose(p)                              # PSUM [TILE_KV, group]
            p_t = nl.ndarray(p_t_psum.shape, dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(p_t, p_t_psum)

            v_t_psum = nl.transpose(v_tile)                        # PSUM [TILE_KV, d]
            v_t = nl.ndarray(v_t_psum.shape, dtype=v_tile.dtype, buffer=nl.sbuf)
            nisa.tensor_copy(v_t, v_t_psum)

            pv_psum = nl.matmul(p_t, v_t, transpose_x=True)        # PSUM [group, d]
            pv = nl.ndarray(pv_psum.shape, dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(pv, pv_psum)

            # fold this tile into the accumulator: acc = acc*rebase_factor + pv
            acc_scaled = nl.ndarray(acc.shape, dtype=acc.dtype, buffer=nl.sbuf)
            nisa.tensor_scalar(acc_scaled, acc, op0=nl.multiply, operand0=rebase_factor)
            new_acc = nl.add(acc_scaled, pv)                       # [group, d]

            # commit the loop-carried state in place (all olds were read above).
            m_state[...] = new_m
            l_state[...] = new_l
            acc[...] = new_acc

        # finalize this group: o = acc / l
        inv_l = nl.reciprocal(l_state)               # [group, 1]
        o_group = nl.ndarray(acc.shape, dtype=q.dtype, buffer=nl.sbuf)
        nisa.tensor_scalar(o_group, acc, op0=nl.multiply, operand0=inv_l)

        nl.store(out[i_kv * group:(i_kv + 1) * group, :], value=o_group)

    return out


# =====================================================================
# Reference math implementations in NumPy, for testing the kernels above.
# =====================================================================
def numpy_decode_reference(q, k_cache, v_cache, scale):
    """
    Natural (math) layout, single head:
      - q: (d,)
      - k_cache: (seqlen_kv, d)
      - v_cache: (seqlen_kv, d)
      - returns o: (d,)
    """
    s = scale * (k_cache @ q)        # (seqlen_kv,)  logits
    s = s - s.max()                  # online-softmax max (stability)
    p = np.exp(s)                    # (seqlen_kv,)  unnormalized weights
    p = p / p.sum()                  # normalize
    o = p @ v_cache                  # (d,)  attention output
    return o


def numpy_decode_gqa_reference(q, k_cache, v_cache, n_q_heads, n_kv_heads, scale):
    """
    Natural (math) layout, GQA, single batch element. The oracle for decode_attention_gqa_fwd.
      - q: (n_q_heads, d)
      - k_cache: (n_kv_heads, seqlen_kv, d)
      - v_cache: (n_kv_heads, seqlen_kv, d)
      - returns o: (n_q_heads, d)
    Query head h is served by KV head (h // group), group = n_q_heads // n_kv_heads.
    This is exactly `repeat_kv` + per-head softmax attention, done the slow, obvious way.
    """
    group = n_q_heads // n_kv_heads
    d = q.shape[1]
    o = np.empty((n_q_heads, d), dtype=np.float32)
    for h in range(n_q_heads):
        kv = h // group                      # which KV head this query head shares
        s = scale * (k_cache[kv] @ q[h])     # (seqlen_kv,)  logits
        s = s - s.max()                      # stability
        p = np.exp(s)
        p = p / p.sum()                      # softmax over cached tokens
        o[h] = p @ v_cache[kv]               # (d,)  blend of values
    return o


# =====================================================================
# Local correctness check: runs on CPU via nki.simulate_kernel (no device).
# On a real Neuron instance, swap to:
#   out = nki.baremetal()(decode_attention_fwd)(q_t, k_t, v_t, scale)
# =====================================================================
def check_correct():
    np.random.seed(42)
    d, seqlen_kv = 128, 128
    scale = 1.0 / math.sqrt(d)

    # natural-layout random inputs (where the numbers come from is irrelevant)
    q = np.random.randn(d).astype(np.float32)
    k_cache = np.random.randn(seqlen_kv, d).astype(np.float32)
    v_cache = np.random.randn(seqlen_kv, d).astype(np.float32)

    ref = numpy_decode_reference(q, k_cache, v_cache, scale)        # (d,)

    # kernel layout: d on the partition axis -> transpose K, V
    q_t = q.reshape(d, 1).astype(np.float32)                       # (d, 1)
    k_t = np.ascontiguousarray(k_cache.T)                         # (d, seqlen_kv)
    v_t = np.ascontiguousarray(v_cache.T)                         # (d, seqlen_kv)

    out = nki.simulate_kernel(decode_attention_fwd, q_t, k_t, v_t, scale)
    out = np.asarray(out).reshape(-1)                             # (d,)

    max_diff = float(np.abs(out - ref).max())
    ok = np.allclose(out, ref, atol=1e-2, rtol=1e-2)
    print(f"[check_correct] d={d} seqlen_kv={seqlen_kv}  max|diff|={max_diff:.3e}")
    print("PASS" if ok else "FAIL")
    return ok


def check_correct_gqa():
    np.random.seed(42)
    d = 128
    seqlen_kv = 512                  # 4 tiles of TILE_KV=128 -> exercises online softmax
    n_q_heads, n_kv_heads = 8, 2     # group = 4 (real GQA). Try (4, 4) for group=1 first.
    scale = 1.0 / math.sqrt(d)

    # natural-layout random inputs
    q = np.random.randn(n_q_heads, d).astype(np.float32)
    k_cache = np.random.randn(n_kv_heads, seqlen_kv, d).astype(np.float32)
    v_cache = np.random.randn(n_kv_heads, seqlen_kv, d).astype(np.float32)

    ref = numpy_decode_gqa_reference(q, k_cache, v_cache, n_q_heads, n_kv_heads, scale)

    # kernel layout: d on the partition axis -> move d to the front
    q_t = np.ascontiguousarray(q.T)                          # (d, n_q_heads)
    k_t = np.ascontiguousarray(k_cache.transpose(0, 2, 1))   # (n_kv_heads, d, seqlen_kv)
    v_t = np.ascontiguousarray(v_cache.transpose(0, 2, 1))   # (n_kv_heads, d, seqlen_kv)

    out = nki.simulate_kernel(decode_attention_gqa_fwd, q_t, k_t, v_t, n_q_heads, n_kv_heads, scale)
    out = np.asarray(out)            # (n_q_heads, d)

    max_diff = float(np.abs(out - ref).max())
    ok = np.allclose(out, ref, atol=1e-2, rtol=1e-2)
    print(f"[check_correct_gqa] d={d} seqlen_kv={seqlen_kv} "
          f"group={n_q_heads // n_kv_heads}  max|diff|={max_diff:.3e}")
    print("PASS" if ok else "FAIL")
    
    return ok


def main():
    check_correct()        # Milestone A
    check_correct_gqa()    # Milestone B


if __name__ == "__main__":
    main()
