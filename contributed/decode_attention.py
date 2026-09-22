"""
Decode (flash-decoding) attention kernel.

Decode step of autoregressive attention: a single query position (seqlen_q = 1) attending over a cached K/V of length seqlen_kv. 
This is the memory-bound complement to the compute-bound prefill kernel in `pipelined_attention.py`.

Two kernels live here. 
`decode_attention_fwd` is the simplest correct version: one head, one KV tile, seqlen_kv <= 128. 
`decode_attention_gqa_fwd` lifts that length limit with KV tiling and an online softmax, and 
adds grouped-query attention (GQA) so query heads sharing a KV head also share its K/V loads.

Author: Varun (varuntej.dev@gmail.com)

Validation:
   - Numerics are checked against the NumPy references in this file, via
     check_correct / check_correct_gqa. The same checks run two ways:
     on CPU through nki.simulate, which needs no device, and on a NeuronDevice
     by calling the kernel directly.
   - Validated on Trn2 during upstream review of #129, and on Inf2
     (NeuronCore-v2) by the on-device path added here.
   - No latency numbers here. Timing a plain kernel(*args) call measures the
     compiler rather than the kernel: on NKI 0.6.0 the standalone path re-runs
     the frontend on every invocation, ~1.5 s per call on Inf2.
     Latency lives in decode_attention_benchmark.py, which compiles once
     through the parser frontend and replays the NEFF. It needs a device;
     the checks here do not, which is why the two are separate files.

   Requires NKI 0.6.0 or newer (Neuron SDK 2.32+). Earlier releases exposed
   nki.simulate_kernel / nki.baremetal / nki.benchmark, which are gone now;
   the neuronxcc.nki versions that remain cannot drive a top-level @nki.jit kernel.

   Inputs are fp32. bf16 does not compile on NeuronCore-v2, because the
   tensor engine requires an fp32 matmul destination there and nl.matmul
   takes its destination dtype from the operands. See BF16_SUPPORTED.

   Run `python decode_attention.py` for the numeric checks; the backend is
   auto-detected, so it simulates on a machine with no Neuron device.

WARNING: These kernels:
   - Have not been tested across all input configurations
   - Carry no compatibility guarantees
   - May change without prior notice

Status:
   - [A] done: single-head, single-tile decode (MHA, seqlen_kv <= 128)
   - [B] done: KV tiling + online softmax + GQA (decode_attention_gqa_fwd)
   - [C] planned: flash-decoding split-KV for long context

"""
import argparse
import math
import os
import sys

import numpy as np

import nki
# nisa - Neuron Instruction Set Architecture. This is the low-level API to Neuron hardware.
import nki.isa as nisa
import nki.language as nl

# bf16 inputs need ml_dtypes for the NumPy side.
# Optional: without it the file still runs, it just skips the bf16 cases.
try:
    from ml_dtypes import bfloat16
except ImportError:
    bfloat16 = None

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
    # Plain builtin range: on NKI 0.6.0 nl.affine_range / nl.sequential_range /
    # nl.static_range are deprecated aliases whose backend implementations all
    # `return range(start, stop, step)` verbatim, so the name carries no meaning
    # to the compiler. Iterations here are independent anyway: each gets its own
    # softmax state and touches disjoint slices of q and out.
    for i_kv in range(n_kv_heads):
        # grouping the query heads: slice grabs the group w.r.t. n_kv_heads, then load from HBM -> SBUF.
        q_group = nl.load(q[:, i_kv * group:(i_kv + 1) * group])

        # running online-softmax state for the 'group' rows (lives across tiles)
        # SOFTMAX = final o = (Σ exp(logit - m)·v) /  Σ exp(logit - m) 
        # Keeping numerator and denominator UNNORMALIZED here and divide by l once at the end. 
        m_state = nl.full((group, 1), -np.inf, dtype=nl.float32, buffer=nl.sbuf)   # running max per query head, initialized to -inf
        acc = nl.zeros((group, d), dtype=nl.float32, buffer=nl.sbuf)              # acc = running Σ exp(logit - m)·v (numerator)
        
        l_state = nl.zeros((group, 1), dtype=nl.float32, buffer=nl.sbuf)        # running sum of exp(logits - m) -> denominator (normalizer)

        # Order matters here: tile i_t reads tile i_t-1's (m, l, acc). The loop is
        # unrolled in program order, which is what keeps the carried state correct.
        for i_t in range(num_tiles):
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
# Test harness.
# =====================================================================
# Two ways to run a kernel:
#
#   simulate   nki.simulate(kernel)(*args)   CPU, no device. Real outputs. Slow.
#   device     kernel(*args)                 NeuronDevice. Real outputs.
#
# On NKI 0.6.0 a @nki.jit kernel called with numpy arrays
# "compiles and executes standalone, without a framework" (nki.jit's own docstring),
# so a plain call IS the on-device path. That is why nki.baremetal no longer exists.
# nki.simulate_kernel is gone the same way, replaced by nki.simulate.
#
# Note for anyone porting older NKI samples: nki.baremetal, nki.benchmark and
# nki.simulate_kernel still exist under the deprecated neuronxcc.nki namespace,
# but they cannot drive a kernel decorated with the current top-level @nki.jit.
# They raise AttributeError: 'Kernel' object has no attribute 'grid',
# because they expect the older TraceKernel object.
#
# There is deliberately no latency benchmark here. The standalone path
# recompiles on every call: nki/framework/compiled.py passes enable_cache=False
# to compile_kernel_to_nir, so timing kernel(*args) measures the compiler, not
# the kernel. Measured that way a decode step over a 1 MB cache "takes" ~1.5 s
# steady state (~10 s on the first call), against tens of microseconds of
# actual kernel time.
#
# A compile-once benchmark is possible. It goes through the parser frontend,
# ParserFrontend().compile() -> CompiledKernel.from_frontend() -> CompiledKernel.benchmark(),
# which is the path @nki.jit itself takes. Note that nki.compiler.kernel_builder.compile_kernel
# is NOT that path: it is a separate authoring API whose kernel arguments arrive as TileViews,
# so a kernel written against nl/nisa cannot be compiled by it. Benchmarks are a
# separate change rather than a silently wrong column in this one.

# bf16 inputs do not compile on NeuronCore-v2 (gen2: inf2, trn1).
# nl.matmul infers its PSUM destination dtype from the operands,
# and the tensor engine rejects a non-fp32 matmul destination on gen2:
#
#   `nc_matmul dst dtype must be float32 on gen2, got bfloat16`
#
# Fixing it means replacing nl.matmul with an explicit fp32 PSUM tile plus nisa.nc_matmul
# at all four matmul sites, and passing dtype=nl.float32 to nl.transpose at the
# four transpose sites, since a transpose also runs on the tensor engine.
# That is a change to the kernels
# rather than to this harness, so it is left for a follow-up. gen3 (trn2) appears to accept a
# bf16 destination, which is why upstream review on Trn2 never hit this.
BF16_SUPPORTED = False


def _dtype_name(dtype):
    return np.dtype(dtype).name


def _quantize(x, dtype):
    """Round fp32 data to the kernel's input dtype, then back to fp32.

    The kernel gets the low-precision values; the NumPy reference gets the
    *same* values widened back to fp32. That isolates what we actually want
    to measure (kernel error given low-precision inputs) from NumPy's own
    low-precision arithmetic, which is a different question.
    """
    narrowed = x.astype(dtype)
    return narrowed, narrowed.astype(np.float32)


def _make_mha_inputs(d=128, seqlen_kv=128, dtype=np.float32, seed=42):
    """Build inputs for decode_attention_fwd. Returns (args, ref, meta)."""
    rng = np.random.default_rng(seed)
    scale = 1.0 / math.sqrt(d)

    q = rng.standard_normal(d).astype(np.float32)
    k_cache = rng.standard_normal((seqlen_kv, d)).astype(np.float32)
    v_cache = rng.standard_normal((seqlen_kv, d)).astype(np.float32)

    q, q_ref = _quantize(q, dtype)
    k_cache, k_ref = _quantize(k_cache, dtype)
    v_cache, v_ref = _quantize(v_cache, dtype)

    ref = numpy_decode_reference(q_ref, k_ref, v_ref, scale)        # (d,)

    # kernel layout: d on the partition axis -> transpose K, V.
    # ascontiguousarray is load-bearing: nl.load slices assume a C-contiguous
    # HBM tensor in exactly this layout, and a bare .T is only a view.
    q_t = q.reshape(d, 1)                                          # (d, 1)
    k_t = np.ascontiguousarray(k_cache.T)                          # (d, seqlen_kv)
    v_t = np.ascontiguousarray(v_cache.T)                          # (d, seqlen_kv)

    meta = dict(kernel="mha", d=d, seqlen_kv=seqlen_kv, n_q_heads=1,
                n_kv_heads=1, group=1, dtype=dtype)
    return (q_t, k_t, v_t, scale), ref, meta


def _make_gqa_inputs(d=128, seqlen_kv=512, n_q_heads=8, n_kv_heads=2,
                     dtype=np.float32, seed=42):
    """Build inputs for decode_attention_gqa_fwd. Returns (args, ref, meta)."""
    rng = np.random.default_rng(seed)
    scale = 1.0 / math.sqrt(d)

    q = rng.standard_normal((n_q_heads, d)).astype(np.float32)
    k_cache = rng.standard_normal((n_kv_heads, seqlen_kv, d)).astype(np.float32)
    v_cache = rng.standard_normal((n_kv_heads, seqlen_kv, d)).astype(np.float32)

    q, q_ref = _quantize(q, dtype)
    k_cache, k_ref = _quantize(k_cache, dtype)
    v_cache, v_ref = _quantize(v_cache, dtype)

    ref = numpy_decode_gqa_reference(q_ref, k_ref, v_ref,
                                     n_q_heads, n_kv_heads, scale)

    # kernel layout: d on the partition axis -> move d to the front.
    q_t = np.ascontiguousarray(q.T)                                # (d, n_q_heads)
    k_t = np.ascontiguousarray(k_cache.transpose(0, 2, 1))         # (n_kv, d, seqlen_kv)
    v_t = np.ascontiguousarray(v_cache.transpose(0, 2, 1))         # (n_kv, d, seqlen_kv)

    meta = dict(kernel="gqa", d=d, seqlen_kv=seqlen_kv, n_q_heads=n_q_heads,
                n_kv_heads=n_kv_heads, group=n_q_heads // n_kv_heads, dtype=dtype)
    return (q_t, k_t, v_t, n_q_heads, n_kv_heads, scale), ref, meta


def _run(kernel, args, backend="simulate"):
    """Run kernel(*args) and return its output as an ndarray."""
    if backend == "simulate":
        return np.asarray(nki.simulate(kernel)(*args))
    if backend == "baremetal":
        # A plain call is the on-device path on NKI 0.6.0. See the note at
        # the top of this section.
        return np.asarray(kernel(*args))
    raise ValueError(f"unknown backend: {backend!r}")


def _check_dtypes():
    """Which input dtypes this host can actually run, and why if fewer."""
    if not BF16_SUPPORTED:
        print("note: bf16 skipped. nl.matmul cannot target a non-fp32 PSUM "
              "destination on gen2 (inf2/trn1). See BF16_SUPPORTED.")
    elif bfloat16 is None:
        print("note: bf16 skipped, ml_dtypes not installed "
              "(pip install ml_dtypes).")
    else:
        return [np.float32, bfloat16]
    return [np.float32]


def check_correct(backend="simulate", dtype=np.float32, d=128, seqlen_kv=128):
    """Milestone A: single head, single KV tile."""
    args, ref, _ = _make_mha_inputs(d=d, seqlen_kv=seqlen_kv, dtype=dtype)
    out = _run(decode_attention_fwd, args, backend=backend)
    out = out.reshape(-1).astype(np.float32)                       # (d,)

    max_diff = float(np.abs(out - ref).max())
    ok = np.allclose(out, ref, atol=1e-2, rtol=1e-2)
    print(f"[check_correct] {backend:9s} {_dtype_name(dtype):8s} "
          f"d={d} seqlen_kv={seqlen_kv} max|diff|={max_diff:.3e}  "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


def check_correct_gqa(backend="simulate", dtype=np.float32, d=128,
                      seqlen_kv=512, n_q_heads=8, n_kv_heads=2):
    """Milestone B: KV tiling + online softmax + GQA.

    seqlen_kv=512 is four TILE_KV tiles, so the online-softmax rescale path
    actually runs. group=4 makes it real GQA rather than the degenerate case.
    """
    args, ref, meta = _make_gqa_inputs(d=d,
                                       seqlen_kv=seqlen_kv,
                                       n_q_heads=n_q_heads,
                                       n_kv_heads=n_kv_heads,
                                       dtype=dtype
                                       )
    out = _run(decode_attention_gqa_fwd, args, backend=backend)
    out = out.astype(np.float32)                                   # (n_q_heads, d)

    max_diff = float(np.abs(out - ref).max())
    ok = np.allclose(out, ref, atol=1e-2, rtol=1e-2)
    
    print(f"[check_correct_gqa] {backend:9s} {_dtype_name(dtype):8s} "
          f"d={d} seqlen_kv={seqlen_kv} group={meta['group']}  "
          f"max|diff|={max_diff:.3e}  {'PASS' if ok else 'FAIL'}")
    return ok


def check_all(backend="simulate"):
    """Both kernels, every input dtype this host supports."""
    results = []
    for dtype in _check_dtypes():
        results.append(check_correct(backend=backend, dtype=dtype))
        results.append(check_correct_gqa(backend=backend, dtype=dtype))

    print(f"\n{sum(results)}/{len(results)} checks passed")
    return all(results)


# =====================================================================

def _auto_backend():
    """Use the device if there is one, otherwise fall back to CPU simulation,
    so `python decode_attention.py` does the right thing either way."""
    return "baremetal" if os.path.exists("/dev/neuron0") else "simulate"


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Decode (flash-decoding) attention kernels: correctness checks.")

    parser.add_argument("--backend",
                        choices=("simulate", "baremetal"),
                        default=None,
                        help="default: baremetal if a Neuron device is present")
    args = parser.parse_args(argv)

    return 0 if check_all(backend=args.backend or _auto_backend()) else 1


if __name__ == "__main__":
    sys.exit(main())
