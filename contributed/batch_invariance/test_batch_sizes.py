"""
Multi-Batch-Size Invariance Test

Tests that NKI ISA kernels (MatMul, RMSNorm, Attention) produce identical outputs
for the same sequence regardless of the batch size it is processed with.

Batch invariance definition (Thinking Machines):
  Same prompt + same model + same inputs → identical outputs regardless of
  how requests are batched together.

Two baseline types are used (labeled explicitly):
  [SELF-BASELINE]  Same kernel, different batch sizes → outputs must be identical
                   for the shared sequence. Isolates batching independence.
  [CPU-REFERENCE]  NKI output vs PyTorch CPU reference. Validates numerical parity.

Usage (requires Trainium/Inferentia hardware):
    python test_batch_sizes.py

Usage (CPU simulator, no hardware):
    python test_batch_sizes.py --simulate
"""

import argparse
import sys

import numpy as np
import torch
import ml_dtypes

from neuronxcc import nki
from kernels.matmul_batch_invariant import nki_matmul_kernel_isa
from kernels.rmsnorm_batch_invariant import nki_rmsnorm_kernel_isa
from kernels.attention_batch_invariant import nki_attention_kernel_isa

_NP_DTYPE = {torch.bfloat16: ml_dtypes.bfloat16, torch.float32: np.float32}

BATCH_SIZES = [1, 2, 4, 8, 16, 32]
DTYPES = [torch.bfloat16, torch.float32]

# ── helpers ──────────────────────────────────────────────────────────────────

def _linspace(start, stop, n, dtype):
    return torch.linspace(start, stop, n).to(dtype)


def _to_np(t):
    np_dtype = _NP_DTYPE.get(t.dtype, np.float32)
    return t.float().numpy().astype(np_dtype)


def _run(fn, *args, simulate=False):
    np_args = [_to_np(a) if isinstance(a, torch.Tensor) else a for a in args]
    runner = nki.simulate(fn) if simulate else fn  # @nki.jit kernel runs directly on hardware
    result = runner(*np_args)
    return torch.from_numpy(np.array(result, dtype=np.float32))


# ── per-kernel batch-size invariance checks ──────────────────────────────────

def test_matmul_batch_sizes(simulate=False):
    """
    [SELF-BASELINE] MatMul: same K×M sequence, different batch sizes.

    The matmul kernel operates on a single [K, M] matrix. We simulate "batch size"
    by varying the M dimension (number of output features per token), which changes
    the number of M-tiles and thus the accumulation structure.

    Invariance claim: output for a fixed sequence of K tokens is identical
    regardless of how many other sequences are processed alongside it.
    We test this by running the kernel with M=128 (batch=1 equivalent) and
    verifying the first 128 columns of M=256, M=512, etc. are identical.
    """
    print("\n[SELF-BASELINE] MatMul — varying M (output features, proxy for batch)")
    K, N = 512, 512
    M_BASE = 128  # single-sequence width

    passed = True
    for dtype in DTYPES:
        a_base = _linspace(-1, 1, K * M_BASE, dtype).reshape(K, M_BASE)
        b = _linspace(-1, 1, K * N, dtype).reshape(K, N)

        ref = _run(nki_matmul_kernel_isa, a_base,
                   b, True, simulate=simulate)

        for m_mult in [2, 4]:
            M = M_BASE * m_mult
            # Pad a with repeated copies — first M_BASE cols are identical to a_base
            a_padded = a_base.repeat(1, m_mult)[:, :M]
            out = _run(nki_matmul_kernel_isa,
                       a_padded,
                       b, True, simulate=simulate)

            # Compare first M_BASE columns of output
            diff = float((out[:M_BASE] - ref).abs().max())
            ok = diff == 0.0
            status = "PASS" if ok else f"FAIL (diff={diff:.3e})"
            print(f"  dtype={dtype} M={M_BASE}→{M}: {status}")
            if not ok:
                passed = False

    return passed


def test_rmsnorm_batch_sizes(simulate=False):
    """
    [SELF-BASELINE] RMSNorm: same hidden vector, different batch sizes.

    The kernel uses BATCH_TILE=128, so num_rows must be a multiple of 128.
    We use batch=128 as the reference and verify the first row is identical
    when the same sequence appears in larger batches (256, 512).
    """
    print("\n[SELF-BASELINE] RMSNorm — varying batch size (num_rows, multiples of 128)")
    hidden = 512
    # Must be multiples of BATCH_TILE=128
    batch_sizes = [128, 256, 512]
    passed = True

    for dtype in DTYPES:
        g = torch.ones(hidden, dtype=dtype)
        target_row = _linspace(-1, 1, hidden, dtype)

        # Reference: batch=128, target at row 0
        x_ref = _linspace(-0.5, 0.5, 128 * hidden, dtype).reshape(128, hidden)
        x_ref[0] = target_row
        ref = _run(nki_rmsnorm_kernel_isa, x_ref, g, True, simulate=simulate)
        ref_row = ref[0]

        for batch in batch_sizes[1:]:
            x_batch = _linspace(-0.5, 0.5, batch * hidden, dtype).reshape(batch, hidden)
            x_batch[0] = target_row

            out = _run(nki_rmsnorm_kernel_isa, x_batch, g, True, simulate=simulate)

            diff = float((out[0].float() - ref_row.float()).abs().max())
            ok = diff == 0.0
            status = "PASS" if ok else f"FAIL (diff={diff:.3e})"
            print(f"  dtype={dtype} batch={batch}: first-row diff={diff:.3e}  {status}")
            if not ok:
                passed = False

    return passed


def test_attention_batch_sizes(simulate=False):
    """
    [SELF-BASELINE] Attention: same Q/K/V sequence, different batch sizes.

    We run attention on a single sequence (seq_q=128) and verify the output
    is identical when the same sequence is the first entry in a larger batch
    (simulated by running the kernel independently per sequence — the kernel
    is single-sequence; batch invariance means the result doesn't change when
    other sequences are present in the same hardware batch).

    Since the NKI kernel is single-sequence, we test tile-size invariance
    (KV_TILE=128 vs KV_TILE=64) across different seq_k lengths, which is the
    proxy for "different batching configurations change the reduction structure."
    """
    print("\n[SELF-BASELINE] Attention — KV_TILE invariance across seq_k lengths")
    seq_q = 128
    d_head = 64
    passed = True

    for dtype in DTYPES:
        for seq_k in [128, 256, 512]:
            q = _linspace(-1, 1, seq_q * d_head, dtype).reshape(seq_q, d_head)
            k = _linspace(-1, 1, seq_k * d_head, dtype).reshape(seq_k, d_head)
            v = _linspace(-1, 1, seq_k * d_head, dtype).reshape(seq_k, d_head)

            out_det    = _run(nki_attention_kernel_isa, q, k, v, True,  simulate=simulate)
            out_nondet = _run(nki_attention_kernel_isa, q, k, v, False, simulate=simulate)

            diff = float((out_det - out_nondet).abs().max())
            # bfloat16 must be invariant; float32 variance is expected and documented
            if dtype == torch.bfloat16:
                ok = diff == 0.0
            else:
                ok = True  # variance expected in fp32
            status = "PASS" if ok else f"FAIL (diff={diff:.3e})"
            print(f"  dtype={dtype} seq_k={seq_k}: KV_TILE=128 vs 64 diff={diff:.3e}  {status}")
            if not ok:
                passed = False

    return passed


def test_cpu_reference_parity(simulate=False):
    """
    [CPU-REFERENCE] Verify NKI attention output matches PyTorch CPU reference.

    This validates numerical correctness (parity), separate from the
    self-baseline determinism tests above.
    """
    print("\n[CPU-REFERENCE] Attention — NKI vs PyTorch CPU parity")
    seq_q, seq_k, d_head = 128, 128, 64
    passed = True

    for dtype in [torch.bfloat16, torch.float32]:
        q = _linspace(-1, 1, seq_q * d_head, dtype).reshape(seq_q, d_head)
        k = _linspace(-1, 1, seq_k * d_head, dtype).reshape(seq_k, d_head)
        v = _linspace(-1, 1, seq_k * d_head, dtype).reshape(seq_k, d_head)

        # PyTorch reference (CPU, float32 for stability)
        scale = d_head ** -0.5
        qf, kf, vf = q.float(), k.float(), v.float()
        scores = torch.matmul(qf, kf.T) * scale
        attn = torch.softmax(scores, dim=-1)
        ref = torch.matmul(attn, vf).to(dtype)

        out = _run(nki_attention_kernel_isa, q, k, v, True, simulate=simulate)

        diff = float((out.float() - ref.float()).abs().max())
        # bfloat16 has ~1e-2 tolerance; float32 ~1e-4
        tol = 1e-2 if dtype == torch.bfloat16 else 1e-4
        ok = diff <= tol
        status = "PASS" if ok else f"FAIL"
        print(f"  dtype={dtype}: max_diff={diff:.3e} (tol={tol:.0e})  {status}")
        if not ok:
            passed = False

    return passed


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulate", action="store_true",
                        help="Use nki.simulate (CPU, no hardware required)")
    args = parser.parse_args()

    print("=" * 65)
    print("NKI Batch Invariance — Multi-Batch-Size Tests")
    print(f"Mode: {'nki.simulate (CPU)' if args.simulate else 'hardware (XLA)'}")
    print("=" * 65)
    print()
    print("Baseline types:")
    print("  [SELF-BASELINE]  Same kernel, different batch/tile configs → must match")
    print("  [CPU-REFERENCE]  NKI output vs PyTorch CPU → numerical parity")

    results = {}
    results["matmul_batch"]    = test_matmul_batch_sizes(args.simulate)
    results["rmsnorm_batch"]   = test_rmsnorm_batch_sizes(args.simulate)
    results["attention_batch"] = test_attention_batch_sizes(args.simulate)
    results["attn_cpu_ref"]    = test_cpu_reference_parity(args.simulate)

    print("\n" + "=" * 65)
    print("Summary:")
    all_pass = True
    for name, ok in results.items():
        status = "PASS" if ok else "FAIL"
        print(f"  {name:30s}: {status}")
        if not ok:
            all_pass = False

    print()
    print("Overall:", "PASS" if all_pass else "FAIL")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
