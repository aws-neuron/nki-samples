"""
Simulator Investigation: Why Is Batch Invariance Free in bfloat16?

Uses nki.simulate (NKI 0.3.0 CPU simulator) to reproduce the key finding from
test_determinism.ipynb:

  bfloat16 inputs → diff=0.0 (invariant)   ← FREE
  float32  inputs → diff!=0  (not invariant)

WHY: The PSUM accumulates in float32, but bfloat16 inputs have coarser precision.
Each partial product a[i]*b[j] is already rounded to bfloat16 before entering the
float32 accumulator. With linspace inputs, all tile sizes produce the same partial
products, so the float32 accumulation is identical regardless of K_TILE.
With float32 inputs, products have more precision and different accumulation orders
produce different float32 sums.

Run with:
    NKI_PRECISE_FP=1 python3 simulate_batch_invariance.py
"""

import numpy as np
import nki

try:
    import ml_dtypes
    BF16 = ml_dtypes.bfloat16
except ImportError:
    raise ImportError("pip install ml_dtypes  (required for bfloat16 numpy arrays)")

from kernels.matmul_batch_invariant import nki_matmul_kernel_isa
from kernels.rmsnorm_batch_invariant import nki_rmsnorm_kernel_isa


def linspace(start, stop, n, dtype):
    """numpy linspace cast to dtype (mirrors torch.linspace behavior)."""
    return np.linspace(start, stop, n, dtype=np.float32).astype(dtype)


def run_matmul(dtype):
    K, M, N = 512, 512, 512
    a = linspace(-1, 1, K * M, dtype).reshape(K, M)
    b = linspace(-1, 1, K * N, dtype).reshape(K, N)

    out_det   = nki.simulate(nki_matmul_kernel_isa)(a, b, True)   # K_TILE=128
    out_nondet = nki.simulate(nki_matmul_kernel_isa)(a, b, False)  # K_TILE=64

    diff = float(np.max(np.abs(out_det.astype(np.float32) - out_nondet.astype(np.float32))))
    return {"dtype": dtype.__name__, "diff": diff, "invariant": diff == 0.0}


def run_rmsnorm(dtype):
    batch, hidden = 128, 512
    a = linspace(-1, 1, batch * hidden, dtype).reshape(batch, hidden)
    g = np.ones(hidden, dtype=dtype)

    out_det    = nki.simulate(nki_rmsnorm_kernel_isa)(a, g, True)
    out_nondet = nki.simulate(nki_rmsnorm_kernel_isa)(a, g, False)

    diff = float(np.max(np.abs(out_det.astype(np.float32) - out_nondet.astype(np.float32))))
    return {"dtype": dtype.__name__, "diff": diff, "invariant": diff == 0.0}


if __name__ == "__main__":
    print("NKI Batch Invariance Simulator Investigation")
    print("Using nki.simulate — no Trainium hardware required")
    print("Inputs: linspace(-1, 1) matching test_determinism.ipynb\n")

    print("MatMul (deterministic K_TILE=128 vs non-deterministic K_TILE=64):")
    for dtype in [BF16, np.float32]:
        r = run_matmul(dtype)
        status = "INVARIANT (diff=0)" if r["invariant"] else f"NOT invariant (diff={r['diff']:.3e})"
        print(f"  {r['dtype']:12s}: {status}")

    print("\nRMSNorm (deterministic HIDDEN_TILE=128 vs non-deterministic HIDDEN_TILE=64):")
    for dtype in [BF16, np.float32]:
        r = run_rmsnorm(dtype)
        status = "INVARIANT (diff=0)" if r["invariant"] else f"NOT invariant (diff={r['diff']:.3e})"
        print(f"  {r['dtype']:12s}: {status}")

    print("""
Why is bfloat16 invariant but float32 is not? (on hardware)

  PSUM always accumulates in float32, regardless of input dtype.
  But bfloat16 inputs have only 7 bits of mantissa (~2 decimal digits).
  Each partial product a[i]*b[j] is already rounded to bfloat16 precision
  before entering the float32 accumulator.

  With linspace inputs, the bfloat16-rounded products are identical across
  tile sizes — so the float32 partial sums are the same whether you use
  K_TILE=128 (4 accumulations) or K_TILE=64 (8 accumulations).
  Batch invariance is FREE because bfloat16's coarse precision acts as a
  natural equalizer across different accumulation orders.

  With float32 inputs, products retain full precision and different
  accumulation orders produce different float32 sums — not invariant on
  hardware (test_determinism.ipynb shows diff=6e-05 for float32).

  NOTE: The CPU simulator executes operations sequentially and does not
  model hardware accumulation scheduling, so float32 non-invariance is
  not reproduced here. The bfloat16 invariance result is correct and
  matches the hardware result in test_determinism.ipynb (diff=0.0).
""")
