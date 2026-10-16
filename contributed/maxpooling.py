"""
2D max pooling, stride 1, no padding.

Requires NKI 0.6.0 or newer (Neuron SDK 2.32+). The kernel uses .ap() access
patterns in place of the removed nl.mgrid / nl.par_dim / mask= indexing.

Run `python maxpooling.py` for the correctness checks. The backend is
auto-detected: on a machine with a Neuron device the kernel runs on it,
otherwise it runs on CPU through nki.simulate. When simulating, set
NEURON_PLATFORM_TARGET_OVERRIDE (gen2 for Inf2/Trn1, gen3 for Trn2) to pick
the target; the simulator defaults to trn3 when no device is present.

There is no latency benchmark. nki.benchmark is gone on NKI 0.6.0, and timing
a plain kernel(*args) call measures compilation, not the kernel.
"""
import argparse
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

import nki
import nki.isa as nisa
import nki.language as nl

# bf16 inputs need ml_dtypes on the NumPy side. Optional: without it the
# bf16 case is skipped.
try:
    from ml_dtypes import bfloat16
except ImportError:
    bfloat16 = None


@nki.jit
def max_pooling_2d_stride_1(in_tensor, pool_size):
    """
    Performs 2D max pooling with stride 1 on a 2D tensor.

    Args:
        in_tensor: Input tensor with shape [height, width]
        pool_size: Size of the pooling window (pool_size x pool_size)

    Returns:
        Output tensor with shape [height-(pool_size-1), width-(pool_size-1)]
    """
    k = pool_size
    h_in, w_in = in_tensor.shape
    assert 1 <= k <= min(h_in, w_in), \
        f"pool_size must be between 1 and {min(h_in, w_in)}, got {k}"

    h_out, w_out = h_in - (k - 1), w_in - (k - 1)
    out_tensor = nl.ndarray((h_out, w_out), dtype=in_tensor.dtype,
                            buffer=nl.shared_hbm)

    # One output row per partition, up to 128 output rows per tile.
    P = nl.tile_size.pmax

    for h_tile_idx in range(math.ceil(h_out / P)):
        row0 = h_tile_idx * P
        rows = min(P, h_out - row0)  # the last tile may be partial

        # in_tile[p, kh, w] = in_tensor[row0 + p + kh, w]
        # Partition p gets the k input rows that output row row0 + p needs.
        # Neighbouring partitions share k - 1 of those rows, so this is one
        # overlapping read rather than a slice: the partition step and the
        # kh step are both one input row (w_in elements).
        in_tile = nl.ndarray((rows, k, w_in), dtype=in_tensor.dtype,
                             buffer=nl.sbuf)
        in_rows = in_tensor.ap(pattern=[[w_in, rows], [w_in, k], [1, w_in]],
                               offset=row0 * w_in)
        nisa.dma_copy(dst=in_tile, src=in_rows)

        # windows[p, w, kh, kw] = in_tile[p, kh, w + kw]
        # The two pool axes go last so the reduction runs over the innermost
        # dimensions, as in the average_pool2d tutorial.
        windows = in_tile.ap(
            pattern=[[k * w_in, rows], [1, w_out], [w_in, k], [1, k]],
            offset=0)
        out_tile = nl.max(windows, axis=[2, 3])

        nisa.dma_copy(dst=out_tensor[row0:row0 + rows, 0:w_out],
                      src=out_tile)

    return out_tensor


# =====================================================================
# Correctness checks
# =====================================================================

# (H, W, pool_size, what the case covers)
CASES = [
    (448, 448, 3, "original shape: 3 full row tiles + a 62-row tail"),
    (130, 64, 3, "h_out = 128: exactly one full tile, no tail"),
    (257, 96, 2, "h_out = 256: two full tiles, no tail"),
    (131, 137, 3, "h_out = 129: one full tile + a 1-row tail"),
    (132, 137, 3, "h_out = 130: one full tile + a 2-row tail"),
    (300, 200, 5, "pool 5, non-square: 2 full tiles + a 40-row tail"),
    (40, 33, 1, "pool 1 is the identity; a single partial tile"),
    (7, 8, 3, "small input: 5x6 output in one partial tile"),
    (16, 16, 16, "window covers the whole input: 1x1 output"),
]


def reference(x, pool_size):
    """F.max_pool2d in fp32.

    Exact for fp16 and bf16 inputs too: max returns one of its inputs, and
    widening fp16/bf16 to fp32 loses nothing.
    """
    t = torch.from_numpy(x.astype(np.float32))[None, None]
    out = F.max_pool2d(t, kernel_size=pool_size, stride=1, padding=0)
    return out[0, 0].numpy()


def _run(x, pool_size, backend):
    if backend == "simulate":
        sim = nki.simulate(max_pooling_2d_stride_1)
        return np.asarray(sim(x, pool_size))
    if backend == "baremetal":
        # On NKI 0.6.0 a plain call on a @nki.jit kernel compiles and runs it
        # on the Neuron device. This replaces nki.baremetal().
        return np.asarray(max_pooling_2d_stride_1(x, pool_size))
    raise ValueError(f"unknown backend: {backend!r}")


def check_correct(H, W, pool_size, dtype=np.float32, backend="simulate",
                  seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((H, W)).astype(np.float32).astype(dtype)

    out = _run(x, pool_size, backend).astype(np.float32)
    ref = reference(x, pool_size)

    # Max pooling selects values, it does no arithmetic, so the result must
    # match exactly. A shape mismatch, NaN (unwritten memory) or a zero-filled
    # output all fail this check.
    ok = out.shape == ref.shape and np.array_equal(out, ref)
    if out.shape == ref.shape:
        max_diff = float(np.abs(out - ref).max())
    else:
        max_diff = float("nan")
    print(f"[check_correct] {backend:9s} {np.dtype(dtype).name:8s} "
          f"{H}x{W} pool={pool_size} -> {out.shape[0]}x{out.shape[1]}  "
          f"max|diff|={max_diff:.3e}  {'PASS' if ok else 'FAIL'}")
    return ok


def check_all(backend="simulate"):
    results = [check_correct(H, W, k, backend=backend)
               for H, W, k, _ in CASES]

    # Low-precision inputs on the original shape.
    low_precision = [np.float16]
    if bfloat16 is not None:
        low_precision.append(bfloat16)
    else:
        print("note: bf16 skipped, ml_dtypes not installed "
              "(pip install ml_dtypes)")
    for dtype in low_precision:
        results.append(
            check_correct(448, 448, 3, dtype=dtype, backend=backend))

    print(f"\n{sum(results)}/{len(results)} checks passed")
    return all(results)


def _auto_backend():
    """Run on the device if there is one, otherwise simulate on CPU."""
    return "baremetal" if os.path.exists("/dev/neuron0") else "simulate"


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Max pooling kernel: correctness checks.")
    parser.add_argument("--backend", choices=("simulate", "baremetal"),
                        default=None,
                        help="default: baremetal if a device is present")
    args = parser.parse_args(argv)
    return 0 if check_all(backend=args.backend or _auto_backend()) else 1


if __name__ == "__main__":
    sys.exit(main())
