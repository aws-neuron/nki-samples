"""
Inspect PSUM accumulation: does the intermediate float32 sum differ
between K_TILE=128 and K_TILE=64 for bfloat16 vs float32 inputs?
"""

import numpy as np
import nki
import nki.isa as nisa
import nki.language as nl

try:
    import ml_dtypes
    BF16 = ml_dtypes.bfloat16
except ImportError:
    raise ImportError("pip install ml_dtypes")


@nki.jit
def matmul_dump_psum(a, b, k_tile):
    """Matmul that dumps the PSUM after every K tile accumulation."""
    K, M = a.shape
    N = b.shape[1]
    M_TILE = 128

    # One output slot per K tile to capture intermediate PSUM state
    n_tiles = K // k_tile
    snapshots = nl.ndarray((n_tiles, M_TILE, N), dtype=nl.float32, buffer=nl.shared_hbm)

    c_psum = nl.ndarray((M_TILE, N), dtype=nl.float32, buffer=nl.psum)

    for k in nl.static_range(n_tiles):
        a_tile = nl.ndarray((k_tile, M_TILE), dtype=a.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=a_tile, src=a[k*k_tile:(k+1)*k_tile, 0:M_TILE])

        b_tile = nl.ndarray((k_tile, N), dtype=b.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=b_tile, src=b[k*k_tile:(k+1)*k_tile, 0:N])

        nisa.nc_matmul(dst=c_psum, stationary=a_tile, moving=b_tile)

        # Snapshot the running PSUM (float32) after this accumulation
        snap = nl.ndarray((M_TILE, N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=snap, src=c_psum)
        nisa.dma_copy(dst=snapshots[k, 0:M_TILE, 0:N], src=snap)

    return snapshots


def inspect(dtype, label):
    K, M, N = 512, 128, 512
    a = np.linspace(-1, 1, K * M, dtype=np.float32).reshape(K, M).astype(dtype)
    b = np.linspace(-1, 1, K * N, dtype=np.float32).reshape(K, N).astype(dtype)

    snaps_128 = nki.simulate(matmul_dump_psum)(a, b, 128)  # 4 tiles
    snaps_64  = nki.simulate(matmul_dump_psum)(a, b, 64)   # 8 tiles

    # After K tiles accumulated, both should have processed the same K elements.
    # Compare PSUM after K=128 elements (tile 0 of 128-tiling vs tiles 0+1 of 64-tiling)
    psum_after_128_via_128 = snaps_128[0]                                    # 1 tile of 128
    psum_after_128_via_64  = snaps_64[0].astype(np.float32) + snaps_64[1].astype(np.float32)  # 2 tiles of 64 — but these are snapshots of running sum, so just use snap[1]
    psum_after_128_via_64  = snaps_64[1]  # running sum after 2×64 = 128 elements

    diff = np.max(np.abs(psum_after_128_via_128.astype(np.float32) -
                         psum_after_128_via_64.astype(np.float32)))

    # Also compare final PSUM (all K elements accumulated)
    final_128 = snaps_128[-1]
    final_64  = snaps_64[-1]
    final_diff = np.max(np.abs(final_128.astype(np.float32) -
                               final_64.astype(np.float32)))

    print(f"\n{label}")
    print(f"  PSUM after first 128 K-elements: K_TILE=128 vs K_TILE=64 → diff={diff:.6e}")
    print(f"  PSUM after all 512 K-elements:   K_TILE=128 vs K_TILE=64 → diff={final_diff:.6e}")
    print(f"  Sample PSUM values (K_TILE=128, row 0, cols 0-3): {final_128[0, :4].astype(np.float32)}")
    print(f"  Sample PSUM values (K_TILE=64,  row 0, cols 0-3): {final_64[0, :4].astype(np.float32)}")


if __name__ == "__main__":
    print("Inspecting float32 PSUM accumulation via nki.simulate")
    print("Inputs: linspace(-1, 1), K=512, M=N=128")

    inspect(BF16,        "bfloat16 inputs:")
    inspect(np.float32,  "float32  inputs:")

    print("""
Interpretation:
  If PSUM diff = 0 for bfloat16: the float32 accumulator sees identical
  partial products regardless of tile size — invariance is established
  at the multiply step, not the cast-back step.

  If PSUM diff != 0 for float32: the float32 accumulator sees different
  partial sums depending on grouping — accumulation order matters.
""")
