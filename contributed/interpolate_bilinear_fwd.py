"""
Copyright (c) 2025, Amazon.com. All Rights Reserved

WARNING: These kernels:
   - Are tested only against internal nightly builds
   - May not be compatible with public NeuronSDK releases
   - Have not been extensively tested across all input configurations
   - Carry no compatibility guarantees
   - The behavior of these kernels may be modified without prior notice
"""

import math
import numpy as np
import torch

from neuronxcc import nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.typing as nt
from neuronxcc.nki import benchmark

# For debugging
# import os
# os.environ["NEURON_FRAMEWORK_DEBUG"] = "1"
# os.environ["NEURON_CC_FLAGS"]= " --disable-dge "

@nki.jit
def interpolate_bilinear_2x_fwd(src_arr: nt.tensor, chunk_size: int = 10) -> None:
    """
    This function performs a forward-pass linear interpolation on a 4D input array for `scaling_factor=2` and `align_corners=False`.

    Source array N & C dimensions are mapped to the SBUF partition dimension. H & W dimensions are mapped to the 
    SBUF free dimension. The kernel chunks the source array along the H dimension in chunks of `chunk_size` rows. 
    From a physical memory perspective, the kernel therefore operates on source array tiles of size (pmax, chunk_size * w_src).

    Args:
        src_arr (nt.tensor): Input HBM tensor of shape (N, C, H_src, W_src).
        chunk_size (int): Size of the chunk along the H free dimension.
    """
    P_TILE_SIZE = nl.tile_size.pmax # 128

    n_src, c_src, h_src, w_src = src_arr.shape
    n_dst, c_dst, h_dst, w_dst = n_src, c_src, h_src * 2, w_src * 2

    assert n_src == n_dst, "Input batch and output batch sizes must be identical"
    assert c_src == c_dst, "Input channel and output channel sizes must be identical"
    assert h_dst == 2 * h_src, "Output height must be twice the input height"
    assert w_dst == 2 * w_src, "Output width must be twice the input width"
    n, c = n_src, c_src

    weight_1d = 1 / 4 # specific to scaling_factor=2.0 & align_corners=False

    src_arr = src_arr.reshape((n * c, h_src, w_src))
    dst_arr = nl.ndarray(
        [n_dst * c_dst, h_dst, w_dst], 
        dtype=src_arr.dtype,
        buffer=nl.shared_hbm
    )
    
    wdw_size = chunk_size
    step_size = wdw_size - 1

    for h in nl.static_range(math.ceil((h_src - wdw_size) / step_size) + 1):
        h_start_hbm_src = max(0, h * step_size) 
        h_end_hbm_src = min(wdw_size + h * step_size, h_src)
        
        h_tile_size_src = h_end_hbm_src - h_start_hbm_src
        h_tile_size_dst = 2 * h_tile_size_src
        
        h_start_tile_dst = 1 if h_start_hbm_src else 0
        h_end_tile_dst = h_tile_size_dst
        
        h_start_hbm_dst = 2 * h_start_hbm_src + 1 if h_start_hbm_src else 0
        h_end_hbm_dst = 2 * h_end_hbm_src

        for p in nl.affine_range(math.ceil(n * c / P_TILE_SIZE)):
            out_tile = nl.ndarray([P_TILE_SIZE, h_tile_size_dst, w_dst], dtype=src_arr.dtype, buffer=nl.sbuf)

            ### Load input array from HBM
            i_p, i_h, i_w = nl.mgrid[0:P_TILE_SIZE, h_start_hbm_src:h_end_hbm_src, 0:w_src]
            i_p = p * P_TILE_SIZE + i_p

            in_tile = nl.load(src_arr[i_p, i_h, i_w], mask=(i_p < n * c))

            ### Core region
            weight_2d = weight_1d**2

            i_p, i_h_x, i_w_x, i_h_y, i_w_y = nl.mgrid[0:P_TILE_SIZE, 0:2, 0:2, 0:(h_tile_size_src-1), 0:(w_src-1)]
            i_h_dst = (2 * i_h_y + 1) + i_h_x
            i_w_dst = (2 * i_w_y + 1) + i_w_x
            
            i_h_src_056 = i_h_y + i_h_x
            i_w_src_056 = i_w_y + i_w_x
            i_h_src_006 = i_h_y + (-1 * i_h_x + 1)
            i_w_src_006 = i_w_y + (-1 * i_w_x + 1)
            i_h_src_018_h = i_h_y + (-1 * i_h_x + 1)
            i_w_src_018_h = i_w_y + i_w_x
            i_h_src_018_w = i_h_y + i_h_x
            i_w_src_018_w = i_w_y + (-1 * i_w_x + 1)
            
            out_tile[i_p, i_h_dst, i_w_dst] = (
                9 * weight_2d * in_tile[i_p, i_h_src_056, i_w_src_056] + \
                3 * weight_2d * in_tile[i_p, i_h_src_018_h, i_w_src_018_h] + \
                3 * weight_2d * in_tile[i_p, i_h_src_018_w, i_w_src_018_w] + \
                1 * weight_2d * in_tile[i_p, i_h_src_006, i_w_src_006]
            )
        
            ### Edges
            ## Upper & lower edges, i.e. h=0 or h=h_dst-1 (corners excluded)
            i_p, i_w_x, i_h, i_w_y = nl.mgrid[0:P_TILE_SIZE, 0:2, 0:2, 0:(w_src-1)]

            i_h_dst = (h_tile_size_dst - 1) * i_h
            i_w_dst = (2 * i_w_y + 1) + i_w_x
            
            i_h_src = (h_tile_size_src - 1) * i_h
            i_w_src_075 = i_w_y + i_w_x
            i_w_src_025 = i_w_y + (-1 * i_w_x + 1)
            
            out_tile[i_p, i_h_dst, i_w_dst] = (
                3 * weight_1d * in_tile[i_p, i_h_src, i_w_src_075] + \
                1 * weight_1d * in_tile[i_p, i_h_src, i_w_src_025]
            )
        
            ## Right & left edges, i.e. w=0 or w=w_dst-1 (corners excluded)
            i_p, i_h_x, i_w, i_h_y = nl.mgrid[0:P_TILE_SIZE, 0:2, 0:2, 0:(h_tile_size_src-1)]

            i_h_dst = (2 * i_h_y + 1) + i_h_x
            i_w_dst = (w_dst - 1) * i_w
            
            i_h_src_075 = i_h_y + i_h_x
            i_h_src_025 = i_h_y + (-1 * i_h_x + 1)
            i_w_src = (w_src - 1) * i_w
            
            out_tile[i_p, i_h_dst, i_w_dst] = (
                3 * weight_1d * in_tile[i_p, i_h_src_075, i_w_src] + \
                1 * weight_1d * in_tile[i_p, i_h_src_025, i_w_src]
            )
        
            ## Corners
            i_p, i_w, i_h = nl.mgrid[0:P_TILE_SIZE, 0:2, 0:2]

            i_h_dst = ((h_tile_size_dst - 1) * i_h)
            i_w_dst = ((w_dst - 1) * i_w)
            i_h_src = ((h_tile_size_src - 1) * i_h)
            i_w_src = ((w_src - 1) * i_w)
        
            out_tile[i_p, i_h_dst, i_w_dst] = in_tile[i_p, i_h_src, i_w_src]

            ### Write output array to HBM
            i_p, i_h_tile, i_w = nl.mgrid[0:P_TILE_SIZE, h_start_tile_dst:h_end_tile_dst, 0:w_dst]
            _, i_h_hbm, _ = nl.mgrid[0:P_TILE_SIZE, h_start_hbm_dst:h_end_hbm_dst, 0:w_dst]
            i_p_hbm = p * P_TILE_SIZE + i_p
            nl.store(dst_arr[i_p_hbm, i_h_hbm, i_w], value=out_tile[i_p, i_h_tile, i_w], mask=(i_p_hbm < n * c))

    dst_arr = dst_arr.reshape((n, c, h_dst, w_dst))
    return dst_arr


def check_correct():
  print("\nChecking Correctness")
  N, C, W, H = 2, 64, 128, 128 
  src_arr = np.random.random_sample([N, C, W, H]).astype(np.float32)
  
  baremetal_func = nki.baremetal()(interpolate_bilinear_2x_fwd)
  dst_arr = baremetal_func(src_arr)
  src_arr = torch.from_numpy(src_arr)
  nki_dst_arr = torch.from_numpy(dst_arr)
  torch_dst_arr = torch.nn.functional.interpolate(src_arr, scale_factor=2, mode="bilinear")
  print("Is close: ", torch.allclose(nki_dst_arr, torch_dst_arr, atol=1e-4, rtol=1e-2))

def benchmark_kernel():
  print("\nBenchmarking")
  N, C, W, H = 128, 64, 128, 128
  src_arr = np.random.random_sample([N, C, W, H]).astype(np.float32)

  benchmark_kernel = benchmark(warmup=10, iters=100, save_neff_name='file.neff', save_trace_name='profile.ntff')(interpolate_bilinear_2x_fwd)
  _ = benchmark_kernel(src_arr)
  metrics = benchmark_kernel.benchmark_result.nc_latency
  print("Latency (P50): " + str(metrics.get_latency_percentile(50)))
  print("Latency (P99): " + str(metrics.get_latency_percentile(99)))


def main():
  check_correct()
  benchmark_kernel()

if __name__ == "__main__":
  main()
