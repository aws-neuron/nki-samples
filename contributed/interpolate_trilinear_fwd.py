"""
Copyright (c) 2025, Amazon.com. All Rights Reserved

Authors: Mun Kim and Pierre Lienhart (equal contributions, alphabetical order) 
         from AWS GenAI Innovation Center

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
from torch import nn
import torch_xla.core.xla_model as xm

from neuronxcc import nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.typing as nt
from neuronxcc.nki import benchmark

# For debugging
# import os
# os.environ["NEURON_FRAMEWORK_DEBUG"] = "1"
# os.environ["NEURON_CC_FLAGS"]= " --disable-dge "


@nki.jit
def interpolate_trilinear_2x_fwd(src_arr: nt.tensor, chunk_size: int = 4) -> nt.tensor:
    """
    This function implements a NKI kernel for trilinear 2x upscaling (interpolation) on a 5D 
    input tensor array for the `align_corners=False` case and using advanced indexing only.

    Source array N & C dimensions are mapped to the SBUF partition dimension. D, H & W dimensions are mapped to the 
    SBUF free dimension. The kernel chunks the source array along the D dimension in chunks of `chunk_size` faces. 
    From a physical memory perspective, the kernel therefore operates on source array tiles of 
    size (pmax, chunk_size * h_src * w_src).

    Args:
        src_arr (nt.tensor): Input HBM tensor of shape (N, C, D_src, H_src, W_src).
        chunk_size (int): Size of the chunk along the D free dimension.
    """
    P_TILE_SIZE = nl.tile_size.pmax # 128
    n_src, c_src, d_src, h_src, w_src = src_arr.shape
    n_dst, c_dst, d_dst, h_dst, w_dst = n_src, c_src, d_src * 2, h_src * 2, w_src * 2

    assert n_src == n_dst, "Input batch and output batch sizes must be identical"
    assert c_src == c_dst, "Input channel and output channel sizes must be identical"
    assert d_dst == 2 * d_src, "Output depth must be twice the input depth"
    assert h_dst == 2 * h_src, "Output height must be twice the input height"
    assert w_dst == 2 * w_src, "Output width must be twice the input width"

    n, c = n_src, c_src
    
    weight_1d = 1 / 4 # specific to scaling_factor=2.0 & align_corners=False
    src_arr = src_arr.reshape((n * c, d_src, h_src, w_src))
    dst_arr = nl.ndarray(
        [n_dst * c_dst, d_dst, h_dst, w_dst], 
        dtype=src_arr.dtype,
        buffer=nl.shared_hbm
    )

    wdw_size = chunk_size
    step_size = wdw_size - 1
    for d in nl.static_range(math.ceil((d_src - wdw_size)/step_size) + 1):
        d_start_hbm_src = max(0, d * step_size) 
        d_end_hbm_src = min(wdw_size + d * step_size, d_src)
        
        d_tile_size_src = d_end_hbm_src - d_start_hbm_src
        d_tile_size_dst = 2 * d_tile_size_src
        
        d_start_tile_dst = 1 if d_start_hbm_src else 0
        d_end_tile_dst = d_tile_size_dst
        
        d_start_hbm_dst = 2 * d_start_hbm_src + 1 if d_start_hbm_src else 0
        d_end_hbm_dst = 2 * d_end_hbm_src

        for p in nl.affine_range(math.ceil(n * c / P_TILE_SIZE)):
            out_tile = nl.ndarray([P_TILE_SIZE, d_tile_size_dst, h_dst, w_dst], dtype=src_arr.dtype, buffer=nl.sbuf)

            ### Load input array from HBM
            i_p, i_d, i_h, i_w = nl.mgrid[0:P_TILE_SIZE, d_start_hbm_src:d_end_hbm_src, 0:h_src, 0:w_src]
            i_p = p * P_TILE_SIZE + i_p
            in_tile = nl.load(src_arr[i_p, i_d, i_h, i_w], mask=(i_p < n * c))

            ### Core region
            weight_3d = weight_1d ** 3
            i_p, i_d_x, i_h_x, i_w_x, i_d_y, i_h_y, i_w_y = nl.mgrid[
                0:P_TILE_SIZE, 0:2, 0:2, 0:2, 0:(d_tile_size_src-1), 0:(h_src-1), 0:(w_src-1)
            ]
            i_d_dst = (2 * i_d_y + 1) + i_d_x
            i_h_dst = (2 * i_h_y + 1) + i_h_x
            i_w_dst = (2 * i_w_y + 1) + i_w_x
            
            # (9*3)*weight_3d = 0.42
            i_d_src_042 = i_d_y + i_d_x
            i_h_src_042 = i_h_y + i_h_x
            i_w_src_042 = i_w_y + i_w_x
            
            # (3*3)*weight_3d (depth) = 0.14
            i_d_src_014_d = i_d_y + (-1 * i_d_x + 1)
            i_h_src_014_d = i_h_y + i_h_x
            i_w_src_014_d = i_w_y + i_w_x
            
            # (3*3)*weight_3d (height) = 0.14
            i_d_src_014_h = i_d_y + i_d_x
            i_h_src_014_h = i_h_y + (-1 * i_h_x + 1)
            i_w_src_014_h = i_w_y + i_w_x
            
            # (1*3)*weight_3d (width) = 0.05
            i_d_src_005_w = i_d_y + (-1 * i_d_x + 1)
            i_h_src_005_w = i_h_y + (-1 * i_h_x + 1)
            i_w_src_005_w = i_w_y + i_w_x
            
            # (9*1)*weight_3d (width) = 0.14
            i_d_src_014_w = i_d_y + i_d_x
            i_h_src_014_w = i_h_y + i_h_x
            i_w_src_014_w = i_w_y + (-1 * i_w_x + 1)
            
            # (3*1)*weight_3d (depth) = 0.05
            i_d_src_005_d = i_d_y + (-1 * i_d_x + 1)
            i_h_src_005_d = i_h_y + i_h_x
            i_w_src_005_d = i_w_y + (-1 * i_w_x + 1)
            
            # (3*1)*weight_3d (height) = 0.05
            i_d_src_005_h = i_d_y + i_d_x
            i_h_src_005_h = i_h_y + (-1 * i_h_x + 1)
            i_w_src_005_h = i_w_y + (-1 * i_w_x + 1)
            
            # (1*1)*weight_3d = 0.01
            i_d_src_001 = i_d_y + (-1 * i_d_x + 1)
            i_h_src_001 = i_h_y + (-1 * i_h_x + 1)
            i_w_src_001 = i_w_y + (-1 * i_w_x + 1)
            
            out_tile[i_p, i_d_dst, i_h_dst, i_w_dst] = (
                (9 * 3) * weight_3d * in_tile[i_p, i_d_src_042, i_h_src_042, i_w_src_042] + \
                (3 * 3) * weight_3d * in_tile[i_p, i_d_src_014_d, i_h_src_014_d, i_w_src_014_d] + \
                (3 * 3) * weight_3d * in_tile[i_p, i_d_src_014_h, i_h_src_014_h, i_w_src_014_h] + \
                (1 * 3) * weight_3d * in_tile[i_p, i_d_src_005_w, i_h_src_005_w, i_w_src_005_w] + \
                (9 * 1) * weight_3d * in_tile[i_p, i_d_src_014_w, i_h_src_014_w, i_w_src_014_w] + \
                (3 * 1) * weight_3d * in_tile[i_p, i_d_src_005_d, i_h_src_005_d, i_w_src_005_d] + \
                (3 * 1) * weight_3d * in_tile[i_p, i_d_src_005_h, i_h_src_005_h, i_w_src_005_h] + \
                (1 * 1) * weight_3d * in_tile[i_p, i_d_src_001, i_h_src_001, i_w_src_001]
            )
            
            ### Faces
            weight_2d = weight_1d ** 2
            
            ## d=0 and d=d_dst - 1 faces (borders excluded)
            i_p, i_d, i_h_x, i_w_x, i_h_y, i_w_y = nl.mgrid[0:P_TILE_SIZE, 0:2, 0:2, 0:2, 0:(h_src-1), 0:(w_src-1)]
            i_d_dst = ((d_tile_size_dst - 1) * i_d)
            i_h_dst = (2 * i_h_y + 1) + i_h_x
            i_w_dst = (2 * i_w_y + 1) + i_w_x
            
            i_d_src = ((d_tile_size_src - 1) * i_d)
            i_h_src_056 = i_h_y + i_h_x
            i_w_src_056 = i_w_y + i_w_x
            i_h_src_018_h = i_h_y + (-1 * i_h_x + 1)
            i_w_src_018_h = i_w_y + i_w_x
            i_h_src_018_w = i_h_y + i_h_x
            i_w_src_018_w = i_w_y + (-1 * i_w_x + 1)
            i_h_src_006 = i_h_y + (-1 * i_h_x + 1)
            i_w_src_006 = i_w_y + (-1 * i_w_x + 1)
            
            out_tile[i_p, i_d_dst, i_h_dst, i_w_dst] = (
                9 * weight_2d * in_tile[i_p, i_d_src, i_h_src_056, i_w_src_056] + \
                3 * weight_2d * in_tile[i_p, i_d_src, i_h_src_018_h, i_w_src_018_h] + \
                3 * weight_2d * in_tile[i_p, i_d_src, i_h_src_018_w, i_w_src_018_w] + \
                1 * weight_2d * in_tile[i_p, i_d_src, i_h_src_006, i_w_src_006]
            )
            
            ## h=0 and h=h_dst - 1 faces (borders excluded)
            i_p, i_d_x, i_h, i_w_x, i_d_y, i_w_y = nl.mgrid[0:P_TILE_SIZE, 0:2, 0:2, 0:2, 0:(d_tile_size_src-1), 0:(w_src-1)]
            i_d_dst = (2 * i_d_y + 1) + i_d_x
            i_h_dst = ((h_dst - 1) * i_h)
            i_w_dst = (2 * i_w_y + 1) + i_w_x
            
            i_h_src = ((h_src - 1) * i_h)
            i_d_src_056 = i_d_y + i_d_x
            i_w_src_056 = i_w_y + i_w_x
            i_d_src_018_d = i_d_y + (-1 * i_d_x + 1)
            i_w_src_018_d = i_w_y + i_w_x
            i_d_src_018_w = i_d_y + i_d_x
            i_w_src_018_w = i_w_y + (-1 * i_w_x + 1)
            i_d_src_006 = i_d_y + (-1 * i_d_x + 1)
            i_w_src_006 = i_w_y + (-1 * i_w_x + 1)
            
            out_tile[i_p, i_d_dst, i_h_dst, i_w_dst] = (
                9 * weight_2d * in_tile[i_p, i_d_src_056, i_h_src, i_w_src_056] + \
                3 * weight_2d * in_tile[i_p, i_d_src_018_d, i_h_src, i_w_src_018_d] + \
                3 * weight_2d * in_tile[i_p, i_d_src_018_w, i_h_src, i_w_src_018_w] + \
                1 * weight_2d * in_tile[i_p, i_d_src_006, i_h_src, i_w_src_006]
            )
            
            ## w=0 and w=w_dst - 1 faces (borders excluded)
            i_p, i_d_x, i_h_x, i_w, i_d_y, i_h_y = nl.mgrid[0:P_TILE_SIZE, 0:2, 0:2, 0:2, 0:(d_tile_size_src-1), 0:(h_src-1)]
            i_d_dst = (2 * i_d_y + 1) + i_d_x
            i_h_dst = (2 * i_h_y + 1) + i_h_x
            i_w_dst = ((w_dst - 1) * i_w)
            
            i_w_src = ((w_src - 1) * i_w)
            i_d_src_056 = i_d_y + i_d_x
            i_h_src_056 = i_h_y + i_h_x
            i_d_src_018_d = i_d_y + (-1 * i_d_x + 1)
            i_h_src_018_d = i_h_y + i_h_x
            i_d_src_018_h = i_d_y + i_d_x
            i_h_src_018_h = i_h_y + (-1 * i_h_x + 1)
            i_d_src_006 = i_d_y + (-1 * i_d_x + 1)
            i_h_src_006 = i_h_y + (-1 * i_h_x + 1)
            
            out_tile[i_p, i_d_dst, i_h_dst, i_w_dst] = (
                9 * weight_2d * in_tile[i_p, i_d_src_056, i_h_src_056, i_w_src] + \
                3 * weight_2d * in_tile[i_p, i_d_src_018_d, i_h_src_018_d, i_w_src] + \
                3 * weight_2d * in_tile[i_p, i_d_src_018_h, i_h_src_018_h, i_w_src] + \
                1 * weight_2d * in_tile[i_p, i_d_src_006, i_h_src_006, i_w_src]
            )
            
            ### Edges
            ## (d=0 and d=d_dst - 1) x (h=0 and h=h_dst - 1) edges (corners excluded)
            i_p, i_d, i_h, i_w_x, i_w_y = nl.mgrid[0:P_TILE_SIZE, 0:2, 0:2, 0:2, 0:(w_src-1)]
            i_d_dst = ((d_tile_size_dst - 1) * i_d)
            i_h_dst = ((h_dst - 1) * i_h)
            i_w_dst = (2 * i_w_y + 1) + i_w_x
            
            i_d_src = ((d_tile_size_src - 1) * i_d)
            i_h_src = ((h_src - 1) * i_h)
            i_w_src_075 = i_w_y + i_w_x
            i_w_src_025 = i_w_y + (-1 * i_w_x + 1)
            
            out_tile[i_p, i_d_dst, i_h_dst, i_w_dst] = (
                3 * weight_1d * in_tile[i_p, i_d_src, i_h_src, i_w_src_075] + \
                1 * weight_1d * in_tile[i_p, i_d_src, i_h_src, i_w_src_025]
            )
            
            ## (d=0 and d=d_dst - 1) x (w=0 and w=w_dst - 1) edges (corners excluded)
            i_p, i_d, i_h_x, i_w, i_h_y = nl.mgrid[0:P_TILE_SIZE, 0:2, 0:2, 0:2, 0:(h_src-1)]
            i_d_dst = ((d_tile_size_dst - 1) * i_d)
            i_h_dst = (2 * i_h_y + 1) + i_h_x
            i_w_dst = ((w_dst - 1) * i_w)
            
            i_d_src = ((d_tile_size_src - 1) * i_d)
            i_w_src = ((w_src - 1) * i_w)
            i_h_src_075 = i_h_y + i_h_x
            i_h_src_025 = i_h_y + (-1 * i_h_x + 1)
            
            out_tile[i_p, i_d_dst, i_h_dst, i_w_dst] = (
                3 * weight_1d * in_tile[i_p, i_d_src, i_h_src_075, i_w_src] + \
                1 * weight_1d * in_tile[i_p, i_d_src, i_h_src_025, i_w_src]
            )
            
            ## (h=0 and h=h_dst - 1) x (w=0 and w=w_dst - 1) edges (corners excluded)
            i_p, i_d_x, i_h, i_w, i_d_y = nl.mgrid[0:P_TILE_SIZE, 0:2, 0:2, 0:2, 0:(d_tile_size_src-1)]
            i_d_dst = (2 * i_d_y + 1) + i_d_x
            i_h_dst = ((h_dst - 1) * i_h)
            i_w_dst = ((w_dst - 1) * i_w)
            
            i_h_src = ((h_src - 1) * i_h)
            i_w_src = ((w_src - 1) * i_w)
            i_d_src_075 = i_d_y + i_d_x
            i_d_src_025 = i_d_y + (-1 * i_d_x + 1)
            
            out_tile[i_p, i_d_dst, i_h_dst, i_w_dst] = (
                3 * weight_1d * in_tile[i_p, i_d_src_075, i_h_src, i_w_src] + \
                1 * weight_1d * in_tile[i_p, i_d_src_025, i_h_src, i_w_src]
            )
            
            ### Corners
            i_p, i_d, i_h, i_w = nl.mgrid[0:P_TILE_SIZE, 0:2, 0:2, 0:2]
            i_d_dst = ((d_tile_size_dst - 1) * i_d)
            i_h_dst = ((h_dst - 1) * i_h)
            i_w_dst = ((w_dst - 1) * i_w)
            
            i_d_src = ((d_tile_size_src - 1) * i_d)
            i_h_src = ((h_src - 1) * i_h)
            i_w_src = ((w_src - 1) * i_w)
            
            out_tile[i_p, i_d_dst, i_h_dst, i_w_dst] = in_tile[i_p, i_d_src, i_h_src, i_w_src]

            ### Write output array to HBM
            i_p, i_d, i_h, i_w = nl.mgrid[0:P_TILE_SIZE, d_start_tile_dst:d_end_tile_dst, 0:h_dst, 0:w_dst]
            i_p_hbm = p * P_TILE_SIZE + i_p
            _, i_d_hbm, _, _ = nl.mgrid[0:P_TILE_SIZE, d_start_hbm_dst:d_end_hbm_dst, 0:h_dst, 0:w_dst]
            nl.store(dst_arr[i_p_hbm, i_d_hbm, i_h, i_w], value=out_tile[i_p, i_d, i_h, i_w], mask=(i_p_hbm < n * c))

    dst_arr = dst_arr.reshape((n, c, d_dst, h_dst, w_dst))
    return dst_arr


def check_correct():
    print("\nChecking Correctness")
    N, C, D, W, H = 2, 64, 23, 23, 23
    src_arr = np.random.random_sample([N, C, D, W, H]).astype(np.float32)

    baremetal_func = nki.baremetal()(interpolate_trilinear_2x_fwd)
    dst_arr = baremetal_func(src_arr)
    src_arr = torch.from_numpy(src_arr)
    nki_dst_arr = torch.from_numpy(dst_arr)
    torch_dst_arr = torch.nn.functional.interpolate(src_arr, scale_factor=2, mode="trilinear")
    print("Is close: ", torch.allclose(nki_dst_arr, torch_dst_arr, atol=1e-4, rtol=1e-2))


def benchmark_kernel():
    print("\nBenchmarking")
    N, C, D, W, H = 2, 64, 23, 23, 23
    src_arr = np.random.random_sample([N, C, D, W, H]).astype(np.float32)

    benchmark_kernel = benchmark(warmup=10, iters=100, save_neff_name='file.neff', save_trace_name='profile.ntff')(interpolate_trilinear_2x_fwd)
    _ = benchmark_kernel(src_arr)
    metrics = benchmark_kernel.benchmark_result.nc_latency
    print("Latency (P50): " + str(metrics.get_latency_percentile(50)))
    print("Latency (P99): " + str(metrics.get_latency_percentile(99)))


def main():
    check_correct()
    #benchmark_kernel()

if __name__ == "__main__":
    main()

