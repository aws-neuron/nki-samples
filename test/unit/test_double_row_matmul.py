"""
Copyright (c) 2025, Amazon.com. All Rights Reserved
"""
import pytest
from nki_samples.reference.double_row_matmul import quantized_double_row_matmul
from neuronxcc.nki import benchmark, baremetal, simulate_kernel
import neuronxcc.nki.language as nl
import numpy as np

def get_target_string():
    """ returns instance type, e.g. trn1, inf2, trn2. """
    fpath = '/sys/devices/virtual/dmi/id/product_name'
    try:
        with open(fpath, 'r') as f:
            fc = f.readline()
    except IOError:
        warnings.warn('Unable to read MLA target.')
        return ""

    instance_type = fc.split('.')[0]
    return instance_type

bench_func = benchmark(warmup=5, iters=10)(quantized_double_row_matmul)

def reshape(matrix):
    """
    Interleaves every [128,512] tiles from every 2 tile rows.

    A [K,N] matrix is reshaped into [K//2, 2*N] where K must be divisible by 128 and 
    N must be divisible by 512.

    E.g. if Tij is the (i,j)-th tile and assuming a matrix with 4x4 [128,512] tiles,
    the reshaped matrix looks as follows

        # T11 T12 T13 T14          
        # T21 T22 T23 T24   reshape   T11 T21 T12 T22 T13 T23 T14 T24
        # T31 T32 T33 T34  -------->  T21 T41 T22 T42 T23 T43 T24 T44
        # T41 T42 T43 T44
    """
    K, N = matrix.shape

    TILE_K = 128
    TILE_N = 512
    
    assert K % TILE_K == 0
    assert N % TILE_N == 0

    result = np.zeros((K // 2, 2 * N))
    
    for k in range(0, K // TILE_K, 2):
      for n in range(N // TILE_N):
        # Get 2 tiles in the same tile column and consecutive tile rows.
        tile1 = matrix[k * TILE_K:(k + 1) * TILE_K, n * TILE_N:(n + 1) * TILE_N]
        tile2 = matrix[(k + 1) * TILE_K:(k + 2) * TILE_K, n * TILE_N:(n + 1) * TILE_N]

        result[(k // 2) * TILE_K:(k // 2 + 1) * TILE_K, n * TILE_N * 2:n * TILE_N * 2 + TILE_N] = tile1
        result[(k//2) * TILE_K:(k // 2 + 1) * TILE_K, n * TILE_N * 2 + TILE_N:(n + 1) * TILE_N * 2] = tile2
        
        # Place the 2 tiles in the same tile row side by side.
        result[(k // 2) * TILE_K:(k // 2 + 1) * TILE_K, n * TILE_N * 2:n * TILE_N * 2+TILE_N] = tile1
        result[(k // 2) * TILE_K:(k // 2 + 1) * TILE_K, n * TILE_N * 2 + TILE_N:n * TILE_N * 2 + TILE_N + TILE_N] = tile2
    
    return result

def column_wise_quantize(matrix):
    """
    Quantizes a matrix.

    Returns a column-wise scale broadcasted to (128, matrix.shape[1]) and the quantized matrix.
    """
    FP8_RANGE = 240
    column_wise_max = np.max(np.abs(matrix), axis=0, keepdims=True)
    column_wise_scale = column_wise_max / FP8_RANGE

    matrix_quantized = matrix / column_wise_scale
    column_wise_scale = np.broadcast_to(column_wise_scale, (128, matrix.shape[1]))

    return column_wise_scale, matrix_quantized

class TestDoubleRowMatmul:

    @pytest.mark.parametrize("M, K, N, dtype, TILES_IN_BLOCK_M, TILES_IN_BLOCK_N, TILES_IN_BLOCK_K, max_p99_latency", [
        [512, 16 * 1024, 1024, nl.bfloat16, 2, 2, 16, 320],
    ])
    def test_double_row_matmul_perf(self, M, K, N, dtype, TILES_IN_BLOCK_M, TILES_IN_BLOCK_N, TILES_IN_BLOCK_K, max_p99_latency):
        if (get_target_string() != "trn2"):
            return
        # Initializing random inputs
        lhs = np.random.rand(M, K)
        rhs = np.random.rand(K, N)

        # Quantizing rhs
        rhs_scale, rhs_quantized = column_wise_quantize(rhs)
        rhs_quantized_reshaped = reshape(rhs_quantized)

        # Casting to the correct data type (rhs is pre-quantized, thus casted to FP8)
        lhs = nl.static_cast(lhs, dtype)
        rhs_scale = nl.static_cast(rhs_scale, dtype)
        rhs_quantized_reshaped = nl.static_cast(rhs_quantized_reshaped, nl.float8_e4m3)
        
        # Latency checks
        bench_func(lhs, rhs_quantized_reshaped, rhs_scale, TILES_IN_BLOCK_M, TILES_IN_BLOCK_N, TILES_IN_BLOCK_K)
        latency_res = bench_func.benchmark_result.nc_latency
        p99_latency = latency_res.get_latency_percentile(99)

    @pytest.mark.simulation
    @pytest.mark.parametrize("M, K, N, dtype, TILES_IN_BLOCK_M, TILES_IN_BLOCK_N, TILES_IN_BLOCK_K", [
        [512, 16 * 1024, 1024, nl.bfloat16, 2, 2, 16],
        [512, 16 * 1024, 1024, nl.bfloat16, 4, 1, 32],
        [512, 16 * 1024, 1024, nl.bfloat16, 4, 2, 128],
    ])
    def test_double_row_matmul_numerical(self, simulation_only, M, K, N, dtype, TILES_IN_BLOCK_M, TILES_IN_BLOCK_N, TILES_IN_BLOCK_K):
        if (get_target_string() != "trn2"):
            return
        # Initializing random inputs
        lhs = np.random.rand(M, K)
        rhs = np.random.rand(K, N)
        
        # Correct CPU results
        result_golden = np.matmul(lhs, rhs)

        # Quantizing rhs
        rhs_scale, rhs_quantized = column_wise_quantize(rhs)
        rhs_quantized_reshaped = reshape(rhs_quantized)

        # Casting to the correct data type (rhs is pre-quantized, thus casted to FP8)
        lhs = nl.static_cast(lhs, dtype)
        rhs_scale = nl.static_cast(rhs_scale, dtype)
        rhs_quantized_reshaped = nl.static_cast(rhs_quantized_reshaped, nl.float8_e4m3)
        
        # Numerical accuracy checks
        numeric_func = baremetal(quantized_double_row_matmul)

        if simulation_only:
            result_nki = simulate_kernel(numeric_func, lhs, rhs_quantized_reshaped, rhs_scale, TILES_IN_BLOCK_M, TILES_IN_BLOCK_N, TILES_IN_BLOCK_K)
        else:
            result_nki = numeric_func(lhs, rhs_quantized_reshaped, rhs_scale, TILES_IN_BLOCK_M, TILES_IN_BLOCK_N, TILES_IN_BLOCK_K)

        # Casting result_nki from dtype BF16 back to FP32 to compare the NumPy and NKI results
        result_nki = result_nki.astype(np.float32)
        
        assert np.allclose(result_golden, result_nki, rtol=2e-2)
