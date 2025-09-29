"""
Copyright (c) 2025, Amazon.com. All Rights Reserved

Unit tests for adaptive average pooling 2D kernel
"""
import pytest
import numpy as np

from nki_samples.reference.vision import adaptive_avg_pool2d_kernel
from neuronxcc.nki import benchmark, baremetal, simulate_kernel
import neuronxcc.nki.language as nl


bench_func = benchmark(warmup=5, iters=10)(adaptive_avg_pool2d_kernel)

def cpu_golden_result_numpy(input_tensor, output_size):
    """
    Pure NumPy reference implementation for adaptive average pooling
    """
    N, C, H, W = input_tensor.shape
    
    if isinstance(output_size, int):
        OH = OW = output_size
    else:
        OH, OW = output_size
    
    output = np.zeros((N, C, OH, OW), dtype=input_tensor.dtype)
    
    for n in range(N):
        for c in range(C):
            for oh in range(OH):
                for ow in range(OW):
                    # Calculate the input region for this output position
                    h_start = (oh * H) // OH
                    h_end = ((oh + 1) * H + OH - 1) // OH
                    w_start = (ow * W) // OW
                    w_end = ((ow + 1) * W + OW - 1) // OW
                    
                    # Compute average over the region
                    region = input_tensor[n, c, h_start:h_end, w_start:w_end]
                    output[n, c, oh, ow] = np.mean(region)
    
    return output


class TestAdaptiveAvgPool2D:
    
    @pytest.mark.parametrize("N, C, H, W, output_size, dtype, latency", [
        # Test cases with various input/output sizes
        [1, 64, 56, 56, 7, np.float32, 65 * 1.05],
        [2, 128, 28, 28, 7, np.float32, 109 * 1.05],
        [1, 256, 14, 14, 7, nl.float16, 88 * 1.05],
        [4, 64, 56, 56, (7, 7), np.float32, 152 * 1.05],
        [2, 128, 28, 28, (14, 14), np.float32, 327 * 1.05],
        [1, 512, 7, 7, 1, np.float32, 14 * 1.05],  # Global average pooling
        [8, 64, 32, 32, (4, 4), nl.float16, 103 * 1.05],
        [2, 2048, 32, 32, (3, 3), np.float32, 564 * 1.05],
        [2, 64, 256, 256, 1, np.float32, 255 * 1.05],
    ])
    def test_adaptive_avg_pool2d_perf(self, N, C, H, W, output_size, dtype, latency):
        """Performance test for adaptive average pooling 2D"""
        # Generate random input
        input_tensor = np.random.randn(N, C, H, W).astype(np.float32)
        
        # Cast to target dtype
        input_dev = nl.static_cast(input_tensor, dtype)
        
        # Run benchmark
        bench_func(input_dev, output_size)
        latency_res = bench_func.benchmark_result.nc_latency
        p99 = latency_res.get_latency_percentile(99)
        
        # Check latency requirement
        assert p99 <= latency, f"P99 latency {p99} exceeds threshold {latency}"
    
    @pytest.mark.simulation
    @pytest.mark.parametrize("N, C, H, W, output_size, dtype", [
        # Numerical accuracy test cases
        [1, 64, 56, 56, 7, np.float32],
        [2, 128, 28, 28, 7, np.float32],
        [1, 256, 14, 14, 7, nl.float16],
        [1, 256, 14, 14, 7, nl.bfloat16],
        [4, 64, 56, 56, (7, 7), np.float32],
        [2, 128, 28, 28, (14, 14), np.float32],
        [1, 512, 7, 7, 1, np.float32],  # Global average pooling
        [1, 64, 32, 32, (8, 8), nl.float16],
        [3, 96, 24, 24, (6, 6), nl.bfloat16],
        [2, 192, 16, 16, (4, 2), np.float32],  # Non-square output
        [2, 2048, 32, 32, (3, 3), np.float32],
        [1, 1, 1, 1, 1, np.float32],  # Single element
        [1, 128, 7, 7, 7, np.float32],  # Same input and output size
        [2, 64, 256, 256, 1, np.float32],  # Global pooling on large input
        [1, 256, 13, 13, 6, np.float32],  # Non-divisible dimensions
        [4, 512, 7, 7, (1, 1), np.float32],  # Tuple output for global pooling
    ])
    def test_adaptive_avg_pool2d_numerical(self, simulation_only, N, C, H, W, output_size, dtype):
        """Numerical accuracy test for adaptive average pooling 2D"""
        # Generate random input
        input_tensor = np.random.randn(N, C, H, W).astype(np.float32)
        
        # Cast to target dtype
        input_dev = nl.static_cast(input_tensor, dtype)
        
        # Run kernel
        numeric_func = baremetal(adaptive_avg_pool2d_kernel)
        if simulation_only:
            output_tensor = simulate_kernel(numeric_func, input_dev, output_size)
        else:
            output_tensor = numeric_func(input_dev, output_size)
        
        # Cast output back to float32 for comparison
        output_tensor = nl.static_cast(output_tensor, np.float32)
        
        # Compute golden result using NumPy implementation
        golden_result = cpu_golden_result_numpy(input_tensor, output_size)
        
        # Compare results with appropriate tolerance
        if dtype in [nl.float16, nl.bfloat16]:
            atol = 1e-2
            rtol = 1e-2
        else:
            atol = 1e-5
            rtol = 1e-5
        
        assert np.allclose(output_tensor, golden_result, atol=atol, rtol=rtol), \
            f"Output mismatch. Max diff: {np.max(np.abs(output_tensor - golden_result))}"
