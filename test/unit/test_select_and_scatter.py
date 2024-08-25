import pytest

from neuronxcc.nki.kernels.vision import select_and_scatter_kernel
from neuronxcc.nki import benchmark, baremetal
import neuronxcc.nki.language as nl
import numpy as np

numeric_func = baremetal(select_and_scatter_kernel)
bench_func = benchmark(warmup=5, iters=10)(select_and_scatter_kernel)

np.random.seed(0)

def cpu_golden_result(operand_tensor, source_tensor, window_dimensions=(3, 3), window_strides=(2, 2),padding=(1, 1)):
    N, C, H, W = operand_tensor.shape  # batch, channel, height, width
    sw_h, sw_w = window_dimensions  # set window dimensions to 3
    stride_h, stride_w = window_strides # set window strides to 2
    src_n, src_c, src_h, src_w = source_tensor.shape
    padded_h = H + sum(padding)
    padded_w = W + sum(padding)
    assert N == src_n and C == src_c
    assert (padded_h - sw_h) // stride_h + 1 == src_h
    assert (padded_w - sw_w) // stride_w + 1 == src_w
    assert H == W and src_h == src_w

    assert operand_tensor.dtype == source_tensor.dtype
    dtype = operand_tensor.dtype

    output_shape = (N, C, H, W)

    padded_operand_tensor = np.pad(operand_tensor, ((0, 0), (0, 0), padding, padding), 'constant')
    output_tensor = np.zeros(output_shape, dtype)

    for n in range(N):
        for c in range(C):
            for h in range(src_h):
                for w in range(src_w):
                    local_max_idx = np.argmax(padded_operand_tensor[n, c, h*stride_h:h*stride_h+sw_h, w*stride_w:w*stride_w+sw_w])
                    local_h, local_w = local_max_idx // sw_w, local_max_idx % sw_w
                    out_h = h * stride_h + local_h - padding[0]
                    out_w = w * stride_w + local_w - padding[1]
                    output_tensor[n, c, out_h, out_w] += source_tensor[n, c, h, w]

    return output_tensor

class TestSelectAndScatter:
    @pytest.mark.parametrize("n, c, operand_h, operand_w, source_h, source_w, dtype, latency", [
 	    [8, 64, 112, 112, 56, 56, np.float32, 4500],
 	])
    def test_select_and_scatter_for_perf(self, n, c, operand_h, operand_w, source_h, source_w, dtype, latency):
        operand_tensor = np.random.random_sample((n, c, operand_h, operand_w)).astype(np.float32)
        source_tensor = np.random.random_sample((n, c, source_h, source_w)).astype(np.float32)
        output_tensor = nl.static_cast(np.ndarray(shape=(n, c, operand_h, operand_w)), dtype)
        
        operand_dev = nl.static_cast(operand_tensor, dtype)
        source_dev = nl.static_cast(source_tensor, dtype)

        bench_func(operand_dev, source_dev, output_tensor)
        latency_res = bench_func.benchmark_result.nc_latency
        p99 = latency_res.get_latency_percentile(99)
        assert p99 <= latency

    @pytest.mark.parametrize("n, c, operand_h, operand_w, source_h, source_w, dtype", [
 	    [8, 64, 112, 112, 56, 56, np.float32],
 	    pytest.param(8, 64, 112, 112, 56, 56, nl.bfloat16, marks=pytest.mark.xfail),
 	])
    def test_select_and_scatter_for_numeric(self, n, c, operand_h, operand_w, source_h, source_w, dtype):
        operand_tensor = np.random.random_sample((n, c, operand_h, operand_w)).astype(np.float32)
        source_tensor = np.random.random_sample((n, c, source_h, source_w)).astype(np.float32)
        output_tensor = nl.static_cast(np.ndarray(shape=(n, c, operand_h, operand_w)), dtype)
        
        operand_dev = nl.static_cast(operand_tensor, dtype)
        source_dev = nl.static_cast(source_tensor, dtype)

        numeric_func(operand_dev, source_dev, output_tensor)
        golden_result = cpu_golden_result(operand_tensor, source_tensor)
        output_tensor = nl.static_cast(output_tensor, np.float32)
        assert np.allclose(output_tensor, golden_result)