"""
Copyright (c) 2023, Amazon.com. All Rights Reserved
"""
import pytest

from nki_samples.reference.vision import resize_nearest_fixed_dma_kernel
from neuronxcc.nki import benchmark, baremetal, simulate_kernel
import neuronxcc.nki.language as nl
import numpy as np

bench_func = benchmark(warmup=5, iters=10)(resize_nearest_fixed_dma_kernel)


def cpu_golden_result(data_tensor, output_shape):
    in_b, in_h, in_w, in_c = data_tensor.shape
    out_b, out_h, out_w, out_c = output_shape

    # Generate nearest map
    h_scale, w_scale = 1.0 * in_h / out_h, 1.0 * in_w / out_w
    h_map = np.floor(np.fromfunction(lambda i, _: i * h_scale, (out_h, out_w), dtype=np.float32))
    w_map = np.floor(np.fromfunction(lambda _, j: j * w_scale, (out_h, out_w), dtype=np.float32))
    map = (h_map * in_w + w_map).astype(np.int32).flatten()

    in_seqlen, out_seqlen = in_h * in_w, out_h * out_w

    data_tensor = data_tensor.reshape((in_b, in_seqlen, in_c))
    out_tensor = np.zeros((out_b, out_seqlen, out_c))

    for b_map in range(in_b):
        for i in range(len(map)):
            for c_map in range(out_c):
                out_tensor[b_map, i, c_map] = data_tensor[b_map, map[i], c_map]

    return out_tensor.reshape(( out_b, out_h, out_w, out_c ))

class TestResizeNearest:

    @pytest.mark.parametrize("in_b, in_h, in_w, in_c, out_b, out_h, out_w, out_c, dtype, latency", [
 	    [10, 30, 20, 1280, 10, 59, 38, 1280, np.float32, 1740],
        [1, 30, 20, 1280, 1, 59, 38, 1280, nl.float16, 659],
        [1, 30, 20, 1280, 1, 59, 38, 1280, nl.bfloat16, 659],
 	])
    def test_resize_nearest_for_perf(self, in_b, in_h, in_w, in_c, out_b, out_h, out_w, out_c, dtype, latency):
        input_tensor = np.random.random_sample((in_b, in_h, in_w, in_c)).astype(np.float32)

        input_dev = nl.static_cast(input_tensor, dtype)

        bench_func_ = bench_func[in_b]
        bench_func_(input_dev, (out_b, out_h, out_w, out_c))
        latency_res = bench_func_.benchmark_result.nc_latency
        p99 = latency_res.get_latency_percentile(50)
        assert p99 <= latency

    @pytest.mark.simulation
    @pytest.mark.parametrize("in_b, in_h, in_w, in_c, out_b, out_h, out_w, out_c, dtype", [
 	    [10, 30, 20, 1280, 10, 59, 38, 1280, np.float32],
        [1, 30, 20, 1280, 1, 59, 38, 1280, nl.float16],
        [1, 30, 20, 1280, 1, 59, 38, 1280, nl.bfloat16],
 	])
    def test_resize_nearest_for_numberic(self, simulation_only, in_b, in_h, in_w, in_c, out_b, out_h, out_w, out_c, dtype):
        input_tensor = np.random.random_sample((in_b, in_h, in_w, in_c)).astype(np.float32)

        input_dev = nl.static_cast(input_tensor, dtype)

        numeric_func = baremetal(resize_nearest_fixed_dma_kernel)
        if simulation_only:
            output_tensor = simulate_kernel(numeric_func[in_b], input_dev, (out_b, out_h, out_w, out_c))
        else:
            output_tensor = numeric_func[in_b](input_dev, (out_b, out_h, out_w, out_c))
        output_tensor = nl.static_cast(output_tensor, np.float32)
        golden_result = cpu_golden_result(input_tensor, output_tensor.shape)
        assert np.allclose(output_tensor, golden_result, atol=1e-2)
