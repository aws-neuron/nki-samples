"""
Copyright (c) 2023, Amazon.com. All Rights Reserved
"""
import os
import pytest
from neuronxcc.nki.kernels.attention import fused_self_attn_for_SD_small_head_size
from neuronxcc.nki import benchmark, baremetal
import neuronxcc.nki.language as nl
import numpy as np
from scipy.special import softmax

test_trace_file_path='local_trace.ntff'
numeric_func = baremetal(fused_self_attn_for_SD_small_head_size)
bench_func = benchmark(warmup=5, iters=10, save_trace_name=test_trace_file_path)(fused_self_attn_for_SD_small_head_size)

def cpu_golden_attn(q, k, v):
    softmax_scale = 0.125
    q_scaled = q * softmax_scale
    raw_score = np.matmul(q_scaled.transpose(0, 2, 1), k.transpose(0, 2, 1))
    
    norm_score = softmax(raw_score, axis=-1)

    # Transpose the result so it has the same layout as ours
    return np.matmul(norm_score, v)

class TestAttention:

    @pytest.mark.parametrize("bs,seqlen,d,dtype,latency", [
        [1, 4096, 128, np.float32, 600],
        [1, 4096, 128, nl.bfloat16, 480],
        [1, 4096, 64, nl.float16, 520]
    ])
    def test_attention_for_SD_perf(self, bs, seqlen, d, dtype, latency):
        q = np.random.random_sample((bs, d, seqlen)).astype(np.float32)
        k = np.random.random_sample((bs, seqlen, d)).astype(np.float32)
        v = np.random.random_sample((bs, seqlen, d)).astype(np.float32)
        out = nl.static_cast(np.ndarray(shape=(bs, seqlen, d)), dtype)
        
        q_dev = nl.static_cast(q, dtype)
        k_dev = nl.static_cast(k, dtype)
        v_dev = nl.static_cast(v, dtype)

        bench_func[bs](q_dev, k_dev, v_dev, out)
        latency_res = bench_func.benchmark_result.nc_latency
        p99 = latency_res.get_latency_percentile(99)
        assert p99 <= latency
        assert os.path.getsize(test_trace_file_path) > 0

    @pytest.mark.parametrize("bs,seqlen,d,dtype", [
        [1, 4096, 128, np.float32],
        [1, 4096, 128, nl.bfloat16]
    ])
    def test_attention_for_SD_numberic(self, bs, seqlen, d, dtype):
        q = np.random.random_sample((bs, d, seqlen)).astype(np.float32)
        k = np.random.random_sample((bs, seqlen, d)).astype(np.float32)
        v = np.random.random_sample((bs, seqlen, d)).astype(np.float32)
        out = nl.static_cast(np.ndarray(shape=(bs, seqlen, d)), dtype)
        
        q_dev = nl.static_cast(q, dtype)
        k_dev = nl.static_cast(k, dtype)
        v_dev = nl.static_cast(v, dtype)

        numeric_func[bs](q_dev, k_dev, v_dev, out)
        out = nl.static_cast(out, np.float32)
        golden_result = cpu_golden_attn(q, k, v)
        assert np.allclose(out, golden_result, atol=1e-2)
