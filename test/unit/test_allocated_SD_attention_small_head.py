"""
Copyright (c) 2023, Amazon.com. All Rights Reserved
"""
import os
import pytest
from neuronxcc.nki.kernels.allocated_attention import allocated_fused_self_attn_for_SD_small_head_size
from neuronxcc.nki import benchmark, baremetal
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import numpy as np
from scipy.special import softmax

test_trace_file_path='local_trace.ntff'
numeric_func = baremetal(allocated_fused_self_attn_for_SD_small_head_size)
bench_func = benchmark(warmup=5, iters=20, save_trace_name=test_trace_file_path)(allocated_fused_self_attn_for_SD_small_head_size)

def cpu_golden_attn(q, k, v):
    softmax_scale = 0.125
    q_scaled = q * softmax_scale
    raw_score = np.matmul(q_scaled.transpose(0, 2, 1), k)
    
    norm_score = softmax(raw_score, axis=-1)

    # Transpose the result so it has the same layout as ours
    return np.matmul(norm_score, v).transpose(0, 2, 1)

class TestAttention:

    @pytest.mark.parametrize("bs,seqlen,d,dtype,latency", [
        [1, 4096, 128, np.float32, 410],
        [1, 4096, 128, nl.bfloat16, 350],
        [1, 5120, 128, nl.bfloat16, 586]
    ])
    def test_allocated_attention_for_SD_perf(self, bs, seqlen, d, dtype, latency):
        q = np.random.random_sample((bs, d, seqlen)).astype(np.float32)
        k = np.random.random_sample((bs, d, seqlen)).astype(np.float32)
        v = np.random.random_sample((bs, seqlen, d)).astype(np.float32)

        q_dev = nl.static_cast(q, dtype)
        k_dev = nl.static_cast(k, dtype)
        v_dev = nl.static_cast(v, dtype)

        bench_func_ = bench_func[bs]
        bench_func_(q_dev, k_dev, v_dev)
        latency_res = bench_func_.benchmark_result.nc_latency
        p50 = latency_res.get_latency_percentile(50)
        assert p50 <= latency * 1.05 # short running kernels are subjected to hardware fluctuation
        assert os.path.getsize(test_trace_file_path) > 0

    @pytest.mark.parametrize("bs,seqlen,d,dtype", [
        [1, 4096, 128, np.float32],
        [1, 4096, 128, nl.bfloat16],
        [1, 5120, 128, nl.bfloat16]
    ])
    def test_allocated_attention_for_SD_numberic(self, bs, seqlen, d, dtype):
        q = np.random.random_sample((bs, d, seqlen)).astype(np.float32)
        k = np.random.random_sample((bs, d, seqlen)).astype(np.float32)
        v = np.random.random_sample((bs, seqlen, d)).astype(np.float32)

        q_dev = nl.static_cast(q, dtype)
        k_dev = nl.static_cast(k, dtype)
        v_dev = nl.static_cast(v, dtype)

        out = numeric_func[bs](q_dev, k_dev, v_dev)
        out = nl.static_cast(out, np.float32)
        golden_result = cpu_golden_attn(q, k, v)
        assert np.allclose(out, golden_result, atol=1e-2)
