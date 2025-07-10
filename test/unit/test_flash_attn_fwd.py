"""
Copyright (c) 2023, Amazon.com. All Rights Reserved
"""
import pytest
from nki_samples.reference.attention import flash_fwd, FlashConfig
from neuronxcc.nki import benchmark, baremetal, simulate_kernel
import neuronxcc.nki.language as nl
import numpy as np

bench_func = benchmark(warmup=5, iters=10)(flash_fwd)
 
def softmax(x: np.ndarray, dim: int, zero_max_mode=False,
            mixed_precision=False, return_max_reduce=False):
    max_value = np.amax(x, axis=dim, keepdims=True)
    max_value = np.maximum(0, max_value) if zero_max_mode else max_value
    exp = np.exp(x - max_value)
    if mixed_precision:
        reduce = np.add.reduce(exp.astype(np.float32), axis=dim, keepdims=True).astype(x.dtype)
    else:
        reduce = np.add.reduce(exp, axis=dim, keepdims=True)
    if return_max_reduce:
        return exp / reduce, -max_value, np.reciprocal(reduce)
    return exp / reduce
 
 
def cpu_attention_forward(q, k, v, use_causal_mask=True, sliding_window=-1, mixed_precision=True):
    def mixed_precision_matmul(a, b):
        input_dtype = a.dtype
        a, b = a.astype(np.float32), b.astype(np.float32)
        c = np.matmul(a, b)
        return c.astype(input_dtype)

    _, _, d, _ = q.shape

    # Compute golden output
    softmax_scale = 1.0 / (d ** 0.5)
    q_scaled = q * softmax_scale
    nheads = q.shape[1]
    kv_heads = k.shape[1]
    if nheads > kv_heads:
        k = np.repeat(k, nheads//kv_heads, axis=1)
        v = np.repeat(v, nheads//kv_heads, axis=1)
    raw_score = mixed_precision_matmul(q_scaled.transpose(0, 1, 3, 2), k)

    if use_causal_mask:
        _, _, Q, K = raw_score.shape

        q_idx = np.arange(Q)[:, None]
        k_idx = np.arange(K)[None, :]

        if sliding_window > 0:
            mask = (k_idx > q_idx) | (k_idx < q_idx - sliding_window + 1)
        else:  # causal
            mask = k_idx > q_idx

        # Broadcast mask to shape (1, 1, Q, K)
        raw_score = raw_score.copy()
        # -inf triggers invalid input error in softmax implementation, use a small negative instead
        raw_score[:, :, mask] = -9984.0


    norm_score, cached_negative_max, cached_sum_reciprocal = \
        softmax(raw_score, dim=-1, mixed_precision=mixed_precision, return_max_reduce=True)

    # Transpose the result so it has the same layout as ours
    out_golden = mixed_precision_matmul(norm_score, v.transpose(0, 1, 3, 2)).transpose(0, 1, 3, 2)

    return out_golden, cached_negative_max, cached_sum_reciprocal
 
class TestAttention:
 
    @pytest.mark.parametrize("bs, nheads, seqlen_q, seqlen_k, d, dtype, use_causal_mask, \
                              sliding_window, mixed_precision, training, tile_size, kv_heads, should_transpose_v, latency", [
    [1, 6, 32*1024, 32*1024, 96, nl.bfloat16, True, -1, True, True, 2048, 3, False, 87000000000],
    [1, 1, 32*1024, 32*1024, 96, nl.bfloat16, True, -1, True, False, 2048, None, False, 15100000000],
    # Non-square
    [1, 3, 32*1024, 16*1024, 96, nl.bfloat16, True, -1, True, False, 2048, None, False, 7550000000],
    [1, 3, 16*1024, 32*1024, 96, nl.bfloat16, True, -1, True, False, 2048, None, False, 7550000000],
    # Causal vs. Sliding - test sliding window is faster
    [1, 1, 16*1024, 16*1024, 96, nl.bfloat16, True, -1, True, False, 2048, None, False, 4000000000], 
    [1, 1, 16*1024, 16*1024, 96, nl.bfloat16, True, 4096, True, False, 2048, None, False, 3000000000],
    ])
    def test_flash_attn_fwd_perf(self, bs, nheads, seqlen_q, seqlen_k, d, dtype, use_causal_mask,
                                 sliding_window, mixed_precision, training, tile_size, kv_heads, should_transpose_v,latency):
        q = (np.random.random_sample([bs, nheads, d, seqlen_q]) - 0.5) * 2
        k = (np.random.random_sample([bs, nheads, d, seqlen_k]) - 0.5) * 2
        if should_transpose_v:
            v = (np.random.random_sample([bs, nheads, d, seqlen_k]) - 0.5) * 2
        else:
            v = (np.random.random_sample([bs, nheads, seqlen_k, d]) - 0.5) * 2
        o_proj = np.zeros(shape=[bs, nheads, seqlen_q, d], dtype=dtype)
        out_lse = np.zeros(shape=[bs, nheads, int(nl.tile_size.pmax), seqlen_q // nl.tile_size.pmax], 
                                  dtype=nl.float32 if mixed_precision else dtype) if training else None
        seed = None
        
        q = nl.static_cast(q, dtype)
        k = nl.static_cast(k, dtype)
        v = nl.static_cast(v, dtype)
        config = FlashConfig(**{'seq_tile_size':tile_size, 'training':training, 'should_transpose_v':should_transpose_v})

        heads = nheads if kv_heads is None else kv_heads

        bench_func_ = bench_func[bs, heads]
        bench_func_(q, k, v, seed, use_causal_mask=use_causal_mask, sliding_window=sliding_window,
                    mixed_precision=mixed_precision, config=config)
        latency_res = bench_func_.benchmark_result.nc_latency
        p50 = latency_res.get_latency_percentile(50)
        assert p50 <= latency
    
    @pytest.mark.simulation
    @pytest.mark.parametrize("bs, nheads, seqlen_q, seqlen_k, d, dtype, use_causal_mask, \
                              sliding_window, training, tile_size, kv_heads, should_transpose_v", [
    [1, 6, 4096, 4096, 128, np.float32, True, -1, True, 2048, 3, False],
    [1, 1, 4096, 4096, 128, np.float32, True, -1, False, 2048, None, False],
    [1, 1, 8192, 4096, 128, np.float32, True, -1, False, 2048, None, False],
    [1, 1, 4096, 8192, 128, np.float32, True, -1, False, 2048, None, False],
    [1, 1, 4096, 4096, 128, np.float32, True, 1024, False, 2048, None, False],
    ])
    def test_flash_attn_fwd_numerical(self, simulation_only, bs, nheads, seqlen_q, seqlen_k, d, dtype, use_causal_mask, 
                                    sliding_window, training, tile_size, kv_heads, should_transpose_v):
        q = (np.random.random_sample([bs, nheads, d, seqlen_q]) - 0.5) * 2
        k = (np.random.random_sample([bs, kv_heads or nheads, d, seqlen_k]) - 0.5) * 2
        if should_transpose_v:
            v = (np.random.random_sample([bs, nheads, d, seqlen_k]) - 0.5) * 2
            cpu_permute = (0, 1, 2, 3)
        else:
            v = (np.random.random_sample([bs, kv_heads or nheads, seqlen_k, d]) - 0.5) * 2
            cpu_permute = (0, 1, 3, 2)

        q = nl.static_cast(q, dtype)
        k = nl.static_cast(k, dtype)
        v = nl.static_cast(v, dtype)
        seed = None

        o_proj_golden, cached_negative_max, cached_sum_reciprocal  = \
          cpu_attention_forward(q, k, v.transpose(cpu_permute), use_causal_mask=use_causal_mask, sliding_window=sliding_window, mixed_precision=True)
        o_proj_golden = o_proj_golden.transpose(0,1,3,2) # (b,h, d, seq)
        cached_negative_max = cached_negative_max.reshape(bs, nheads, seqlen_q // nl.tile_size.pmax,
                                                          nl.tile_size.pmax).transpose(0, 1, 3, 2)
        cached_sum_reciprocal = cached_sum_reciprocal.reshape(bs, nheads, seqlen_q // nl.tile_size.pmax,
                                                              nl.tile_size.pmax).transpose(0, 1, 3, 2)
        lse_golden = -1.0 * (cached_negative_max + np.log(cached_sum_reciprocal)) if training else None
        config = FlashConfig(**{'seq_tile_size':tile_size, 'training':training, 'should_transpose_v':should_transpose_v})

        heads = nheads if kv_heads is None else kv_heads

        numeric_func = baremetal(flash_fwd)
        if simulation_only:
            results = simulate_kernel(numeric_func[bs, heads], q, k, v, seed,
                                          use_causal_mask=use_causal_mask,
                                          sliding_window=sliding_window,
                                          mixed_precision=True,
                                          config=config)
        else:
            results = numeric_func[bs, heads](q, k, v, seed,
                                          use_causal_mask=use_causal_mask,
                                          sliding_window=sliding_window,
                                          mixed_precision=True,
                                          config=config)

        if training:
            o_proj, out_lse = results
            assert np.allclose(o_proj, o_proj_golden, atol=1e-2)
            assert np.allclose(out_lse, lse_golden, atol=1e-2)
        else:
            o_proj = results
            assert np.allclose(o_proj, o_proj_golden, atol=1e-2)
