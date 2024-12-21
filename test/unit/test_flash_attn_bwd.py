"""
Copyright (c) 2023, Amazon.com. All Rights Reserved
"""
import pytest
from nki_samples.reference.attention import flash_attn_bwd
from neuronxcc.nki import benchmark, baremetal, simulate_kernel
import neuronxcc.nki.language as nl
import numpy as np

xfail = pytest.mark.arch_specific_xfail


bench_func = benchmark(warmup=5, iters=10)(flash_attn_bwd)

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

def softmax_dx(dy: np.ndarray, y: np.ndarray, dim: int, mixed_precision=False):
    # dx_i = (dy_i - sum(dy_k*y_k)) * y_i
    prod = dy * y
    if mixed_precision:
        reduce = np.add.reduce(prod.astype(np.float32), axis=dim, keepdims=True).astype(dy.dtype)
    else:
        reduce = np.add.reduce(prod, axis=dim, keepdims=True)
    subtract = dy - reduce
    return subtract * y

def cpu_attention_backward(q, k, v, dy, use_causal_mask=True, mixed_precision=True):
  """
  Compute the attention backward with the softmax recomputation
  """
  def mixed_precision_matmul(a, b):
    input_dtype = a.dtype
    a, b = a.astype(np.float32), b.astype(np.float32)
    c = np.matmul(a, b)
    return c.astype(input_dtype)

  _, _, d, _ = q.shape
  # Compute golden output
  softmax_scale = 1.0 / (d ** 0.5)
  q_scaled = q * softmax_scale
  raw_score = mixed_precision_matmul(q_scaled.transpose(0, 1, 3, 2), k)

  if use_causal_mask:
    # raw_score has K seq in the most inner dim
    # we want to mask all elements where Q idx is smaller than K idx with -inf
    # this maps to the upper triangle of the final two axes
    for i in range(raw_score.shape[0]):
      for j in range(raw_score.shape[1]):
        # -inf triggers invalid input error in softmax implementation, use a small negative instead
        # k=1 to exclude the diagonal, because each token can still attend to itself
        raw_score[i, j][np.triu_indices_from(raw_score[i, j], k=1)] = -9984.0

  norm_score, cached_negative_max, cached_sum_reciprocal = \
    softmax(raw_score, dim=-1, mixed_precision=mixed_precision, return_max_reduce=True)

  # Calculate softmax_dy = (dL/dy)^T @ V
  softmax_dy = mixed_precision_matmul(dy.transpose(0, 1, 3, 2), v)

  # Calculate dv = (dL/dy) @ softmax_y
  dv_golden = mixed_precision_matmul(dy, norm_score)

  # Calculate softmax_dx
  softmax_dx_golden = softmax_dx(softmax_dy, norm_score, dim=-1, mixed_precision=mixed_precision)

  # Calculate dq
  dq_golden = mixed_precision_matmul(k, softmax_dx_golden.transpose(0, 1, 3, 2)) * softmax_scale

  # Calculate dk
  dk_golden = mixed_precision_matmul(q_scaled, softmax_dx_golden)

  # Calculate output projection
  o_proj = np.matmul(norm_score, v.transpose(0, 1, 3, 2)).transpose(0, 1, 3, 2)

  # Calculate 
  return dq_golden, dk_golden, dv_golden, cached_negative_max, cached_sum_reciprocal, o_proj

class TestAttention:

    @xfail # P167481231
    @pytest.mark.parametrize("bs, nheads, seqlen, d, dtype, latency", [
        [1, 4, 32*1024, 128, nl.bfloat16, 117000],
    ])
    def test_flash_attn_bwd_perf(self, bs, nheads, seqlen, d, dtype, latency):
        q = (np.random.random_sample([bs, nheads, d, seqlen]) - 0.5) * 2
        k = (np.random.random_sample([bs, nheads, d, seqlen]) - 0.5) * 2
        v = (np.random.random_sample([bs, nheads, d, seqlen]) - 0.5) * 2
        dy = (np.random.random_sample([bs, nheads, d, seqlen]) - 0.5) * 2
        o_proj = (np.random.random_sample([bs, nheads, d, seqlen]) - 0.5) * 2
        lse = np.random.random_sample([bs, nheads, nl.tile_size.pmax, seqlen // nl.tile_size.pmax]).astype(np.float32)
        seed = None

        q = nl.static_cast(q, dtype)
        k = nl.static_cast(k, dtype)
        v = nl.static_cast(v, dtype)
        o_proj = nl.static_cast(o_proj, dtype)
        dy = nl.static_cast(dy, dtype)  

        bench_func_ = bench_func[bs, nheads]
        bench_func_(q, k, v, o_proj, dy, lse, seed,
                    use_causal_mask=True, mixed_precision=True)
        latency_res = bench_func_.benchmark_result.nc_latency
        p99 = latency_res.get_latency_percentile(50)
        assert p99 <= latency

    @pytest.mark.simulation
    @pytest.mark.parametrize("bs, nheads, seqlen, d, dtype", [
        [1, 4, 4096, 128, np.float32],
    ])
    def test_flash_attn_bwd_numerical(self, simulation_only, bs, nheads, seqlen, d, dtype):
        q = (np.random.random_sample([bs, nheads, d, seqlen]) - 0.5) * 2
        k = (np.random.random_sample([bs, nheads, d, seqlen]) - 0.5) * 2
        v = (np.random.random_sample([bs, nheads, d, seqlen]) - 0.5) * 2
        dy = (np.random.random_sample([bs, nheads, d, seqlen]) - 0.5) * 2
        q = nl.static_cast(q, dtype)
        k = nl.static_cast(k, dtype)
        v = nl.static_cast(v, dtype)
        dy = nl.static_cast(dy, dtype)
        seed = None

        dq_golden, dk_golden, dv_golden, cached_negative_max, cached_sum_reciprocal, o_proj = \
          cpu_attention_backward(q, k, v, dy, use_causal_mask=True)
        cached_negative_max = cached_negative_max.reshape(bs, nheads, seqlen // nl.tile_size.pmax,
                                                          nl.tile_size.pmax).transpose(0, 1, 3, 2)
        cached_sum_reciprocal = cached_sum_reciprocal.reshape(bs, nheads, seqlen // nl.tile_size.pmax,
                                                              nl.tile_size.pmax).transpose(0, 1, 3, 2)
        lse = -1.0 * (cached_negative_max + np.log(cached_sum_reciprocal))

        numeric_func = baremetal(flash_attn_bwd)
        if simulation_only:
           out_dq, out_dk, out_dv = simulate_kernel(numeric_func[bs, nheads], q, k, v, o_proj, dy, lse, seed,
                                                          use_causal_mask=True,
                                                          mixed_precision=True)
        else:
          out_dq, out_dk, out_dv = numeric_func[bs, nheads](q, k, v, o_proj, dy, lse, seed,
                                                          use_causal_mask=True,
                                                          mixed_precision=True)

        assert np.allclose(out_dq, dq_golden, atol=1e-2)
        assert np.allclose(out_dk, dk_golden, atol=1e-2)
        assert np.allclose(out_dv, dv_golden, atol=1e-2)
