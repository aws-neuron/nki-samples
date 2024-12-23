"""
Copyright (c) 2024, Amazon.com. All Rights Reserved
"""
import pytest
from nki_samples.reference.allocated_fused_linear import allocated_fused_rms_norm_qkv
from neuronxcc.nki import benchmark, baremetal, simulate_kernel
import neuronxcc.nki.language as nl
import numpy as np

bench_func = benchmark(warmup=5, iters=10)(allocated_fused_rms_norm_qkv)

np.random.seed(0)


def rms_norm(hidden, gamma, eps=1e-6):
  rms = np.sqrt(np.mean(np.square(hidden), axis=-1, keepdims=True))
  norm = hidden * np.reciprocal(rms + eps)
  if gamma is not None:
    norm *= gamma
  return norm

def cpu_golden_result(hidden, gamma, qkv_weights, dtype, do_norm=True):
  if do_norm:
      hidden = rms_norm(hidden, gamma)
  qkv_out = (hidden @ qkv_weights).astype(dtype)
  return qkv_out

class TestRMSNormQKV:
  @pytest.mark.parametrize("batch, seqlen, dim, d_head, dtype, latency", [
    [1, 128, 512, 512, np.float16, 25],
    [1, 512, 1024, 384, nl.bfloat16, 40],
    [1, 128, 1024, 512, nl.bfloat16, 28],
    # [1, 1024, 8192, 512, nl.bfloat16, 301 * 1.02], # FIXME: performance is flaky
  ])
  def test_allocated_rmsnorm_qkv_perf(self, batch, seqlen, dim, d_head, dtype, latency):
    hidden = np.random.random_sample((batch, seqlen, dim)).astype(np.float32)
    weights = np.random.random_sample((dim, d_head)).astype(np.float32)

    hidden = nl.static_cast(hidden, dtype)
    weights = nl.static_cast(weights, dtype)

    bench_func(hidden, weights)
    latency_res = bench_func.benchmark_result.nc_latency
    p99 = latency_res.get_latency_percentile(50)
    assert p99 <= latency

  @pytest.mark.simulation
  @pytest.mark.parametrize("batch, seqlen, dim, d_head, dtype", [
    [1, 128, 512, 512, np.float16],
    [1, 512, 1024, 384, nl.bfloat16],
    [1, 128, 1024, 512, nl.bfloat16],
    [1, 1024, 8192, 512, nl.bfloat16]
  ])
  def test_allocated_rmsnorm_qkv_numeric(self, simulation_only, batch, seqlen, dim, d_head, dtype):
    hidden = np.random.random_sample((batch, seqlen, dim))
    weights = np.random.random_sample((dim, d_head))

    hidden_dev = nl.static_cast(hidden, dtype)
    weights_dev = nl.static_cast(weights, dtype)

    numeric_func = baremetal(allocated_fused_rms_norm_qkv)
    if simulation_only:
      out = simulate_kernel(numeric_func, hidden_dev, weights_dev)
    else:
      out = numeric_func(hidden_dev, weights_dev)
    out = nl.static_cast(out, np.float32)
    golden_res = nl.static_cast(cpu_golden_result(hidden, None, weights, dtype, do_norm=True), np.float32)
    assert np.allclose(out, golden_res, atol=1e-2, rtol=1e-2)

