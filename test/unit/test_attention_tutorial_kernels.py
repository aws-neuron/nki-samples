"""
Copyright (c) 2025, Amazon.com. All Rights Reserved
"""
import random
import pytest
import numpy as np
import neuronxcc.nki.language as nl
from neuronxcc.nki import benchmark, baremetal, simulate_kernel
from nki_samples.tutorials.attention_fwd_performance.attention_kernels import *

# Test parameters
VERSIONS = [
    attn_fwd_v1, attn_fwd_v2, attn_fwd_v3, attn_fwd_v4, attn_fwd_v5,
    attn_fwd_v6, attn_fwd_v7, attn_fwd_v8, attn_fwd_v8a, attn_fwd_v9, 
    attn_fwd_v10, attn_fwd_v11
]

SKIP_V10 = random.randint(0, 1)

def numpy_attention(q, k, v):
    """NumPy reference implementation"""
    d_head, seqlen_q = q.shape
    qk = np.matmul(q.T, k)
    row_max = np.max(qk, axis=1, keepdims=True)
    norm_row = qk - row_max
    exp_row = np.exp(norm_row)
    sum_row = np.sum(exp_row, axis=1, keepdims=True)
    scores = exp_row / sum_row
    v_t = v.T
    attn_out = np.matmul(scores, v_t)
    return attn_out

@pytest.mark.simulation
@pytest.mark.parametrize("version", VERSIONS)
def test_attention_accuracy(simulation_only, version):
    """Test attention kernel accuracy against numpy reference"""
    if not simulation_only and version == attn_fwd_v10 and SKIP_V10:
        pytest.skip("Skipping v10 this iteration")
    if not simulation_only and version == attn_fwd_v9 and not SKIP_V10:
        pytest.skip("Skipping v9 this iteration")    
    # Use smaller sequence length for v1 and v2
    seqlen = 128 if version in [attn_fwd_v1, attn_fwd_v2] else 4096
    d_head = 128
    dtype = nl.float32
    
    # Generate test data
    q = nl.static_cast((np.random.random_sample([d_head, seqlen]) - 0.5) * 2, dtype)
    k = nl.static_cast((np.random.random_sample([d_head, seqlen]) - 0.5) * 2, dtype)
    v = nl.static_cast((np.random.random_sample([d_head, seqlen]) - 0.5) * 2, dtype)
    
    numeric_func = baremetal(version)
    # Compare outputs
    if simulation_only:
        attn_out = simulate_kernel(numeric_func, q, k, v)
    else:
        attn_out = numeric_func(q, k, v)

    numpy_output = numpy_attention(q, k, v)
    
    assert np.allclose(attn_out.astype(np.float32), numpy_output.astype(np.float32), atol=1e-2)
