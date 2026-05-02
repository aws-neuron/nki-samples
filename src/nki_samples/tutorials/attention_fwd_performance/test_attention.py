"""
Copyright (C) 2026, Amazon.com. All Rights Reserved

Test attention kernels against a reference NumPy implementation.
"""
from attention_kernels import *
import nki
import nki.language as nl
import numpy as np

####################################################################
# v0: Using Numpy to implement self-attention
####################################################################
def numpy_attention(q, k, v):
    """NumPy reference implementation"""
    d_head, seqlen_q = q.shape
    seqlen_kv = seqlen_q

    # Q^T @ K (NKI layout: partition=d_head, free=seqlen)
    qk = np.matmul(q.T, k)

    # Softmax
    row_max = np.max(qk, axis=1, keepdims=True)
    norm_row = qk - row_max
    exp_row = np.exp(norm_row)
    sum_row = np.sum(exp_row, axis=1, keepdims=True)
    scores = exp_row / sum_row

    # scores @ V^T
    attn_out = np.matmul(scores, v.T)

    return attn_out

####################################################################
# Test each attention kernel version
####################################################################
def test_attn_version(version, seqlen=1024):
    d_head = 128
    np.random.seed(42)
    q = np.random.rand(d_head, seqlen).astype(np.float32)
    k = np.random.rand(d_head, seqlen).astype(np.float32)
    v = np.random.rand(d_head, seqlen).astype(np.float32)

    numpy_output = numpy_attention(q, k, v)
    attn_out = np.array(version(q, k, v))
    match = np.allclose(attn_out, numpy_output, atol=1e-2, rtol=1e-2)
    print(f"{version.__name__}: {'PASS' if match else 'FAIL'} (max diff: {np.max(np.abs(attn_out - numpy_output)):.6f})")
    return match

if __name__ == "__main__":
    # v1 and v2 only support 128x128
    results = []
    for v in [attn_fwd_v1, attn_fwd_v2]:
        results.append(test_attn_version(v, seqlen=128))
    for v in [attn_fwd_v3, attn_fwd_v4, attn_fwd_v5, attn_fwd_v6,
              attn_fwd_v7, attn_fwd_v8, attn_fwd_v8a]:
        results.append(test_attn_version(v, seqlen=1024))

    passed = sum(results)
    total = len(results)
    print(f"\n{passed}/{total} versions passed")
