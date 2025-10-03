import torch
import unittest
import os
import sys
import pytest
import torch.nn as nn

os.environ["NEURON_CC_FLAGS"] = "--model-type=transformer --distribution-strategy=llm-training"
os.environ["NEURON_FUSE_SOFTMAX"] = "1"

import torch_xla
import torch_xla.core.xla_model as xm
from neuronxcc.starfish.support.util import allclose

from flash_attention import nki_flash_attn_func

@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float32])
@pytest.mark.parametrize('causal,sliding_window', [
    (True, -1),             # causal, no sliding window
    (True, 128),            # causal, sliding window of size 128
    (True, float("inf")),   # causal, sliding window size same as sequence length
    (False, -1),            # non-causal, no sliding window
])
def test_attention(dtype, causal, sliding_window):
    def torch_golden_attn_cpu(query_states, key_states, value_states, causal, sliding_window):
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * (query_states.shape[-1] ** (-0.5))

        if sliding_window == float("inf"): # same as causal attention
            causal = True
            sliding_window = -1

        if sliding_window > 0:
            causal_mask = torch.triu(
                torch.tril(torch.ones(1, 1, query_states.shape[2], key_states.shape[2])),
                diagonal=-(sliding_window - 1),
            ).bool()
            causal_mask = ~causal_mask
            attn_weights = attn_weights.masked_fill_(causal_mask, -10000.0)
        elif causal:
            causal_mask = torch.triu(torch.ones((1, 1, query_states.shape[2], key_states.shape[2])), diagonal=1).bool()
            attn_weights = attn_weights.masked_fill_(causal_mask, -10000.0)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.double).to(query_states.dtype)

        attn_output = torch.matmul(attn_weights, value_states)
        return attn_output

    torch.manual_seed(0)
    bs = 1
    num_heads = 4
    head_dim = 128
    seq_len = 4096
    query_states_cpu = torch.randn(bs, num_heads, seq_len, head_dim, dtype=dtype) - 0.5
    query_states_cpu.requires_grad_()
    key_states_cpu = torch.randn(bs, num_heads, seq_len, head_dim, dtype=dtype) - 0.5
    key_states_cpu.requires_grad_()
    value_states_cpu = torch.randn(bs, num_heads, seq_len, head_dim, dtype=dtype) - 0.5
    value_states_cpu.requires_grad_()

    # Run the CPU golden results
    golden_attn = torch_golden_attn_cpu(query_states_cpu, key_states_cpu, value_states_cpu, causal, sliding_window)
    loss_golden = torch.sum(golden_attn**2)
    loss_golden.backward()

    # Run the Neuron kernel implementation with torch-xla
    query_states_xla = query_states_cpu.to(xm.xla_device()).detach().requires_grad_()
    key_states_xla = key_states_cpu.to(xm.xla_device()).detach().requires_grad_()
    value_states_xla = value_states_cpu.to(xm.xla_device()).detach().requires_grad_()
    attn_nki = nki_flash_attn_func(query_states_xla, key_states_xla, value_states_xla, causal=causal, sliding_window=sliding_window)
    loss_actual = torch.sum(attn_nki**2)
    loss_actual.backward()
    xm.mark_step()

    actual_dv = value_states_xla.grad.to('cpu')
    actual_dq = query_states_xla.grad.to('cpu')
    actual_dk = key_states_xla.grad.to('cpu')

    # Compare against cpu result
    assert(allclose(loss_actual.to('cpu').to(torch.float32).detach().numpy(), loss_golden.to(torch.float32).detach().numpy(), atol=1e-5, rtol=0.05, verbose=1))
    assert(allclose(actual_dv.to(torch.float32).numpy(), value_states_cpu.grad.to(torch.float32).numpy(), atol=1e-5, rtol=0.05, verbose=1))
    assert(allclose(actual_dq.to(torch.float32).numpy(), query_states_cpu.grad.to(torch.float32).numpy(), atol=1e-5, rtol=0.05, verbose=1))
    assert(allclose(actual_dk.to(torch.float32).numpy(), key_states_cpu.grad.to(torch.float32).numpy(), atol=1e-5, rtol=0.05, verbose=1))