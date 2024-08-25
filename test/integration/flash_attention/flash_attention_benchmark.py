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

from perf_utils.LatencyCollector import benchmark

if len(sys.argv) != 2:
    print("Usage: python flash_attention_benchmark.py <metric_path>")
    exit(1)
metric_path = os.path.abspath(sys.argv[1])

torch.manual_seed(0)
dtype = torch.bfloat16
bs = 1
num_heads = 4
head_dim = 128
seq_len = 32*1024
query_states_cpu = torch.randn(bs, num_heads, seq_len, head_dim, dtype=dtype) - 0.5
query_states_cpu.requires_grad_()
key_states_cpu = torch.randn(bs, num_heads, seq_len, head_dim, dtype=dtype) - 0.5
key_states_cpu.requires_grad_()
value_states_cpu = torch.randn(bs, num_heads, seq_len, head_dim, dtype=dtype) - 0.5
value_states_cpu.requires_grad_()

# Run the Neuron kernel implementation with torch-xla
query_states_xla = query_states_cpu.to(xm.xla_device()).detach().requires_grad_()
key_states_xla = key_states_cpu.to(xm.xla_device()).detach().requires_grad_()
value_states_xla = value_states_cpu.to(xm.xla_device()).detach().requires_grad_()

def model_fwd_bwd(q, k, v):
    attn_nki = nki_flash_attn_func(q, k, v)
    loss_actual = torch.sum(attn_nki**2)
    loss_actual.backward()
    xm.mark_step()
    return loss_actual 

n_runs = 10
bench_result = benchmark(n_runs, f"flash_attention_bs{bs}_heads{num_heads}_seq{seq_len}_head_dim{head_dim}",
                         model_fwd_bwd,
                         (query_states_xla, key_states_xla, value_states_xla),
                         metric_path)