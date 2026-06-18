"""
Copyright (C) 2026, Amazon.com. All Rights Reserved

PyTorch/XLA runner for the all-gather + matmul ring NKI tutorial.

Launches ``allgather_matmul_ring`` across ranks using
``torch_xla.distributed.xla_multiprocessing.spawn``, constructs
deterministic LHS/RHS shards, and validates each rank's output against a
reference matmul computed on the host.

Run on a single trn2 device with world_size=8, LNC=1 (8 physical cores
forming a ring within one device):

    NEURON_CC_FLAGS="--lnc=1" \\
    NEURON_LOGICAL_NC_CONFIG=1 \\
    NEURONCORE_NUM_DEVICES=8 \\
      python allgather_matmul_ring_torch.py

``--lnc=1`` tells the compiler to emit code for a single physical core
per rank; ``NEURON_LOGICAL_NC_CONFIG=1`` tells the runtime to launch
that way. Both must agree.

Other world_size × LNC combinations may fail if the replica group does
not map to a valid CPI ring topology on the hardware.
"""

import torch
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.runtime as xr

import nki.collectives as ncc

from allgather_matmul_ring_nki_kernels import (
    allgather_matmul_ring,
    K,
    M_LOCAL,
    N_LOCAL,
    RANK_N,
)


def run_rank(rank):
  device = xm.xla_device()
  world_size = xr.world_size()
  assert world_size == RANK_N, (
      f"need world_size={RANK_N}, got {world_size}"
  )
  replica_group = ncc.ReplicaGroup((tuple(range(world_size)),))

  M_full = M_LOCAL * RANK_N
  N_full = N_LOCAL * RANK_N

  # Deterministic full tensors on every rank; each rank takes its shard.
  torch.manual_seed(0)
  lhs_full = torch.randn((M_full, K), dtype=torch.bfloat16)
  rhs_full = torch.randn((K, N_full), dtype=torch.bfloat16)

  # Reference: this rank owns N_LOCAL columns of the output; compute the
  # full-M matmul on host for that column slice and compare.
  n_start = rank * N_LOCAL
  expected = (
      lhs_full.float() @ rhs_full[:, n_start : n_start + N_LOCAL].float()
  ).to(torch.bfloat16).reshape(RANK_N, M_LOCAL, N_LOCAL)

  # This rank's LHS and RHS shards.
  m_start = rank * M_LOCAL
  lhs_shard = lhs_full[m_start : m_start + M_LOCAL, :].contiguous().to(device)
  rhs_shard = rhs_full[:, n_start : n_start + N_LOCAL].contiguous().to(device)

  out = allgather_matmul_ring(lhs_shard, rhs_shard, replica_group)
  xm.mark_step()
  actual = out.cpu()

  actual_f, expected_f = actual.float(), expected.float()
  max_abs_err = (actual_f - expected_f).abs().max().item()
  rel_err = max_abs_err / (expected_f.abs().max().item() + 1e-9)
  ok = rel_err < 0.01
  print(
      f"Rank {rank}: out shape={tuple(actual.shape)} "
      f"max_abs_err={max_abs_err:.3f} rel_err={rel_err:.4f} "
      f"{'PASS' if ok else 'FAIL'}",
  )
  assert ok, f"Rank {rank}: output mismatch"


if __name__ == "__main__":
  xmp.spawn(run_rank, args=())
