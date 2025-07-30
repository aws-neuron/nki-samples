"""
Copyright (c) 2025, Amazon.com. All Rights Reserved

Kernels written by the AWS Neuron Team.

WARNING: These kernels:
   - Are tested only against internal nightly compiler builds
   - May rely on internal compiler feature/flags and not be compatible with public NeuronSDK
   - Have not been extensively tested across all input configurations
   - Carry no compatibility guarantees
   - The behavior of these kernels may be modified without prior notice

Flash Paged Attention kernels with variable-length sequence inputs.

"""

import os
from dataclasses import dataclass, fields
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from ml_dtypes import bfloat16

from constants import B_P_SIZE, B_FMAX_SIZE
from varlen_attention_kernel import flash_attn_varlen_nkifunc
from execution_planner import (
    VarlenAttentionPlanner,
    TilePlan as ContextAttnPlan,
)
from test_utils import (
    BlockDiagonalCausalFromBottomRightMask,
    get_active_block_tables,
    ceil_div,
    is_power_of_2,
    pad_to_multiple,
    pad_to_next_power_of_2,
    convert_torch_tensor_to_numpy,
)


def _decide_execution_mode(exec_mode):
    _supported_exec_mode = [
        "xla",
        "baremetal",
    ]
    if not exec_mode:
        exec_mode = os.getenv("TEST_EXEC_MODE", "")
    if exec_mode.lower() not in _supported_exec_mode:
        if exec_mode:
            print(f"Execution mode {exec_mode} is not supported.")
        exec_mode = _supported_exec_mode[0]
        print(f"Default to {exec_mode} mode for execution")
    else:
        exec_mode = exec_mode.lower()
        print(f"Use {exec_mode} mode for execution")
    return exec_mode


@dataclass(frozen=True)
class ContextAttnInputs:
    tile_q_indices: torch.Tensor
    tile_block_tables: torch.Tensor
    tile_masks: torch.Tensor
    num_dynamic_loop_steps: torch.Tensor
    last_tile_indices: torch.Tensor
    q_update_pred: Optional[torch.Tensor]

    def as_dict(self, prefix=""):
        return {
            (prefix + field.name): getattr(self, field.name) for field in fields(self)
        }


class NKIFlashPagedAttentionRunner:
    def __init__(
        self,
        query_lens,
        context_lens,
        large_q_tile_size,
        large_kv_tile_size,
        block_size,
        *,
        dynamic_loop_unrolling_size=8,
        skip_active=False,
        exec_mode=None,
    ):
        """
        Kernel executor to generate and cache tile-plan, and dispatch for execution
        """
        assert large_kv_tile_size >= B_P_SIZE
        self.query_lens = query_lens
        self.context_lens = context_lens
        self.large_q_tile_size = large_q_tile_size
        self.large_kv_tile_size = large_kv_tile_size
        self.block_size = block_size
        self.dynamic_loop_unrolling_size = dynamic_loop_unrolling_size
        self.exec_mode = _decide_execution_mode(exec_mode)
        self.skip_active = skip_active
        self.numpy_kernel_use_bf16 = True  # use ml_dtypes.bfloat16 for baremetal mode
        self.num_actual_tokens = None
        self.prefill_ctx_inputs: Optional[ContextAttnInputs] = None
        self.decode_ctx_inputs: Optional[ContextAttnInputs] = None
        assert self.batch_size <= B_P_SIZE
        assert is_power_of_2(self.large_q_tile_size)
        assert is_power_of_2(self.large_kv_tile_size)
        self._preprocess()

    @property
    def batch_size(self):
        return len(self.query_lens)

    def _get_prefill_decode_batch_size(self):
        decode_batch_size = 0
        assert torch.all(self.query_lens > 0), f"Expect nonzero {self.query_lens}"
        for x in reversed(self.query_lens):
            if x > 1:
                break
            decode_batch_size += 1
        batch_size = self.batch_size
        assert decode_batch_size <= batch_size
        prefill_batch_size = batch_size - decode_batch_size
        assert prefill_batch_size > 0 and decode_batch_size > 0, (
            "On-chip while loop cannot scale to zero. "
            "Must have both prefill and decode requests"
        )
        return prefill_batch_size, decode_batch_size

    def _decide_padded_query_len(self):
        if self.num_actual_tokens < B_P_SIZE:
            num_active_tokens_after_padding = pad_to_next_power_of_2(
                self.num_actual_tokens,
            )
        elif self.num_actual_tokens < B_FMAX_SIZE:
            num_active_tokens_after_padding = pad_to_multiple(
                self.num_actual_tokens,
                B_P_SIZE,
            )
        else:
            num_active_tokens_after_padding = pad_to_multiple(
                self.num_actual_tokens,
                B_FMAX_SIZE,
            )
        self.num_active_tokens_after_padding = pad_to_multiple(
            num_active_tokens_after_padding,
            self.large_q_tile_size,
        )

    def _preprocess(self):
        self.num_actual_tokens = self.query_lens.sum().item()
        self._decide_padded_query_len()
        prefill_batch_size, decode_batch_size = self._get_prefill_decode_batch_size()
        self._build_kernel_plan(prefill_batch_size, decode_batch_size)

        self.prefill_batch_size = prefill_batch_size
        self.active_mask = self._build_active_token_mask()

    def get_num_active_tokens_after_padding(self):
        return self.num_active_tokens_after_padding

    def prepare_tile_plan_inputs(self, block_tables, max_kv_cache_size):
        if self.prefill_batch_size > 0:
            # prepare prefill
            self.prefill_ctx_inputs = self._build_inputs_from_plan(
                ctx_plan=self.prefill_plan,
                block_tables_2d=block_tables[: self.prefill_batch_size],
                context_lens=self.context_lens[: self.prefill_batch_size],
                is_decode_plan=False,
                max_kv_cache_size=max_kv_cache_size,
            )
        if self.prefill_batch_size < self.batch_size:
            # prepare decode
            decode_start_offset = (
                self.query_lens[: self.prefill_batch_size].sum().item()
            )
            self.decode_ctx_inputs = self._build_inputs_from_plan(
                ctx_plan=self.decode_plan,
                block_tables_2d=block_tables[self.prefill_batch_size :],
                context_lens=self.context_lens[self.prefill_batch_size :],
                is_decode_plan=True,
                max_kv_cache_size=max_kv_cache_size,
                query_start_offset=decode_start_offset,
            )

    def _build_active_token_mask(self):
        # Build attention masks
        active_mask = BlockDiagonalCausalFromBottomRightMask._from_seqlens(
            query_lens=self.query_lens,
            seq_lens=self.query_lens,
        )
        active_mask = F.pad(
            active_mask,
            (
                0,
                self.num_active_tokens_after_padding - active_mask.shape[1],
                0,
                self.num_active_tokens_after_padding - active_mask.shape[0],
            ),
            "constant",
            0,
        ).to(torch.uint8)
        return active_mask

    def _build_kernel_plan(self, prefill_batch_size, decode_batch_size):
        unroll_size = self.dynamic_loop_unrolling_size
        assert unroll_size > 0

        def pad_plan_for_loop_unroll(plan: ContextAttnPlan, q_pad_value: int = 0):
            assert plan is not None
            if unroll_size == 1:
                return plan
            # make sure unrolling is correct if used in dynamic loop
            num_real_tiles = plan.num_tiles
            padded_tiles = pad_to_multiple(num_real_tiles, unroll_size)
            if padded_tiles != num_real_tiles:
                plan = plan.pad_plan(padded_tiles, q_pad_value=q_pad_value)
            print(f"{plan.num_real_tiles=}")
            print(f"{plan.num_tiles=}")
            return plan

        assert prefill_batch_size > 0 and decode_batch_size > 0
        prefill_planner = VarlenAttentionPlanner(
            self.query_lens[:prefill_batch_size].int().numpy(),
            self.context_lens[:prefill_batch_size].int().numpy(),
            tile_size_q=self.large_q_tile_size,
            tile_size_kv=self.large_kv_tile_size,
            block_size=self.block_size,
        )
        self.prefill_plan = pad_plan_for_loop_unroll(
            prefill_planner.generate_plan(),
            q_pad_value=self.num_active_tokens_after_padding * 10,
        )
        decode_planner = VarlenAttentionPlanner(
            self.query_lens[prefill_batch_size:].int().numpy(),
            self.context_lens[prefill_batch_size:].int().numpy(),
            tile_size_q=1,
            tile_size_kv=self.large_kv_tile_size,
            block_size=self.block_size,
        )
        self.decode_plan = pad_plan_for_loop_unroll(decode_planner.generate_plan())

    def _prepare_buffer_unroll_info(self, plan: ContextAttnPlan, max_num_q_tiles: int):
        max_num_q_tiles = pad_to_next_power_of_2(max_num_q_tiles)
        q_update_pred, last_tile_indices = plan.build_tile_update_indices(
            max_num_q_tiles=max_num_q_tiles,
        )
        # reshape q_update_pred for loop unrolling
        if q_update_pred is not None:
            num_tiles = plan.num_tiles
            assert num_tiles % self.dynamic_loop_unrolling_size == 0
            num_loop_steps = num_tiles // self.dynamic_loop_unrolling_size
            q_update_pred = q_update_pred.reshape(
                num_loop_steps,
                self.dynamic_loop_unrolling_size,
            )
            q_update_pred = torch.tensor(q_update_pred, dtype=torch.uint8)
        # reshape last_tile_indices for fast vector dge loading
        last_tile_indices = last_tile_indices.reshape(max_num_q_tiles, 1)
        last_tile_indices = torch.tensor(last_tile_indices, dtype=torch.int32)
        return q_update_pred, last_tile_indices

    def _build_tile_block_tables(
        self,
        ctx_plan: ContextAttnPlan,
        block_tables_2d,
        context_lens,
        dma_skip_value,
    ):
        num_active_blocks = ceil_div(context_lens, self.block_size).sum().item()
        num_active_blocks = pad_to_multiple(
            num_active_blocks, self.large_kv_tile_size // self.block_size
        )
        active_block_table = get_active_block_tables(
            block_tables_2d,
            context_lens,
            self.block_size,
            num_active_blocks,
        )
        tile_block_tables = ctx_plan.build_tile_block_tables(
            active_block_table,
            skip_value=dma_skip_value,
        )
        tile_block_tables = torch.tensor(tile_block_tables)
        assert tile_block_tables.dtype == torch.int32
        return tile_block_tables

    def _build_inputs_from_plan(
        self,
        ctx_plan: ContextAttnPlan,
        block_tables_2d: torch.Tensor,
        context_lens: torch.Tensor,
        is_decode_plan: bool,
        max_kv_cache_size: int,
        query_start_offset: int = 0,
    ):
        skip_value = 0 if is_decode_plan else self.num_active_tokens_after_padding * 10
        tile_q_indices = torch.tensor(
            ctx_plan.build_tile_q_indices(skip_value=skip_value)
        )
        assert tile_q_indices.dtype == torch.int32
        tile_masks = ctx_plan.build_tile_masks(decode_kq_matmul=is_decode_plan)
        tile_masks = torch.tensor(tile_masks).to(torch.uint8)
        num_dynamic_loop_steps = torch.empty((1, 1), dtype=torch.int32)
        num_dynamic_loop_steps[0] = ceil_div(
            ctx_plan.num_real_tiles, self.dynamic_loop_unrolling_size
        )
        print(f"{num_dynamic_loop_steps=}")

        if is_decode_plan:
            tile_block_tables = self._build_tile_block_tables(
                ctx_plan=ctx_plan,
                block_tables_2d=block_tables_2d,
                context_lens=context_lens,
                dma_skip_value=0,
            )
            q_update_pred, last_tile_indices = self._prepare_buffer_unroll_info(
                ctx_plan,
                self.batch_size,
            )
            return ContextAttnInputs(
                tile_q_indices=tile_q_indices + query_start_offset,
                tile_block_tables=tile_block_tables,
                tile_masks=tile_masks,
                num_dynamic_loop_steps=num_dynamic_loop_steps,
                q_update_pred=q_update_pred,
                last_tile_indices=last_tile_indices,
            )
        else:
            tile_block_tables = self._build_tile_block_tables(
                ctx_plan=ctx_plan,
                block_tables_2d=block_tables_2d,
                context_lens=context_lens,
                dma_skip_value=0,
            )
            q_update_pred, last_tile_indices = self._prepare_buffer_unroll_info(
                ctx_plan,
                min(ctx_plan.num_tiles, B_P_SIZE),
            )
            return ContextAttnInputs(
                tile_q_indices=tile_q_indices,
                tile_block_tables=tile_block_tables,
                tile_masks=tile_masks,
                num_dynamic_loop_steps=num_dynamic_loop_steps,
                q_update_pred=q_update_pred,
                last_tile_indices=last_tile_indices,
            )

    def _prepare_kernel(
        self,
        query,
        k_active,
        v_active,
        k_cache,
        v_cache,
        head_size,
        num_kv_heads,
        mixed_precision,
    ):
        # check input shapes
        # query: (1, num_heads, seq_q, d)
        # key:   (1, num_kv_heads, d, seq_k)
        # value: (1, num_kv_heads, seq_v, d)
        num_heads = query.shape[1]
        num_active_token = self.num_active_tokens_after_padding
        assert (
            query.shape[2] == num_active_token
        ), f"QKV sequence length must be padded to {self.get_num_active_tokens_after_padding()=}"
        assert query.shape == (1, num_heads, num_active_token, head_size)
        assert k_active.shape == (1, num_kv_heads, head_size, num_active_token)
        assert v_active.shape == (1, num_kv_heads, num_active_token, head_size)
        input_kwargs = dict(
            query=query,
            key=k_active,
            value=v_active,
            key_cache=k_cache,
            value_cache=v_cache,
            active_mask=self.active_mask,
            n_kv_head=num_kv_heads,
            head_size=head_size,
            dynamic_loop_unroll_factor=self.dynamic_loop_unrolling_size,
            mixed_precision=mixed_precision,
            skip_active=self.skip_active,
        )
        input_kwargs.update(self.prefill_ctx_inputs.as_dict(prefix="prefill_"))
        input_kwargs.update(self.decode_ctx_inputs.as_dict(prefix="decode_"))
        self.kernel_func = flash_attn_varlen_nkifunc
        return input_kwargs

    def _run_nki_xla(self, **input_kwargs):
        # execute using xla

        import torch_xla.core.xla_model as xm

        device = xm.xla_device()
        kernel_kwargs = {
            arg_name: (arg.to(device) if isinstance(arg, torch.Tensor) else arg)
            for arg_name, arg in input_kwargs.items()
        }
        output_nki = self.kernel_func(**kernel_kwargs)

        # - o: shape (bs, n_heads, seq_q, d) -> (bs, seq_q, n_heads, d)
        output_nki = output_nki.cpu()
        output_nki = output_nki.permute(0, 2, 1, 3)
        output_nki = output_nki[0, : self.num_actual_tokens, :, :]
        return output_nki

    def _run_nki_numpy(self, **input_kwargs):
        output_nki = self.kernel_func(**input_kwargs)

        # - o: shape (bs, n_heads, seq_q, d) -> (bs, seq_q, n_heads, d)
        output_nki = output_nki.transpose(0, 2, 1, 3)
        output_nki = output_nki[0, : self.num_actual_tokens, :, :]
        return output_nki

    def __call__(
        self,
        query,
        k_active,
        v_active,
        k_cache,
        v_cache,
        mixed_precision,
    ):
        _, num_kv_heads, _, head_size = k_cache.shape
        input_kwargs = self._prepare_kernel(
            query,
            k_active,
            v_active,
            k_cache,
            v_cache,
            head_size,
            num_kv_heads,
            mixed_precision,
        )
        compiler_flags = [
            "-O1",
            "--lnc=1",
            "--retry_failed_compilation",
        ]
        compiler_flags_str = " ".join(compiler_flags)
        os.environ["NEURON_CC_FLAGS"] = compiler_flags_str
        if self.exec_mode == "xla":
            return self._run_nki_xla(**input_kwargs)
        else:
            kernel_kwargs = convert_torch_tensor_to_numpy(
                input_kwargs, use_bf16=self.numpy_kernel_use_bf16
            )
            result = self._run_nki_numpy(**kernel_kwargs)
            if self.numpy_kernel_use_bf16 and result.dtype == bfloat16:
                result = result.astype(np.float32)
            return torch.tensor(result).to(query.dtype)
