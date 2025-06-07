import os

import numpy as np
import torch
import torch.nn.functional as F

from kernel_impl import flash_attn_varlen_blocksparse_nkifunc

from blocksparse_planner import FlashAttentionPlanner
from test_utils import (
    BlockDiagonalCausalFromBottomRightMask,
    get_active_block_tables,
    ceil_div,
    is_power_of_2,
    pad_to_multiple,
    pad_to_next_power_of_2,
)

B_P_SIZE = 128
B_FMAX_SIZE = 512


def _decide_execution_mode(exec_mode):
    _supported_exec_mode = [
        "xla",
        "baremetal",
        # "simulation",  # simulation mode may not work properly
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


class NKIFlashPagedAttentionRunner:
    def __init__(
        self,
        query_lens,
        context_lens,
        large_q_tile_size,
        large_kv_tile_size,
        block_size,
        prefill_kernel_cache_kv,
        pad_to_power2_bucket_size,
        dynamic_loop_unrolling_size,
        exec_mode=None,
    ):
        assert large_kv_tile_size >= B_P_SIZE
        self.query_lens = query_lens
        self.context_lens = context_lens
        self.large_q_tile_size = large_q_tile_size
        self.large_kv_tile_size = large_kv_tile_size
        self.block_size = block_size
        self.prefill_kernel_cache_kv = prefill_kernel_cache_kv
        self.dynamic_loop_unrolling_size = dynamic_loop_unrolling_size
        self.pad_to_power2_bucket_size = pad_to_power2_bucket_size
        self.exec_mode = _decide_execution_mode(exec_mode)
        self.is_decode_kernel = torch.all(query_lens <= 1).item()
        self.numpy_kernel_use_bf16 = True  # use ml_dtypes.bfloat16 for baremetal mode
        self.save_artifact = False
        self.num_actual_tokens = None
        assert is_power_of_2(self.large_q_tile_size)
        assert is_power_of_2(self.large_kv_tile_size)
        assert (
            not self.is_decode_kernel or large_q_tile_size == 1
        ), f"{large_q_tile_size=} must be 1 for decode scenario"
        self._preprocess()

    def _preprocess(self):
        self.num_actual_tokens = self.query_lens.sum().item()
        self.seq_lens = self.query_lens + self.context_lens
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
        self.active_masks = self._build_active_token_masks()
        self.context_token_plan = self._build_kernel_plan(
            self.pad_to_power2_bucket_size
        )
        self._build_inputs_from_plan()

    def get_num_active_tokens_after_padding(self):
        return self.num_active_tokens_after_padding

    def set_block_tables(self, block_tables, max_kv_cache_size):
        # transform block table
        num_active_blocks = ceil_div(self.context_lens, self.block_size).sum().item()
        num_active_blocks = pad_to_multiple(
            num_active_blocks, self.large_kv_tile_size // self.block_size
        )
        active_block_table = get_active_block_tables(
            block_tables,
            self.query_lens,
            self.seq_lens,
            self.block_size,
            num_active_blocks,
        )
        dma_skip_value = max_kv_cache_size * 2
        tile_block_tables = self.context_token_plan.build_tile_block_tables(
            active_block_table,
            skip_value=dma_skip_value if not self.is_decode_kernel else 0,
            dynamic_loop_unrolling=self.dynamic_loop_unrolling_size,
        )
        self.tile_block_tables = torch.tensor(tile_block_tables)
        assert self.tile_block_tables.dtype == torch.int32

    def _build_active_token_masks(self):
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

    def _build_kernel_plan(self, pad_to_power2_bucket_size):
        planner = FlashAttentionPlanner(
            self.query_lens.int().numpy(),
            self.context_lens.int().numpy(),
            tile_size_q=self.large_q_tile_size,
            tile_size_kv=self.large_kv_tile_size,
            block_size=self.block_size,
            traverse_in_column_order=self.prefill_kernel_cache_kv,
            kv_dma_skipping=True,
        )
        ctx_token_plan = planner.generate_plan()
        num_real_tiles = ctx_token_plan.num_tiles
        if pad_to_power2_bucket_size:
            num_tile_after_padding = pad_to_next_power_of_2(num_real_tiles)
        else:
            num_tile_after_padding = num_real_tiles
        if self.dynamic_loop_unrolling_size > 1:
            # make sure unrolling is correct if used in dynamic loop
            num_tile_after_padding = pad_to_multiple(
                num_tile_after_padding,
                self.dynamic_loop_unrolling_size,
            )
        if num_tile_after_padding != num_real_tiles:
            ctx_token_plan = ctx_token_plan.pad_plan(num_tile_after_padding)
        print(f"{num_real_tiles=}")
        print(f"{num_tile_after_padding=}")
        return ctx_token_plan

    def _prepare_decode_additional_inputs(self):
        q_update_pred, last_tile_indices = (
            self.context_token_plan.build_decode_tile_update_indices(
                padded_seqlen_q=self.num_active_tokens_after_padding
            )
        )
        # reshape q_update_pred for loop unrolling
        if q_update_pred is not None:
            num_tiles = self.ctx_token_plan.num_tiles
            if self.dynamic_loop_unrolling_size > 1:
                assert num_tiles % self.dynamic_loop_unrolling_size == 0
                num_loop_steps = num_tiles // self.dynamic_loop_unrolling_size
                q_update_pred = q_update_pred.reshape(
                    num_loop_steps, self.dynamic_loop_unrolling_size
                )
            else:
                q_update_pred = q_update_pred.reshape(1, num_tiles)
        tile_size = min(self.num_active_tokens_after_padding, B_P_SIZE)
        # reshape last_tile_indices for fast vector dge loading
        last_tile_indices = last_tile_indices.reshape(
            self.num_active_tokens_after_padding // tile_size, tile_size
        ).transpose(1, 0)
        if q_update_pred is not None:
            q_update_pred = torch.tensor(q_update_pred, dtype=torch.uint8)
        last_tile_indices = torch.tensor(last_tile_indices, dtype=torch.int32)
        return q_update_pred, last_tile_indices

    def _build_inputs_from_plan(self):
        ctx_plan = self.context_token_plan
        self.tile_q_indices = torch.tensor(
            ctx_plan.build_tile_q_indices(
                skip_value=(
                    0
                    if self.is_decode_kernel
                    else self.num_active_tokens_after_padding * 10
                )
            )
        )
        assert self.tile_q_indices.dtype == torch.int32
        tile_masks = ctx_plan.build_tile_masks(
            decode_kq_matmul=self.is_decode_kernel,
        )
        self.tile_masks = torch.tensor(tile_masks).to(torch.uint8)

        if self.is_decode_kernel:
            q_update_pred, last_tile_indices = self._prepare_decode_additional_inputs()
        else:
            q_update_pred, last_tile_indices = None, None
        self.last_tile_indices = last_tile_indices
        self.q_update_pred = q_update_pred

        # dyanmic loop num iteration
        if self.dynamic_loop_unrolling_size:
            num_dynamic_loop_steps = torch.empty((1, 1), dtype=torch.int32)
            num_dynamic_loop_steps[0] = ceil_div(
                ctx_plan.num_real_tiles, self.dynamic_loop_unrolling_size
            )
            print(f"{num_dynamic_loop_steps=}")
        else:
            num_dynamic_loop_steps = None
        self.num_dynamic_loop_steps = num_dynamic_loop_steps

    def _prepare_kernel_inputs(
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
        assert query.shape == (1, num_heads, num_active_token, head_size)
        assert k_active.shape == (1, num_kv_heads, head_size, num_active_token)
        assert v_active.shape == (1, num_kv_heads, num_active_token, head_size)
        input_args = (
            query,
            k_active,
            v_active,
            k_cache,
            v_cache,
            self.tile_q_indices,
            self.tile_block_tables,
            self.tile_masks,
            self.active_masks,
            self.num_dynamic_loop_steps,
            self.last_tile_indices,
            self.q_update_pred,
        )
        input_kwargs = dict(
            n_kv_head=num_kv_heads,
            head_size=head_size,
            dynamic_loop_unroll_factor=self.dynamic_loop_unrolling_size,
            mixed_precision=mixed_precision,
            decode_mode=self.is_decode_kernel,
        )
        return input_args, input_kwargs

    def run_nki_xla(self, *input_args, **input_kwargs):
        # execute using xla
        compiler_flags = [
            "-O1",
            "--lnc=1",
            "--retry_failed_compilation",
        ]
        compiler_flags_str = " ".join(compiler_flags)
        os.environ["NEURON_CC_FLAGS"] = compiler_flags_str

        import torch_xla.core.xla_model as xm

        device = xm.xla_device()
        input_args = tuple(
            [arg.to(device=device) if arg is not None else None for arg in input_args]
        )
        output_nki = flash_attn_varlen_blocksparse_nkifunc(*input_args, **input_kwargs)

        # - o: shape (bs, n_heads, seq_q, d) -> (bs, seq_q, n_heads, d)
        output_nki = output_nki.cpu().permute(0, 2, 1, 3)
        output_nki = output_nki[0, : self.num_actual_tokens, :, :]
        return output_nki

    def run_nki_numpy(self, *input_args, **input_kwargs):
        output_nki = flash_attn_varlen_blocksparse_nkifunc(*input_args, **input_kwargs)

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
        input_args, input_kwargs = self._prepare_kernel_inputs(
            query,
            k_active,
            v_active,
            k_cache,
            v_cache,
            head_size,
            num_kv_heads,
            mixed_precision,
        )
        if self.exec_mode == "xla":
            return self.run_nki_xla(*input_args, **input_kwargs)
        else:
            if self.numpy_kernel_use_bf16:
                from ml_dtypes import bfloat16
            new_input_args = []
            for arg in input_args:
                if isinstance(arg, torch.Tensor):
                    if arg.dtype == torch.bfloat16:
                        if self.numpy_kernel_use_bf16:
                            arg = arg.float().numpy().astype(bfloat16)
                        else:
                            arg = arg.half().numpy()
                    else:
                        arg = arg.numpy()
                new_input_args.append(arg)
            result = self.run_nki_numpy(
                *new_input_args,
                **input_kwargs,
            )
            if self.numpy_kernel_use_bf16 and result.dtype == bfloat16:
                result = result.astype(np.float32)
            return torch.tensor(result).to(query.dtype)
