import os

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from flash_attn_blocksparse import flash_attn_varlen_blocksparse_nkifunc

from blocksparse_planner import SequenceAlignedScheduler as BlockSparseScheduler
from test_utils import (
    BlockDiagonalCausalFromBottomRightMask,
    ref_context_attention,
    sample_input_sizes,
    sample_inputs,
    get_active_block_tables,
    ceil_div,
    pad_to_multiple,
    pad_to_next_power_of_2,
    assign_neuron_cores,
)


def _run_ref_version(
    query,
    key,
    value,
    query_lens,
    seq_lens,
    head_size,
    num_queries_per_kv,
    return_buffer,
):
    num_actual_tokens = sum(query_lens)
    max_num_queries = pad_to_next_power_of_2(num_actual_tokens)
    output_ref, *_ = ref_context_attention(
        query,
        key,
        value,
        query_lens,
        seq_lens,
        head_size,
        num_queries_per_kv,
        return_buffer=return_buffer,
    )
    output_ref_padded = F.pad(
        output_ref,
        (0, 0, 0, 0, 0, 0, 0, max_num_queries - output_ref.shape[0]),
        "constant",
        0,
    )
    output_ref = output_ref_padded.transpose(0, 1)[0, :num_actual_tokens, :, :]
    return output_ref


class NKIBlockSparseRunner:
    _supported_exec_mode = ["xla", "baremetal", "simulation"]

    def __init__(
        self,
        large_q_tile_size,
        large_kv_tile_size,
        column_order,
        pad_power2_num_tiles,
        decode_mode,
        exec_mode=None,
        numpy_kernel_use_bf16=True,
        numpy_kernel_save_artifact=False,
    ):
        if exec_mode is None:
            exec_mode = os.getenv("TEST_EXEC_MODE", "undefined")
        if exec_mode.lower() not in self._supported_exec_mode:
            self.exec_mode = self._supported_exec_mode[0]
            if exec_mode != "undefined":
                print(f"Execution mode {exec_mode} is not supported.")
            print(f"Default to {self.exec_mode} mode for execution")
        else:
            self.exec_mode = exec_mode.lower()
            print(f"Use {self.exec_mode} mode for execution")
        B_P_SIZE = 128
        assert large_kv_tile_size >= B_P_SIZE
        self.large_q_tile_size = large_q_tile_size
        self.large_kv_tile_size = large_kv_tile_size
        self.column_order = column_order
        self.pad_power2_num_tiles = pad_power2_num_tiles
        self.decode_mode = decode_mode
        self.numpy_kernel_use_bf16 = numpy_kernel_use_bf16
        self.save_artifact = numpy_kernel_save_artifact
        self.num_actual_tokens = None

    def run(
        self,
        query,
        k_active,
        v_active,
        k_cache,
        v_cache,
        block_table,
        query_lens,
        seq_lens,
        block_size,
        head_size,
        num_heads,
        num_queries_per_kv,
        dynamic_loop_unroll_factor,
        mixed_precision,
    ):
        input_args, input_kwargs = self.build_torch_inputs(
            query,
            k_active,
            v_active,
            k_cache,
            v_cache,
            block_table,
            query_lens,
            seq_lens,
            block_size,
            head_size,
            num_heads,
            num_queries_per_kv,
            dynamic_loop_unroll_factor,
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

    def build_torch_inputs(
        self,
        query,
        k_active,
        v_active,
        k_cache,
        v_cache,
        block_table,
        query_lens,
        seq_lens,
        block_size,
        head_size,
        num_heads,
        num_queries_per_kv,
        dynamic_loop_unroll_factor,
        mixed_precision,
    ):
        # calculate input shapes
        self.num_actual_tokens = sum(query_lens)
        max_num_queries = pad_to_next_power_of_2(self.num_actual_tokens)
        context_lens = torch.tensor(seq_lens) - torch.tensor(query_lens)
        num_active_blocks = ceil_div(context_lens, block_size).sum().item()
        num_active_blocks = pad_to_multiple(
            num_active_blocks, self.large_kv_tile_size // block_size
        )
        context_kv_len = num_active_blocks * block_size
        assert (
            context_kv_len % self.large_kv_tile_size == 0
        ), f"invalid context_kv_len={context_kv_len}"

        # pad QKV tensors
        pad_dims = (
            0,
            0,
            0,
            0,
            0,
            max_num_queries - query.shape[0],
        )
        query = F.pad(query, pad_dims, "constant", 0)
        k = F.pad(k_active, pad_dims, "constant", 0)
        v = F.pad(v_active, pad_dims, "constant", 0)

        # permute QKV tensors
        # XXX: changing q layout
        # previously, query: (1, n_heads, d, seq_q)
        # now, query: (1, n_heads, seq_q, d)
        query = query.unsqueeze(0).permute(0, 2, 1, 3).contiguous()
        # key:   (1, n_kv_heads, d, seq_k)
        k = k.unsqueeze(0).permute(0, 2, 3, 1).contiguous()
        # value: (1, n_kv_heads, seq_v, d)
        v = v.unsqueeze(0).permute(0, 2, 1, 3).contiguous()
        k_cache = k_cache.permute(0, 2, 1, 3).contiguous()
        v_cache = v_cache.permute(0, 2, 1, 3).contiguous()
        # transform block table
        active_block_table = get_active_block_tables(
            block_table,
            torch.tensor(query_lens),
            torch.tensor(seq_lens),
            block_size,
            num_active_blocks,
        )

        # Build attention masks
        _, active_mask = BlockDiagonalCausalFromBottomRightMask.from_seqlens(
            query_lens, seq_lens, block_size=block_size
        )
        active_mask = F.pad(
            active_mask,
            (
                0,
                max_num_queries - active_mask.shape[1],
                0,
                max_num_queries - active_mask.shape[0],
            ),
            "constant",
            0,
        ).to(torch.uint8)

        pa_scheduler = BlockSparseScheduler(
            np.array(query_lens, dtype=np.int32),
            context_lens.int().numpy(),
            tile_size_q=self.large_q_tile_size,
            tile_size_kv=self.large_kv_tile_size,
            block_size=block_size,
            column_order=self.column_order,
            kv_dma_skipping=True,
        )
        ctx_token_plan = pa_scheduler.generate_plan()
        num_real_tiles = ctx_token_plan.num_tiles
        if self.pad_power2_num_tiles:
            num_tiles_padded = pad_to_next_power_of_2(num_real_tiles)
        else:
            num_tiles_padded = num_real_tiles
        # make sure unrolling is correct
        if dynamic_loop_unroll_factor:
            num_tiles_padded = pad_to_multiple(
                num_tiles_padded, dynamic_loop_unroll_factor
            )
        if num_tiles_padded != num_real_tiles:
            ctx_token_plan = ctx_token_plan.pad_plan(num_tiles_padded)

        num_kv_heads = num_heads // num_queries_per_kv
        tile_block_tables = ctx_token_plan.build_tile_block_tables(
            active_block_table,
            skip_value=k_cache.shape[0] * 2 if not self.decode_mode else 0,
            dynamic_loop_unrolling=dynamic_loop_unroll_factor,
        )
        tile_masks = ctx_token_plan.build_tile_masks(decode_kq_matmul=self.decode_mode)
        if self.decode_mode:
            q_update_pred, last_tile_indices = (
                ctx_token_plan.build_decode_tile_update_indices(
                    padded_seqlen_q=max_num_queries
                )
            )
            # reshape q_update_pred for loop unrolling
            if q_update_pred is not None:
                num_tiles = ctx_token_plan.num_tiles
                if dynamic_loop_unroll_factor:
                    assert num_tiles % dynamic_loop_unroll_factor == 0
                    num_loop_steps = num_tiles // dynamic_loop_unroll_factor
                    q_update_pred = q_update_pred.reshape(
                        num_loop_steps, dynamic_loop_unroll_factor
                    )
                else:
                    q_update_pred = q_update_pred.reshape(1, num_tiles)
            # reshape last_tile_indices for vector dge
            B_P_SIZE = 128
            tile_size = min(max_num_queries, B_P_SIZE)
            last_tile_indices = last_tile_indices.reshape(
                max_num_queries // tile_size, tile_size
            ).transpose(1, 0)
        else:
            q_update_pred, last_tile_indices = None, None

        tile_q_indices = torch.tensor(
            ctx_token_plan.build_tile_q_indices(
                skip_value=0 if self.decode_mode else max_num_queries * 10
            )
        )
        tile_block_tables = torch.tensor(tile_block_tables)
        tile_masks = torch.tensor(tile_masks).to(torch.uint8)
        assert tile_q_indices.dtype == torch.int32
        assert tile_block_tables.dtype == torch.int32
        assert tile_masks.dtype == torch.uint8
        if self.decode_mode:
            if q_update_pred is not None:
                q_update_pred = torch.tensor(q_update_pred, dtype=torch.uint8)
            last_tile_indices = torch.tensor(last_tile_indices, dtype=torch.int32)
        print(f"{num_real_tiles=}")
        print(f"{num_tiles_padded=}")
        if dynamic_loop_unroll_factor:
            num_dynamic_loop_steps = torch.empty((1, 1), dtype=torch.int32)
            num_dynamic_loop_steps[0] = ceil_div(
                num_real_tiles, dynamic_loop_unroll_factor
            )
            print(f"{num_dynamic_loop_steps=}")
        else:
            num_dynamic_loop_steps = None

        input_args = (
            query,
            k,
            v,
            k_cache,
            v_cache,
            tile_q_indices,
            tile_block_tables,
            tile_masks,
            active_mask,
            num_dynamic_loop_steps,
            last_tile_indices,
            q_update_pred,
        )
        input_kwargs = dict(
            n_kv_head=num_kv_heads,
            head_size=head_size,
            dynamic_loop_unroll_factor=dynamic_loop_unroll_factor,
            mixed_precision=mixed_precision,
            decode_mode=self.decode_mode,
        )
        return input_args, input_kwargs

    def run_nki_xla(self, *input_args, **input_kwargs):
        # execute using xla
        compiler_flags = [
            "-O1",
            "--lnc=1",
            "--retry_failed_compilation",
            # "--internal-compiler-debug-mode=all",
            # "--tensorizer-options='--print-stats --dump-after=All'",
            # "--enable-internal-birsim-after-all",
            # "--enable-internal-data-race-checker",
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


def _run_test(
    query_lens,
    ctx_lens,
    max_model_len,
    num_heads,
    num_queries_per_kv,
    head_size,
    block_size,
    large_q_tile_size,
    large_kv_tile_size,
    pad_power2_num_tiles,
    mixed_precision,
    column_order,
    dynamic_unroll_factor,
    decode_mode,
    nki_block_sparse_runner_cls=NKIBlockSparseRunner,
    save_dir=None,
):
    dtype = torch.bfloat16 if mixed_precision else torch.float32

    max_block_per_request = ceil_div(max_model_len, block_size)
    num_kv_heads = num_heads // num_queries_per_kv
    (
        query,
        k_active,
        v_active,
        k_cache,
        v_cache,
        block_table,
        key,
        value,
        query_lens,
        seq_lens,
    ) = sample_inputs(
        query_lens=query_lens,
        ctx_lens=ctx_lens,
        max_block_per_request=max_block_per_request,
        block_size=block_size,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        dtype=dtype,
    )
    nki_runner = nki_block_sparse_runner_cls(
        large_q_tile_size=large_q_tile_size,
        large_kv_tile_size=large_kv_tile_size,
        column_order=column_order,
        pad_power2_num_tiles=pad_power2_num_tiles,
        decode_mode=decode_mode,
        numpy_kernel_save_artifact=save_dir,
    )
    output_nki = nki_runner.run(
        query=query,
        k_active=k_active,
        v_active=v_active,
        k_cache=k_cache,
        v_cache=v_cache,
        block_table=block_table,
        query_lens=query_lens,
        seq_lens=seq_lens,
        block_size=block_size,
        head_size=head_size,
        num_heads=num_heads,
        num_queries_per_kv=num_queries_per_kv,
        dynamic_loop_unroll_factor=dynamic_unroll_factor,
        mixed_precision=mixed_precision,
    )

    output_ref = _run_ref_version(
        query,
        key,
        value,
        query_lens,
        seq_lens,
        head_size,
        num_queries_per_kv,
        return_buffer=True,
    )

    print(output_nki.shape, output_ref.shape)
    # print(output_nki[:16, 0, :8])
    # print(output_ref[:16, 0, :8])
    print(output_nki[:, 0, :1].view(-1))
    print(output_ref[:, 0, :1].view(-1))
    print("Valid", (~output_nki[:, 0, :1].isnan()).sum())
    print("NaN", output_nki[:, 0, :1].isnan().sum())

    torch.testing.assert_close(output_nki, output_ref, atol=1e-2, rtol=0)


@pytest.mark.parametrize(
    "large_q_tile_size,large_kv_tile_size,block_size",
    [
        (32, 1024, 32),  # 16 blocks
        (64, 2048, 64),  # 32 blocks
        (128, 2048, 32),  # 64 blocks
        (128, 2048, 16),  # 128 blocks
        (128, 8192, 32),  # 256 blocks
        (256, 2048, 256),  # 8 blocks
        (256, 4096, 32),  # 128 blocks
        (256, 1024, 4),  # 256 blocks
    ],
)
@pytest.mark.parametrize(
    "dynamic_unroll_factor,num_heads,num_queries_per_kv,head_size",
    [
        # static loop version
        (0, 4, 2, 16),
        (0, 32, 8, 64),
        (0, 2, 2, 128),
        (0, 8, 1, 32),
        # dynamic loop version, only 1 kv_head supported dut to no spmd launch
        (1, 4, 4, 128),
        (2, 4, 4, 128),
        (4, 4, 4, 128),
        (8, 4, 4, 128),
    ],
)
@pytest.mark.parametrize(
    "prefill_batch_size,decode_batch_size",
    [
        (4, 12),
        (1, 33),
    ],
)
@pytest.mark.parametrize("mixed_precision", [True, False])
@pytest.mark.parametrize("pad_power2_num_tiles", [True, False])
@pytest.mark.parametrize("column_order", [True, False])
@torch.inference_mode()
def test_blocksparse_flash_paged_attention(
    monkeypatch: pytest.MonkeyPatch,
    worker_id: int,
    prefill_batch_size: int,
    decode_batch_size: int,
    num_heads: int,
    num_queries_per_kv: int,
    head_size: int,
    block_size: int,
    large_q_tile_size: int,
    large_kv_tile_size: int,
    dynamic_unroll_factor: int,
    mixed_precision: bool,
    column_order: bool,
    pad_power2_num_tiles: bool,
) -> None:
    assert large_kv_tile_size % block_size == 0

    torch.manual_seed(0)
    torch.set_printoptions(sci_mode=False)

    min_ctx_len = 32
    max_ctx_len = 8192
    min_query_len = 16
    max_query_len = 512
    query_lens, ctx_lens = sample_input_sizes(
        prefill_batch_size=prefill_batch_size,
        decode_batch_size=decode_batch_size,
        min_query_len=min_query_len,
        max_query_len=max_query_len,
        min_ctx_len=min_ctx_len,
        max_ctx_len=max_ctx_len,
    )
    print(f"{query_lens=}")
    print(f"{ctx_lens=}")
    max_model_len = max(max_query_len, max_ctx_len) * 4

    core_ids = assign_neuron_cores(worker_id)
    with monkeypatch.context() as m:
        m.setenv("NEURON_RT_VISIBLE_CORES", core_ids)
        _run_test(
            query_lens=query_lens,
            ctx_lens=ctx_lens,
            max_model_len=max_model_len,
            num_heads=num_heads,
            num_queries_per_kv=num_queries_per_kv,
            head_size=head_size,
            block_size=block_size,
            large_q_tile_size=large_q_tile_size,
            large_kv_tile_size=large_kv_tile_size,
            pad_power2_num_tiles=pad_power2_num_tiles,
            dynamic_unroll_factor=dynamic_unroll_factor,
            mixed_precision=mixed_precision,
            column_order=column_order,
            decode_mode=False,
        )


@pytest.mark.parametrize(
    "large_kv_tile_size,block_size",
    [
        (512, 1),  # 512 blocks
        (1024, 2),  # 512 blocks
        (1024, 4),  # 256 blocks
        (1024, 8),  # 128 blocks
        (2048, 16),  # 128 blocks
        (2048, 256),  # 8 blocks
        (4096, 64),  # 64 blocks
        (8192, 32),  # 256 blocks
    ],
)
@pytest.mark.parametrize(
    "dynamic_unroll_factor,num_heads,num_queries_per_kv,head_size",
    [
        # static loop version
        (0, 4, 2, 16),
        (0, 32, 8, 64),
        (0, 2, 2, 128),
        (0, 8, 1, 32),
        # dynamic loop version, only 1 kv_head supported dut to no spmd launch
        (1, 4, 4, 128),
        (2, 4, 4, 128),
        (4, 4, 4, 128),
        (8, 4, 4, 128),
    ],
)
@pytest.mark.parametrize(
    "decode_batch_size",
    [
        12,
        33,
        53,
        65,
    ],
)
@pytest.mark.parametrize("mixed_precision", [True, False])
@pytest.mark.parametrize("pad_power2_num_tiles", [True, False])
@torch.inference_mode()
def test_decode_only(
    monkeypatch: pytest.MonkeyPatch,
    worker_id: int,
    decode_batch_size: int,
    num_heads: int,
    num_queries_per_kv: int,
    head_size: int,
    block_size: int,
    large_kv_tile_size: int,
    mixed_precision: bool,
    pad_power2_num_tiles: bool,
    dynamic_unroll_factor: int,
) -> None:
    assert large_kv_tile_size % block_size == 0

    torch.manual_seed(0)
    torch.set_printoptions(sci_mode=False)

    min_ctx_len = 32
    max_ctx_len = 8192
    query_lens, ctx_lens = sample_input_sizes(
        prefill_batch_size=0,
        decode_batch_size=decode_batch_size,
        min_query_len=0,
        max_query_len=0,
        min_ctx_len=min_ctx_len,
        max_ctx_len=max_ctx_len,
    )
    max_model_len = max_ctx_len * 4
    large_q_tile_size = 1

    core_ids = assign_neuron_cores(worker_id)
    with monkeypatch.context() as m:
        m.setenv("NEURON_RT_VISIBLE_CORES", core_ids)
        _run_test(
            query_lens=query_lens,
            ctx_lens=ctx_lens,
            max_model_len=max_model_len,
            num_heads=num_heads,
            num_queries_per_kv=num_queries_per_kv,
            head_size=head_size,
            block_size=block_size,
            large_q_tile_size=large_q_tile_size,
            large_kv_tile_size=large_kv_tile_size,
            pad_power2_num_tiles=pad_power2_num_tiles,
            dynamic_unroll_factor=dynamic_unroll_factor,
            mixed_precision=mixed_precision,
            column_order=False,
            decode_mode=True,
        )
