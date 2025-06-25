import pytest
import torch
import torch.nn.functional as F

from kernel_runner import NKIFlashPagedAttentionRunner
from test_utils import (
    ref_context_attention,
    sample_input_sizes,
    sample_input_tensors,
    ceil_div,
    pad_to_next_power_of_2,
    assign_neuron_cores,
)


def _run_ref_version(
    query,
    key,
    value,
    query_lens,
    context_lens,
    head_size,
    num_queries_per_kv,
    return_buffer,
):
    seq_lens = query_lens + context_lens
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


def _run_test(
    query_lens,
    context_lens,
    max_model_len,
    num_heads,
    num_kv_heads,
    head_size,
    block_size,
    large_q_tile_size,
    large_kv_tile_size,
    pad_to_power2_bucket_size,
    mixed_precision,
    dynamic_loop_unroll,
    prefill_kernel_cache_kv=False,
):
    dtype = torch.bfloat16 if mixed_precision else torch.float32
    max_block_per_request = ceil_div(max_model_len, block_size)
    num_queries_per_kv = num_heads // num_kv_heads
    query, k_active, v_active, k_cache, v_cache, block_tables, key, value = (
        sample_input_tensors(
            query_lens=query_lens,
            context_lens=context_lens,
            max_block_per_request=max_block_per_request,
            block_size=block_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            dtype=dtype,
        )
    )

    output_ref = _run_ref_version(
        query,
        key,
        value,
        query_lens,
        context_lens,
        head_size,
        num_queries_per_kv,
        return_buffer=True,
    )

    # prepare plan
    nki_kernel_runner = NKIFlashPagedAttentionRunner(
        query_lens=query_lens,
        context_lens=context_lens,
        large_q_tile_size=large_q_tile_size,
        large_kv_tile_size=large_kv_tile_size,
        block_size=block_size,
        prefill_kernel_cache_kv=prefill_kernel_cache_kv,
        pad_to_power2_bucket_size=pad_to_power2_bucket_size,
        dynamic_loop_unrolling_size=dynamic_loop_unroll,
    )
    nki_kernel_runner.set_block_tables(
        block_tables=block_tables,
        max_kv_cache_size=k_cache.shape[0],
    )

    # pad and change to kernel layout
    num_active_token_after_padding = (
        nki_kernel_runner.get_num_active_tokens_after_padding()
    )
    pad_dims = (
        0,
        0,
        0,
        0,
        0,
        num_active_token_after_padding - query.shape[0],
    )
    query = F.pad(query, pad_dims, "constant", 0)
    k_active = F.pad(k_active, pad_dims, "constant", 0)
    v_active = F.pad(v_active, pad_dims, "constant", 0)
    # permute QKV tensors
    # query: (1, n_heads, seq_q, d)
    query = query.unsqueeze(0).permute(0, 2, 1, 3).contiguous()
    # key:   (1, n_kv_heads, d, seq_k)
    k_active = k_active.unsqueeze(0).permute(0, 2, 3, 1).contiguous()
    # value: (1, n_kv_heads, seq_v, d)
    v_active = v_active.unsqueeze(0).permute(0, 2, 1, 3).contiguous()
    k_cache = k_cache.permute(0, 2, 1, 3).contiguous()
    v_cache = v_cache.permute(0, 2, 1, 3).contiguous()

    # execute kernel
    output_nki = nki_kernel_runner(
        query=query,
        k_cache=k_cache,
        v_cache=v_cache,
        k_active=k_active,
        v_active=v_active,
        mixed_precision=mixed_precision,
    )

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
    "dynamic_loop_unroll,num_heads,num_kv_heads,head_size",
    [
        # static loop version
        (0, 4, 2, 16),
        (0, 32, 4, 64),
        (0, 2, 1, 128),
        (0, 8, 8, 32),
        # dynamic loop version, only 1 kv_head supported dut to no spmd launch
        (1, 4, 1, 128),
        (2, 4, 1, 128),
        (4, 4, 1, 128),
        (8, 4, 1, 128),
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
@pytest.mark.parametrize("pad_to_power2_bucket_size", [True, False])
@pytest.mark.parametrize("prefill_kernel_cache_kv", [True, False])
@torch.inference_mode()
def test_flash_paged_attn_prefill(
    monkeypatch: pytest.MonkeyPatch,
    worker_id: int,
    prefill_batch_size: int,
    decode_batch_size: int,
    num_heads: int,
    num_kv_heads: int,
    head_size: int,
    block_size: int,
    large_q_tile_size: int,
    large_kv_tile_size: int,
    dynamic_loop_unroll: int,
    mixed_precision: bool,
    prefill_kernel_cache_kv: bool,
    pad_to_power2_bucket_size: bool,
) -> None:
    assert large_kv_tile_size % block_size == 0

    torch.manual_seed(0)
    torch.set_printoptions(sci_mode=False)

    min_ctx_len = 32
    max_ctx_len = 8192
    min_query_len = 16
    max_query_len = 512
    query_lens, context_lens = sample_input_sizes(
        prefill_batch_size=prefill_batch_size,
        decode_batch_size=decode_batch_size,
        min_query_len=min_query_len,
        max_query_len=max_query_len,
        min_ctx_len=min_ctx_len,
        max_ctx_len=max_ctx_len,
    )
    max_model_len = max(max_query_len, max_ctx_len) * 4

    core_ids = assign_neuron_cores(worker_id)
    with monkeypatch.context() as m:
        m.setenv("NEURON_RT_VISIBLE_CORES", core_ids)
        _run_test(
            query_lens=query_lens,
            context_lens=context_lens,
            max_model_len=max_model_len,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            block_size=block_size,
            large_q_tile_size=large_q_tile_size,
            large_kv_tile_size=large_kv_tile_size,
            pad_to_power2_bucket_size=pad_to_power2_bucket_size,
            dynamic_loop_unroll=dynamic_loop_unroll,
            mixed_precision=mixed_precision,
            prefill_kernel_cache_kv=prefill_kernel_cache_kv,
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
    "dynamic_loop_unroll,num_heads,num_kv_heads,head_size",
    [
        # static loop version
        (0, 4, 2, 16),
        (0, 32, 4, 64),
        (0, 2, 1, 128),
        (0, 8, 8, 32),
        # dynamic loop version, only 1 kv_head supported dut to no spmd launch
        (1, 4, 1, 128),
        (2, 4, 1, 128),
        (4, 4, 1, 128),
        (8, 4, 1, 128),
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
@pytest.mark.parametrize("pad_to_power2_bucket_size", [True, False])
@torch.inference_mode()
def test_flash_paged_attn_decode(
    monkeypatch: pytest.MonkeyPatch,
    worker_id: int,
    decode_batch_size: int,
    num_heads: int,
    num_kv_heads: int,
    head_size: int,
    block_size: int,
    large_kv_tile_size: int,
    mixed_precision: bool,
    pad_to_power2_bucket_size: bool,
    dynamic_loop_unroll: int,
) -> None:
    assert large_kv_tile_size % block_size == 0

    torch.manual_seed(0)
    torch.set_printoptions(sci_mode=False)

    min_ctx_len = 32
    max_ctx_len = 8192
    query_lens, context_lens = sample_input_sizes(
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
            context_lens=context_lens,
            max_model_len=max_model_len,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            block_size=block_size,
            large_q_tile_size=large_q_tile_size,
            large_kv_tile_size=large_kv_tile_size,
            pad_to_power2_bucket_size=pad_to_power2_bucket_size,
            dynamic_loop_unroll=dynamic_loop_unroll,
            mixed_precision=mixed_precision,
        )
