from typing import Optional
import os

import numpy as np
import torch
import torch.nn.functional as F


def ceil_div(a, b):
    assert b > 0, f"{b=}"
    return (a + b - 1) // b


def is_power_of_2(n):
    return n > 0 and (n & (n - 1) == 0)


def pad_to_multiple(a, b):
    return ceil_div(a, b) * b


def pad_to_next_power_of_2(a):
    return 2 ** int(a - 1).bit_length() if a > 0 else 0


def assign_neuron_cores(worker_id):
    num_cores = 32
    if worker_id == "master":
        worker_id = 0
    else:
        worker_id = int(worker_id[2:])
    num_workers = int(os.environ.get("PYTEST_XDIST_WORKER_COUNT", 1))
    assert worker_id < num_workers, f"{worker_id=} >= {num_workers=}"
    assert num_cores % num_workers == 0, f"{num_cores=} % {num_workers=} != 0"
    num_cores_per_worker = num_cores // num_workers
    core_start = worker_id * num_cores_per_worker
    if num_cores_per_worker > 1:
        core_end = core_start + num_cores_per_worker - 1
        core_ids = f"{core_start}-{core_end}"
    else:
        core_ids = f"{core_start}"
    return core_ids


class BlockDiagonalCausalFromBottomRightMask:

    @staticmethod
    def _from_seqlens(query_lens, seq_lens, block_size=None, skip_active=False):
        from torch import logical_and, logical_or

        contexted = block_size is None
        context_lens = torch.tensor(seq_lens) - torch.tensor(query_lens)
        n_queries = sum(query_lens)
        num_seqs = len(query_lens)
        if contexted:
            key_lens_blockaligned = seq_lens
        else:
            n_blocks_per_seq = (context_lens + block_size - 1) // block_size
            offset_per_seq = n_blocks_per_seq * block_size
            key_lens_blockaligned = offset_per_seq[:num_seqs].tolist()
        n_keys = sum(key_lens_blockaligned)

        a = torch.arange(n_queries).reshape(n_queries, 1).expand(n_queries, n_keys)
        b = torch.arange(n_keys).reshape(1, n_keys).expand(n_queries, n_keys)
        q_cumsum = torch.tensor([0] + query_lens).cumsum(dim=0)
        k_cumsum = torch.tensor([0] + key_lens_blockaligned).cumsum(dim=0)

        prior_mask = torch.zeros(n_queries, n_keys)
        new_masks: list[torch.Tensor] = []
        for seq_id in range(num_seqs):
            ri = q_cumsum[seq_id]
            ci = k_cumsum[seq_id]
            nr = query_lens[seq_id]

            if contexted and not skip_active:
                nc = seq_lens[seq_id]
                a_offset = ci + nc - ri - nr
                new_mask = (a + a_offset) >= b
            else:
                nc = context_lens[seq_id]
                a_offset = ci + nc - 1
                new_mask = a_offset >= b

            left_mask = b >= ci
            top_mask = a >= ri
            bottom_mask = a < (ri + nr)

            new_mask = logical_and(
                logical_and(logical_and(new_mask, left_mask), top_mask),
                bottom_mask,
            )
            prior_mask = logical_or(prior_mask, new_mask)
            new_masks = new_masks + [new_mask]
        return prior_mask

    @staticmethod
    def from_seqlens(query_lens, seq_lens, block_size=None):
        contexted = block_size is None
        if contexted:
            prior_mask = BlockDiagonalCausalFromBottomRightMask._from_seqlens(
                query_lens, seq_lens
            )
            active_mask = None
        else:
            prior_mask = BlockDiagonalCausalFromBottomRightMask._from_seqlens(
                query_lens, seq_lens, block_size
            )
            active_mask = BlockDiagonalCausalFromBottomRightMask._from_seqlens(
                query_lens, query_lens
            )
        return prior_mask, active_mask


def ref_softmax(
    x: torch.Tensor,
    dim: int,
    mixed_precision=False,
    return_max_reduce=False,
):
    print(f"{x.shape=}")
    if mixed_precision:
        x = x.float()
    max_value = torch.amax(x, dim=dim, keepdims=True)
    exp = torch.exp(x - max_value)
    sum_exp = torch.sum(exp, dim=dim, keepdims=True).transpose(0, 1)
    if return_max_reduce:
        return exp, sum_exp.contiguous(), max_value.transpose(0, 1).contiguous()
    return exp, sum_exp


def ref_masked_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    attn_mask: Optional[torch.Tensor] = None,
    return_buffer: Optional[bool] = False,
) -> torch.Tensor:
    kernel_dtype = query.dtype
    query = (query * scale).to(kernel_dtype)
    scaled_qk = torch.einsum("qhd,khd->hqk", query, key).float()
    if attn_mask is not None:
        # masked_score = scaled_qk + attn_mask.float()
        masked_score = torch.where(attn_mask, scaled_qk, -9984)
    if return_buffer:
        score, sum_exp, max_buffer = ref_softmax(
            masked_score,
            dim=-1,
            return_max_reduce=True,
        )
    else:
        score, sum_exp = ref_softmax(masked_score, dim=-1)
    o_buffer = torch.einsum("hqk,khd->qhd", score.to(kernel_dtype), value).contiguous()
    out = (o_buffer / sum_exp).to(kernel_dtype)
    if return_buffer:
        lse_buffer = torch.log(sum_exp) + max_buffer
        return (
            out,
            max_buffer,
            lse_buffer,
            o_buffer,
        )
    else:
        return (out,)


def ref_context_attention(
    query,
    key,
    value,
    query_lens,
    seq_lens,
    head_size,
    num_queries_per_kv,
    return_buffer=False,
):
    scale = float(1.0 / (head_size**0.5))
    if num_queries_per_kv > 1:
        # Handle MQA and GQA
        key = torch.repeat_interleave(key, num_queries_per_kv, dim=1)
        value = torch.repeat_interleave(value, num_queries_per_kv, dim=1)

    attn_mask, _ = BlockDiagonalCausalFromBottomRightMask.from_seqlens(
        query_lens, seq_lens
    )

    # convert binary mask to -inf values
    # attn_mask = torch.logical_not(attn_mask)
    # attn_mask = attn_mask.float() * -30000

    output, *buffers = ref_masked_attention(
        query,
        key,
        value,
        scale,
        attn_mask,
        return_buffer=return_buffer,
    )

    output = output.unsqueeze(1)
    if return_buffer:
        max_buffer, lse_buffer, o_buffer = buffers
        return (
            output,
            max_buffer,
            lse_buffer,
            o_buffer,
        )
    else:
        return output


def _sample_lengths(num, min_len, max_len):
    return torch.randint(min_len, max_len + 1, size=(num,))


def sample_input_sizes(
    prefill_batch_size,
    decode_batch_size,
    min_query_len,
    max_query_len,
    min_ctx_len,
    max_ctx_len,
):
    batch_size = prefill_batch_size + decode_batch_size
    assert batch_size > 0, f"Expecting at least one sequence"
    if prefill_batch_size == 0:
        query_lens = torch.ones(decode_batch_size, dtype=torch.long)
    else:
        prefill_query_lens = _sample_lengths(
            prefill_batch_size,
            min_query_len,
            max_query_len,
        )
        decode_query_lens = torch.ones(decode_batch_size, dtype=torch.long)
        query_lens = torch.cat([prefill_query_lens, decode_query_lens])
    if max_ctx_len == 0:
        ctx_lens = torch.zeros(batch_size, dtype=torch.long)
    else:
        ctx_lens = _sample_lengths(batch_size, min_ctx_len, max_ctx_len)
    return query_lens.tolist(), ctx_lens.tolist()


def sample_inputs(
    query_lens,
    ctx_lens,
    max_block_per_request,
    block_size,
    num_heads,
    num_kv_heads,
    head_size,
    dtype,
):
    batch_size = len(query_lens)
    cache_size = (batch_size * max_block_per_request) + 2
    seq_lens = [a + b for a, b in zip(query_lens, ctx_lens)]

    num_tokens = sum(query_lens)
    query = torch.empty(num_tokens, num_heads, head_size, dtype=dtype)
    query.uniform_(-1, 1)
    torch.empty(num_tokens, num_heads, head_size, dtype=dtype)

    kv = torch.empty(sum(seq_lens), 2, num_kv_heads, head_size, dtype=dtype)
    kv.uniform_(-1, 1)
    key, value = kv.unbind(dim=1)

    k_cache = torch.zeros(cache_size, block_size, num_kv_heads, head_size, dtype=dtype)
    v_cache = torch.zeros(cache_size, block_size, num_kv_heads, head_size, dtype=dtype)
    k = torch.zeros(sum(query_lens), num_kv_heads, head_size, dtype=dtype)
    v = torch.zeros(sum(query_lens), num_kv_heads, head_size, dtype=dtype)
    values = torch.arange(0, cache_size, dtype=torch.long)
    values = values[torch.randperm(cache_size)]
    block_table = values[: batch_size * max_block_per_request].view(
        batch_size, max_block_per_request
    )
    b_ctx_len = torch.tensor(ctx_lens, dtype=torch.long)
    b_start_loc = torch.cumsum(
        torch.tensor([0] + query_lens[:-1], dtype=torch.long),
        dim=0,
    )
    # copy kv to cache
    b_seq_start_loc = torch.cumsum(
        torch.tensor([0] + seq_lens[:-1], dtype=torch.long),
        dim=0,
    )
    for i in range(batch_size):
        for j in range(query_lens[i]):
            k[b_start_loc[i] + j].copy_(key[b_seq_start_loc[i] + b_ctx_len[i] + j])
            v[b_start_loc[i] + j].copy_(value[b_seq_start_loc[i] + b_ctx_len[i] + j])
        cur_ctx = 0
        block_id = 0
        while cur_ctx < b_ctx_len[i]:
            start_loc = b_seq_start_loc[i] + cur_ctx
            if cur_ctx + block_size > b_ctx_len[i]:
                end_loc = b_seq_start_loc[i] + b_ctx_len[i]
            else:
                end_loc = start_loc + block_size
            start_slot = block_table[i, block_id] * block_size
            end_slot = start_slot + end_loc - start_loc
            k_cache.view(-1, num_kv_heads, head_size)[start_slot:end_slot].copy_(
                key[start_loc:end_loc]
            )
            v_cache.view(-1, num_kv_heads, head_size)[start_slot:end_slot].copy_(
                value[start_loc:end_loc]
            )
            cur_ctx += block_size
            block_id += 1

    return query, k, v, k_cache, v_cache, block_table, key, value, query_lens, seq_lens


def get_active_block_tables(block_tables, query_lens, seq_lens, block_size, num_blocks):
    context_lens = seq_lens - query_lens
    blocks_per_seq = (context_lens + block_size - 1) // block_size
    num_seqs = len(seq_lens)
    active_blocks: list[int] = []
    for seq_id in range(num_seqs):
        active_blocks += block_tables[seq_id, : blocks_per_seq[seq_id]].tolist()
    return F.pad(
        torch.tensor(active_blocks, dtype=torch.int32),
        (0, num_blocks - len(active_blocks)),
        "constant",
        0,
    )


def save_numpy_inputs(artifact_dir, kwargs):
    tensor_map = []
    numpy_tensor_dir = "input_tensors"
    os.makedirs(os.path.join(artifact_dir, numpy_tensor_dir), exist_ok=True)
    for key in kwargs:
        value = kwargs[key]
        if isinstance(value, np.ndarray):
            relative_path = os.path.join(numpy_tensor_dir, f"input_{key}.npy")
            np.save(os.path.join(artifact_dir, relative_path), kwargs[key])
            tensor_map.append(f"{key} {relative_path}")
    return tensor_map
