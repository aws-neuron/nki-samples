"""
Copyright (C) 2025, Amazon.com. All Rights Reserved

Basic usage:

python rotary_nki_kernels.py

# Run comprehensive test suite
python rotary_nki_kernels.py \
    --batch-sizes 2 4 8 \
    --num-heads 16 32 \
    --seq-lengths 128 256 512 \
    --head-dims 64 128 \
    --rtol 1e-4 \
    --atol 1e-4

# Run minimal test for quick verification
python rotary_nki_kernels.py \
    --batch-sizes 2 \
    --num-heads 32 \
    --seq-lengths 128 \
    --head-dims 128
"""

import argparse
import json
import os
from datetime import datetime
from typing import Tuple

import neuronxcc.nki.language as nl
import torch
import torch_neuronx
from loguru import logger
from neuronxcc import nki
from torch.profiler import ProfilerActivity, profile, record_function
from torch_xla.core import xla_model as xm
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

os.environ["NEURON_FRAMEWORK_DEBUG"] = "1"
os.environ["NEURON_CC_FLAGS"] = " --disable-dge "


def parse_args():
    """
    Parse command line arguments for rotary embedding benchmark tests.

    Parameters
    ----------
    None

    Returns
    -------
    argparse.Namespace
        Parsed command line arguments containing:
        - batch_sizes : list of int
            Batch sizes to test
        - num_heads : list of int
            Number of attention heads to test
        - seq_lengths : list of int
            Sequence lengths to test
        - head_dims : list of int
            Head dimensions to test
        - rtol : float
            Relative tolerance for tensor comparison
        - atol : float
            Absolute tolerance for tensor comparison
    """

    parser = argparse.ArgumentParser(description="Test Rotary Embedding implementation")
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[2],
        help="List of batch sizes to test",
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        nargs="+",
        default=[32],
        help="List of number of heads to test",
    )
    parser.add_argument(
        "--seq-lengths",
        type=int,
        nargs="+",
        default=[64, 128, 256],
        help="List of sequence lengths to test",
    )
    parser.add_argument(
        "--head-dims",
        type=int,
        nargs="+",
        default=[128],
        help="List of head dimensions to test",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-5,
        help="Relative tolerance for tensor comparison",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-5,
        help="Absolute tolerance for tensor comparison",
    )
    return parser.parse_args()


def generate_pos_embedding(
    head_dim: int, position_ids: torch.tensor, base: int = 10000
) -> Tuple[torch.tensor, torch.tensor]:
    """
    Generate positional embeddings for rotary position encoding.

    Parameters
    ----------
    head_dim : int
        Dimension of each attention head
    position_ids : torch.Tensor
        Tensor of position indices
    base : int, optional
        Base for frequency computation, by default 10000

    Returns
    -------
    tuple of torch.Tensor
        cos : Cosine embeddings for rotary position encoding
        sin : Sine embeddings for rotary position encoding
    """

    # Core RoPE block
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2) / head_dim))
    inv_freq_expanded = (
        inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
    )
    position_ids_expanded = position_ids[:, None, :].float()
    freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos()
    sin = emb.sin()
    return cos, sin


@nki.jit
def _nki_apply_rotary_embedding_core(q_tile, k_tile, cos_tile, sin_tile, output_tile):
    """
    Core NKI implementation of rotary position embedding computation.

    Parameters
    ----------
    q_tile : nl.Tensor
        Query tensor tile
    k_tile : nl.Tensor
        Key tensor tile
    cos_tile : nl.Tensor
        Cosine embedding tile
    sin_tile : nl.Tensor
        Sine embedding tile
    output_tile : nl.Tensor
        Output buffer for results

    Notes
    -----
    The function applies rotary position embedding to query and key tensors
    using the provided cosine and sine embeddings.
    """

    assert q_tile.shape[-1] % 2 == 0, "Sequence length for q_tile must be even!"
    assert k_tile.shape[-1] % 2 == 0, "Sequence length for k_tile must be even!"
    assert (
        q_tile.shape[-1] == k_tile.shape[-1]
    ), "q_tile and k_tile must have the same sequence length"

    seq_len = q_tile.shape[-1]

    # Rotate Q
    output_tile[0, :, :] = q_tile * cos_tile
    output_tile[0, :, : seq_len // 2] = output_tile[0, :, : seq_len // 2] + (
        -1 * q_tile[:, seq_len // 2 :] * sin_tile[:, : seq_len // 2]
    )
    output_tile[0, :, seq_len // 2 :] = output_tile[0, :, seq_len // 2 :] + (
        q_tile[:, : seq_len // 2] * sin_tile[:, seq_len // 2 :]
    )

    # Rotate K
    output_tile[1, :, :] = k_tile * cos_tile
    output_tile[1, :, : seq_len // 2] = output_tile[1, :, : seq_len // 2] + (
        -1 * k_tile[:, seq_len // 2 :] * sin_tile[:, : seq_len // 2]
    )
    output_tile[1, :, seq_len // 2 :] = output_tile[1, :, seq_len // 2 :] + (
        k_tile[:, : seq_len // 2] * sin_tile[:, seq_len // 2 :]
    )


def div_ceil(n: int, d: int) -> int:
    """
    Compute ceiling division of two numbers.

    Parameters
    ----------
    n : int
        Numerator
    d : int
        Denominator

    Returns
    -------
    int
        Ceiling division result
    """
    return (n + d - 1) // d


def neuron_apply_rotary_embedding(
    q: torch.tensor, k: torch.tensor, cos: torch.tensor, sin: torch.tensor
) -> Tuple[torch.tensor, torch.tensor]:
    """
    Original rotary embedding implementation using transformers library.

    Parameters
    ----------
    q : torch.Tensor
        Query tensor
    k : torch.Tensor
        Key tensor
    cos : torch.Tensor
        Cosine embeddings
    sin : torch.Tensor
        Sine embeddings

    Returns
    -------
    tuple of torch.Tensor
        Transformed query and key tensors
    """
    return apply_rotary_pos_emb(q, k, cos, sin)


@nki.jit
def nki_apply_rotary_embedding(q, k, cos, sin):
    """
    NKI implementation of rotary position embedding.

    Parameters
    ----------
    q : torch.Tensor
        Query tensor of shape [batch_size, num_heads, seq_len, head_dim]
    k : torch.Tensor
        Key tensor of shape [batch_size, num_heads, seq_len, head_dim]
    cos : torch.Tensor
        Cosine embeddings
    sin : torch.Tensor
        Sine embeddings

    Returns
    -------
    nl.Tensor
        Output tensor containing transformed query and key tensors

    Raises
    ------
    AssertionError
        If input tensor shapes don't match or head dimension > 128
    """
    assert (
        q.shape == k.shape
    ), f"Shape of Q Tensor: {q.shape} doesn't match shape of K Tensor: {k.shape}"
    assert (
        cos.shape == sin.shape
    ), f"Shape of cos Tensor: {cos.shape} doesn't match shape of sin Tensor: {sin.shape}"
    assert (
        q.shape[-1] <= 128
    ), f"Shape of head dim (last dim) is more than 128: {q.shape}"

    batch_id = nl.program_id(axis=0)
    head_id = nl.program_id(axis=1)
    seq_len = q.shape[2]
    num_seq_batches = div_ceil(seq_len, nl.tile_size.pmax)
    output = nl.ndarray([2] + list(q.shape), dtype=q.dtype, buffer=nl.shared_hbm)
    i_p, i_f = nl.mgrid[0:128, 0:q.shape[-1]]
    for seq_batch_id in nl.affine_range(0, num_seq_batches):
        q_hbm_tile = q[batch_id, head_id]
        k_hbm_tile = k[batch_id, head_id]
        cos_hbm_tile = cos[batch_id]
        sin_hbm_tile = sin[batch_id]

        q_tile = nl.load(
            q_hbm_tile[seq_batch_id * nl.tile_size.pmax + i_p, i_f],
            mask=(seq_batch_id * nl.tile_size.pmax + i_p < seq_len),
        )
        k_tile = nl.load(
            k_hbm_tile[seq_batch_id * nl.tile_size.pmax + i_p, i_f],
            mask=(seq_batch_id * nl.tile_size.pmax + i_p < seq_len),
        )
        output_tile = nl.ndarray(
            [2] + [nl.par_dim(k_tile.shape[0]), k_tile.shape[1]],
            dtype=k_tile.dtype,
            buffer=nl.sbuf,
        )
        cos_tile = nl.load(
            cos_hbm_tile[seq_batch_id * nl.tile_size.pmax + i_p, i_f],
            mask=(seq_batch_id * nl.tile_size.pmax + i_p < seq_len),
        )
        sin_tile = nl.load(
            sin_hbm_tile[seq_batch_id * nl.tile_size.pmax + i_p, i_f],
            mask=(seq_batch_id * nl.tile_size.pmax + i_p < seq_len),
        )

        _nki_apply_rotary_embedding_core(
            q_tile, k_tile, cos_tile, sin_tile, output_tile
        )

        output_q_hbm_tile = output[0, batch_id, head_id]
        output_k_hbm_tile = output[1, batch_id, head_id]

        nl.store(
            output_q_hbm_tile[seq_batch_id * nl.tile_size.pmax + i_p, i_f],
            output_tile[0, :, :],
            mask=(seq_batch_id * nl.tile_size.pmax + i_p < seq_len),
        )
        nl.store(
            output_k_hbm_tile[seq_batch_id * nl.tile_size.pmax + i_p, i_f],
            output_tile[1, :, :],
            mask=(seq_batch_id * nl.tile_size.pmax + i_p < seq_len),
        )

    return output


def verify_results(nki_result, expected_q, expected_k, rtol=1e-5, atol=1e-5):
    """
    Verify NKI implementation results against expected results.

    Parameters
    ----------
    nki_result : tuple of torch.Tensor
        Results from NKI implementation
    expected_q : torch.Tensor
        Expected query tensor
    expected_k : torch.Tensor
        Expected key tensor
    rtol : float, optional
        Relative tolerance, by default 1e-5
    atol : float, optional
        Absolute tolerance, by default 1e-5

    Returns
    -------
    bool
        True if results match within tolerance, False otherwise
    """
    nki_q, nki_k = nki_result[0].cpu(), nki_result[1].cpu()

    q_close = torch.allclose(expected_q, nki_q, rtol=rtol, atol=atol)
    k_close = torch.allclose(expected_k, nki_k, rtol=rtol, atol=atol)

    if not q_close:
        q_max_diff = torch.max(torch.abs(expected_q - nki_q))
        logger.error(f"Q tensors not close! Max difference: {q_max_diff}")

    if not k_close:
        k_max_diff = torch.max(torch.abs(expected_k - nki_k))
        logger.error(f"K tensors not close! Max difference: {k_max_diff}")

    return q_close and k_close


def run_test(
    bs: int, nh: int, sl: int, hd: int, rtol: float = 1e-5, atol: float = 1e-5
):
    """
    Run benchmark test for a single configuration.

    Parameters
    ----------
    bs : int
        Batch size
    nh : int
        Number of attention heads
    sl : int
        Sequence length
    hd : int
        Head dimension
    rtol : float, optional
        Relative tolerance, by default 1e-5
    atol : float, optional
        Absolute tolerance, by default 1e-5

    Returns
    -------
    dict
        Test results containing:
        - nki_result : Output from NKI implementation
        - traced_result : Output from traced implementation
        - profile_traced : Profiling results for traced version
        - profile_nki : Profiling results for NKI version
        - config : Test configuration string

    Raises
    ------
    ValueError
        If output verification fails
    """
    logger.info(
        f"Testing configuration: batch_size={bs}, num_heads={nh}, seq_len={sl}, head_dim={hd}"
    )

    device = xm.xla_device()

    # Initial tensors for warmup
    cache_ids = torch.stack([torch.arange(sl) for _ in range(bs)])
    q = torch.randn(bs, nh, sl, hd)
    k = torch.randn(bs, nh, sl, hd)
    cos, sin = generate_pos_embedding(hd, cache_ids)

    # Traced version warmup
    logger.info("Warming up traced version...")
    traced_apply = torch_neuronx.trace(neuron_apply_rotary_embedding, (q, k, cos, sin))
    _, _ = traced_apply(q, k, cos, sin)

    # Create new tensors for actual profiling
    cache_ids = torch.stack([torch.arange(sl) for _ in range(bs)])
    q = torch.randn(bs, nh, sl, hd)
    k = torch.randn(bs, nh, sl, hd)
    cos, sin = generate_pos_embedding(hd, cache_ids)

    prof_traced = None
    prof_nki = None

    logger.info("Profiling traced version...")
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.XPU],
        record_shapes=True,
        profile_memory=True,
    ) as prof_traced:
        with record_function("traced_rotary"):
            expected_q_emb, expected_k_emb = traced_apply(q, k, cos, sin)
            xm.mark_step()

    logger.info("\nTraced Version Profile:")
    logger.info(prof_traced.key_averages().table(sort_by="cpu_time_total", row_limit=5))

    # NKI version
    logger.info("Running NKI implementation...")
    q_device = q.to(device)
    k_device = k.to(device)
    cos_device = cos.to(device)
    sin_device = sin.to(device)

    # Warmup NKI version
    logger.info("Warming up NKI version...")
    nki_result = nki_apply_rotary_embedding[bs, nh](
        q_device, k_device, cos_device, sin_device
    )
    xm.mark_step()

    logger.info("Profiling NKI version...")
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.XPU],
        record_shapes=True,
        profile_memory=True,
    ) as prof_nki:
        with record_function("nki_rotary"):
            nki_result = nki_apply_rotary_embedding[bs, nh](
                q_device, k_device, cos_device, sin_device
            )
            xm.mark_step()

    logger.info("\nNKI Version Profile:")
    logger.info(prof_nki.key_averages().table(sort_by="cpu_time_total", row_limit=5))

    traced_time = prof_traced.key_averages().table(
        sort_by="cpu_time_total", row_limit=5
    )
    nki_time = prof_nki.key_averages().table(sort_by="cpu_time_total", row_limit=5)

    logger.info("\nPerformance Comparison:")
    logger.info("Traced version timing:")
    logger.info(traced_time)
    logger.info("NKI version timing:")
    logger.info(nki_time)

    logger.info("Verifying results...")
    if verify_results(nki_result, expected_q_emb, expected_k_emb, rtol=rtol, atol=atol):
        logger.success(f"Test passed successfully for dims: {bs}x{nh}x{sl}x{hd}")
    else:
        logger.error(f"Test failed for dims: {bs}x{nh}x{sl}x{hd}")
        raise ValueError("Output verification failed!")

    return {
        "nki_result": nki_result,
        "traced_result": (expected_q_emb, expected_k_emb),
        "profile_traced": prof_traced,
        "profile_nki": prof_nki,
        "config": f"bs={bs}, nh={nh}, sl={sl}, hd={hd}",
    }


def analyze_performance(test_results):
    """
    Analyze and summarize performance results for all test configurations.

    Parameters
    ----------
    test_results : list of dict
        List of test results from run_test()

    Notes
    -----
    The function computes and logs:
    - Individual configuration performance comparisons
    - Minimum, maximum, and average speedup across all configurations
    - Detailed timing breakdown for both implementations
    """
    if not any(r["profile_traced"] for r in test_results):
        return

    logger.info("\nPerformance Analysis Summary by Configuration:")

    for result in test_results:
        if result["profile_traced"] and result["profile_nki"]:
            # Extract configuration from results
            config = result.get("config", "Unknown")

            traced_events = result["profile_traced"].key_averages()
            traced_forward = next(
                (event for event in traced_events if event.key == "neuron::forward_v2"),
                None,
            )
            traced_time = traced_forward.cpu_time_total if traced_forward else 0

            # Get NKI version nki_rotary time
            nki_events = result["profile_nki"].key_averages()
            nki_rotary = next(
                (event for event in nki_events if event.key == "nki_rotary"), None
            )
            nki_time = nki_rotary.cpu_time_total if nki_rotary else 0

            speedup = traced_time / nki_time if nki_time > 0 else 0

            logger.info(f"\nConfiguration: {config}")
            logger.info(f"Traced Version (neuron::forward_v2): {traced_time:.2f} us")
            logger.info(f"NKI Version (nki_rotary): {nki_time:.2f} us")
            logger.info(f"Speedup (Traced/NKI): {speedup:.2f}x")

    speedups = []
    for result in test_results:
        if result["profile_traced"] and result["profile_nki"]:
            traced_forward = next(
                (
                    event
                    for event in result["profile_traced"].key_averages()
                    if event.key == "neuron::forward_v2"
                ),
                None,
            )
            nki_rotary = next(
                (
                    event
                    for event in result["profile_nki"].key_averages()
                    if event.key == "nki_rotary"
                ),
                None,
            )
            if traced_forward and nki_rotary:
                speedup = traced_forward.cpu_time_total / nki_rotary.cpu_time_total
                speedups.append(speedup)

    if speedups:
        logger.info(f"\nSpeedup Statistics:")
        logger.info(f"Min Speedup: {min(speedups):.2f}x")
        logger.info(f"Max Speedup: {max(speedups):.2f}x")
        logger.info(f"Average Speedup: {sum(speedups) / len(speedups):.2f}x")


def main():
    """
    Main function to run rotary embedding benchmark suite.

    Notes
    -----
    Function performs the following operations:
    - Parses command line arguments
    - Runs tests for all configurations
    - Analyzes performance results
    - Saves test summary to JSON file
    - Handles logging and error reporting
    """
    args = parse_args()

    logger.info("Starting Rotary Embedding tests with configurations:")
    logger.info(f"Batch sizes: {args.batch_sizes}")
    logger.info(f"Number of heads: {args.num_heads}")
    logger.info(f"Sequence lengths: {args.seq_lengths}")
    logger.info(f"Head dimensions: {args.head_dims}")
    logger.info(f"Relative tolerance: {args.rtol}")
    logger.info(f"Absolute tolerance: {args.atol}")

    total_tests = (
        len(args.batch_sizes)
        * len(args.num_heads)
        * len(args.seq_lengths)
        * len(args.head_dims)
    )
    current_test = 0
    failed_tests = []
    test_results = []

    for bs in args.batch_sizes:
        for nh in args.num_heads:
            for sl in args.seq_lengths:
                for hd in args.head_dims:
                    current_test += 1
                    logger.info(f"Running test {current_test}/{total_tests}")
                    try:
                        result = run_test(bs, nh, sl, hd, args.rtol, args.atol)
                        test_results.append(result)
                    except Exception as e:
                        logger.error(f"Test failed with error: {str(e)}")
                        logger.exception(e)
                        failed_tests.append((bs, nh, sl, hd))
                    logger.info("=" * 80)

    if failed_tests:
        logger.error(f"Some tests failed! Failed configurations: {failed_tests}")
        logger.error(f"Total failed tests: {len(failed_tests)}/{total_tests}")
    else:
        logger.success(f"All {total_tests} tests completed successfully!")

    analyze_performance(test_results)

    # Save test results summary
    summary = {
        "total_tests": total_tests,
        "failed_tests": failed_tests,
        "configurations": {
            "batch_sizes": args.batch_sizes,
            "num_heads": args.num_heads,
            "seq_lengths": args.seq_lengths,
            "head_dims": args.head_dims,
            "rtol": args.rtol,
            "atol": args.atol,
        },
    }

    with open(
        f"test_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", "w"
    ) as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
