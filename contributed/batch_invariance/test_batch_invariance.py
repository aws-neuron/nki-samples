"""
Simple Batch Invariance Test
"""

import torch
import time
import torch_neuronx
import numpy as np
from kernels.rmsnorm_batch_invariant import nki_rmsnorm_kernel_lang, nki_rmsnorm_kernel_isa
from kernels.matmul_batch_invariant import nki_matmul_kernel_isa, nki_matmul_kernel_lang

# Prove that the kernels match pytorch and are functionally correct
def test_matmul_kernel_correctness():
    """
    Verify NKI matmul kernels produce correct results vs PyTorch.
    
    Validates mathematical correctness before analyzing batch invariance effects.
    """
    print("Testing MatMul Correctness...")
    device = 'xla'
    
    # Test dimensions
    M, K, N = 256, 512, 512
    
    print(f"  Matrix dimensions: [{M}, {K}] @ [{K}, {N}] = [{M}, {N}]")
    print()
    
    # Create test data
    np.random.seed(42)
    a_np = np.random.randn(M, K).astype(np.float32)
    b_np = np.random.randn(K, N).astype(np.float32)
    
    # PyTorch reference (CPU)
    a_torch = torch.tensor(a_np, dtype=torch.float32)
    b_torch = torch.tensor(b_np, dtype=torch.float32)
    
    print("  Computing PyTorch reference (CPU)...")
    start = time.time()
    ref_output = torch.matmul(a_torch, b_torch)
    ref_time = time.time() - start
    print(f"    Time: {ref_time:.6f}s")
    print(f"    Output shape: {ref_output.shape}")
    print(f"    First values: {ref_output[0, :5].numpy()}")
    print()
    
    # Test Lang kernel - expects [M, K] @ [K, N]
    print("  Testing Lang kernel (nl.matmul)...")
    a_xla = torch.tensor(a_np, dtype=torch.float32, device=device)  # [M, K]
    b_xla = torch.tensor(b_np, dtype=torch.float32, device=device)  # [K, N]
    
    start = time.time()
    output_lang = nki_matmul_kernel_lang(a_xla, b_xla, batch_invariant=True)
    lang_time = time.time() - start
    
    output_lang_cpu = output_lang.cpu()
    print(f"    Time: {lang_time:.6f}s")
    print(f"    Output shape: {output_lang_cpu.shape}")
    print(f"    First values: {output_lang_cpu[0, :5].numpy()}")
    
    lang_match = torch.allclose(ref_output, output_lang_cpu, atol=1e-4, rtol=1e-2)
    max_diff_lang = torch.max(torch.abs(ref_output - output_lang_cpu)).item()
    
    if lang_match:
        print(f"    ✓ Matches PyTorch reference")
    else:
        print(f"    ✗ Differs from PyTorch reference")
    print(f"    Max difference: {max_diff_lang:.6f}")
    print()
    
    # Test ISA kernel - expects [K, M] @ [K, N]
    print("  Testing ISA kernel (nisa.nc_matmul)...")
    a_xla_t = torch.tensor(a_np.T, dtype=torch.float32, device=device)  # [K, M] - transposed!
    b_xla = torch.tensor(b_np, dtype=torch.float32, device=device)      # [K, N]
    
    start = time.time()
    output_isa = nki_matmul_kernel_isa(a_xla_t, b_xla, batch_invariant=True)
    isa_time = time.time() - start
    
    output_isa_cpu = output_isa.cpu()
    print(f"    Time: {isa_time:.6f}s")
    print(f"    Output shape: {output_isa_cpu.shape}")
    print(f"    First values: {output_isa_cpu[0, :5].numpy()}")
    
    isa_match = torch.allclose(ref_output, output_isa_cpu, atol=1e-4, rtol=1e-2)
    max_diff_isa = torch.max(torch.abs(ref_output - output_isa_cpu)).item()
    
    if isa_match:
        print(f"    ✓ Matches PyTorch reference")
    else:
        print(f"    ✗ Differs from PyTorch reference")
    print(f"    Max difference: {max_diff_isa:.6f}")
    print()
    
    # Summary
    print("=" * 80)
    if lang_match and isa_match:
        print("✓ Both kernels produce correct results")
    else:
        print("✗ One or more kernels differ from PyTorch reference")
        if not lang_match:
            print(f"  Lang kernel max error: {max_diff_lang:.6f}")
        if not isa_match:
            print(f"  ISA kernel max error: {max_diff_isa:.6f}")
    
    assert lang_match, f"Lang kernel doesn't match PyTorch (max diff: {max_diff_lang})"
    assert isa_match, f"ISA kernel doesn't match PyTorch (max diff: {max_diff_isa})"

def test_matmul_isa():
    """
    ISA kernel K-tiling batch variance with quantization erasure.
    
    Expected: bfloat16 error = 0.0 despite float32 showing differences
    Reason: nisa.nc_matmul produces float32 errors below bfloat16 threshold (~0.008)
    Result: Demonstrates hardware-level numerical stability
    
    Returns:
        dict: Test results with float32 and bfloat16 errors
    """
    print("Testing MatMul batch variance (ISA kernel)...")
    device = 'xla'
    
    K, N = 512, 512
    M_TILE = 128
    large_batch = 256  # 2x M_TILE
    small_batch = 128  # 1x M_TILE
    
    print(f"  K={K} -> batch_invariant=True: K_TILE=128, batch_invariant=False: K_TILE=64")
    print()
    
    # Create data ONCE in float32 - ISA kernel needs [K, M] layout!
    print("  Creating data in float32...")
    a_large_f32 = torch.linspace(-1, 1, large_batch * K, device=device).reshape(K, large_batch).to(torch.float32)
    b_f32 = torch.linspace(-1, 1, K * N, device=device).reshape(K, N).to(torch.float32)
    
    # Test with float32 FIRST
    print("  Testing with float32:")
    a_small_f32 = a_large_f32[:, :small_batch]  # [K, 128]
    
    result_small_f32 = nki_matmul_kernel_isa(a_small_f32, b_f32, batch_invariant=True)
    result_large_f32 = nki_matmul_kernel_isa(a_large_f32, b_f32, batch_invariant=False)
    
    diff_f32 = torch.max(torch.abs(result_small_f32 - result_large_f32[:small_batch])).item()
    print(f"    Max difference: {diff_f32:.6f}")
    print(f"    Results {'identical' if diff_f32 < 1e-6 else 'differ'}")
    print()
    
    # Cast to bfloat16
    print("  Testing with bfloat16:")
    a_large_bf16 = a_large_f32.to(torch.bfloat16)
    b_bf16 = b_f32.to(torch.bfloat16)
    a_small_bf16 = a_large_bf16[:, :small_batch]
    
    result_small_bf16 = nki_matmul_kernel_isa(a_small_bf16, b_bf16, batch_invariant=True)
    result_large_bf16 = nki_matmul_kernel_isa(a_large_bf16, b_bf16, batch_invariant=False)
    
    diff_bf16 = torch.max(torch.abs(result_small_bf16 - result_large_bf16[:small_batch])).item()
    print(f"    Max difference: {diff_bf16:.6f}")
    print(f"    Results {'identical' if diff_bf16 < 1e-6 else 'differ'}")
    print()
    
    if diff_f32 > 0:
        ratio = diff_bf16 / diff_f32
        print(f"  Precision impact: bfloat16 error is {ratio:.2f}x {'larger' if diff_bf16 > diff_f32 else 'smaller'} than float32")
        if diff_bf16 == 0.0:
            print(f"  Note: Float32 error ({diff_f32:.6f}) is below bfloat16 quantization threshold (~0.008)")
            print(f"        Quantization erases the difference rather than amplifying it")
    else:
        ratio = 0.0
        print(f"  Precision impact: N/A (no float32 difference detected)")
    
    return {
        "kernel": "ISA (nisa.nc_matmul)",
        "float32_error": diff_f32,
        "bfloat16_error": diff_bf16,
        "amplification": ratio
    }

def test_matmul_lang():
    """
    Lang kernel K-tiling batch variance with precision amplification.
    
    Expected: bfloat16 error ~170x larger than float32
    Reason: nl.matmul produces float32 errors above bfloat16 threshold
    Result: Demonstrates how reduced precision amplifies tiling strategy effects
    
    Returns:
        dict: Test results with float32 and bfloat16 errors
    """
    print("Testing MatMul batch variance (Lang kernel)...")
    device = 'xla'
    
    K, N = 512, 512
    M_TILE = 128
    large_batch = 256  # 2x M_TILE
    small_batch = 128  # 1x M_TILE
    
    print(f"  K={K} -> batch_invariant=True: K_TILE=128, batch_invariant=False: K_TILE=64")
    print()
    
    # Create data ONCE in float32 - single source of truth
    print("  Creating data in float32...")
    a_large_f32 = torch.linspace(-1, 1, large_batch * K, device=device).reshape(large_batch, K).to(torch.float32)
    b_f32 = torch.linspace(-1, 1, K * N, device=device).reshape(K, N).to(torch.float32)
    
    # Test with float32 FIRST
    print("  Testing with float32:")
    # Test the SAME 128 rows in different batch contexts
    a_small_f32 = a_large_f32[:small_batch, :]
    
    # Process as small batch (128 rows)
    result_small_f32 = nki_matmul_kernel_lang(a_small_f32, b_f32, batch_invariant=True)
    
    # Process as part of large batch (256 rows)
    result_large_f32 = nki_matmul_kernel_lang(a_large_f32, b_f32, batch_invariant=False)
    
    # Compare the SAME rows
    diff_f32 = torch.max(torch.abs(result_small_f32 - result_large_f32[:small_batch])).item()
    print(f"    Max difference between K_TILE strategies: {diff_f32:.6f}")
    print(f"    Results {'identical' if diff_f32 < 1e-6 else 'differ'}")
    print()
    
    # Cast to bfloat16 from the SAME float32 source
    print("  Testing with bfloat16:")
    a_large_bf16 = a_large_f32.to(torch.bfloat16)
    b_bf16 = b_f32.to(torch.bfloat16)
    
    # Test the SAME 128 rows in different batch contexts
    a_small_bf16 = a_large_bf16[:small_batch, :]
    
    # Process as small batch (128 rows)
    result_small_bf16 = nki_matmul_kernel_lang(a_small_bf16, b_bf16, batch_invariant=True)
    
    # Process as part of large batch (256 rows)
    result_large_bf16 = nki_matmul_kernel_lang(a_large_bf16, b_bf16, batch_invariant=False)
    
    # Compare the SAME rows
    diff_bf16 = torch.max(torch.abs(result_small_bf16 - result_large_bf16[:small_batch])).item()
    print(f"    Max difference between K_TILE strategies: {diff_bf16:.6f}")
    print(f"    Results {'identical' if diff_bf16 < 1e-6 else 'differ'}")
    print()
    
    if diff_f32 > 0:
        ratio = diff_bf16 / diff_f32
        print(f"  Precision impact: bfloat16 error is {ratio:.2f}x larger than float32")
        print(f"  This demonstrates how reduced precision amplifies tiling strategy effects")
    else:
        ratio = 0.0
        print(f"  Precision impact: N/A (no float32 difference detected)")
    
    return {
        "kernel": "Lang (nl.matmul)",
        "float32_error": diff_f32,
        "bfloat16_error": diff_bf16,
        "amplification": ratio
    }




def test_rmsnorm_lang():
    """
    RMSNorm Lang kernel HIDDEN_TILE variance with precision effects.
    
    Uses nl.load, nl.store, nl.sum for data movement and reduction.
    Different HIDDEN_TILE sizes create different reduction orders.
    
    Expected: Shows variance in both float32 and bfloat16
    
    Returns:
        dict: Test results with float32 and bfloat16 errors
    """
    print("Testing RMSNorm batch variance (Lang kernel)...")
    device = 'xla'
    hidden_dim = 512
    large_batch = 128
    small_batch = 32
    
    print(f"  hidden_dim={hidden_dim}")
    print(f"    batch_invariant=True:  HIDDEN_TILE=256 (2 chunks, 1 accumulation)")
    print(f"    batch_invariant=False: HIDDEN_TILE=128 (4 chunks, 3 accumulations)")
    print()
    
    # Create data ONCE in float32
    print("  Creating data in float32...")
    a_large_f32 = torch.linspace(-1, 1, large_batch * hidden_dim, device=device).reshape(large_batch, hidden_dim).to(torch.float32)
    g_f32 = torch.ones(hidden_dim, device=device, dtype=torch.float32)
    
    # Test with float32 FIRST
    print("  Testing with float32:")
    a_small_f32 = a_large_f32[:small_batch, :]
    
    result_small_f32 = nki_rmsnorm_kernel_lang(a_small_f32, g_f32, batch_invariant=True)
    result_large_f32 = nki_rmsnorm_kernel_lang(a_large_f32, g_f32, batch_invariant=False)
    
    diff_f32 = torch.max(torch.abs(result_small_f32 - result_large_f32[:small_batch])).item()
    print(f"    Max difference between HIDDEN_TILE strategies: {diff_f32:.6f}")
    print(f"    Results {'identical' if diff_f32 < 1e-6 else 'differ'}")
    print()
    
    # Cast to bfloat16
    print("  Testing with bfloat16:")
    a_large_bf16 = a_large_f32.to(torch.bfloat16)
    g_bf16 = g_f32.to(torch.bfloat16)
    a_small_bf16 = a_large_bf16[:small_batch, :]
    
    result_small_bf16 = nki_rmsnorm_kernel_lang(a_small_bf16, g_bf16, batch_invariant=True)
    result_large_bf16 = nki_rmsnorm_kernel_lang(a_large_bf16, g_bf16, batch_invariant=False)
    
    diff_bf16 = torch.max(torch.abs(result_small_bf16 - result_large_bf16[:small_batch])).item()
    print(f"    Max difference between HIDDEN_TILE strategies: {diff_bf16:.6f}")
    print(f"    Results {'identical' if diff_bf16 < 1e-6 else 'differ'}")
    print()
    
    if diff_f32 > 0:
        ratio = diff_bf16 / diff_f32
        print(f"  Precision impact: bfloat16 error is {ratio:.2f}x {'larger' if diff_bf16 > diff_f32 else 'smaller'} than float32")
        print(f"  Lang kernel shows variance due to different reduction chunking")
    else:
        ratio = 0.0
        print(f"  Precision impact: N/A (no float32 difference detected)")
    
    return {
        "kernel": "RMSNorm Lang (nl.sum)",
        "float32_error": diff_f32,
        "bfloat16_error": diff_bf16,
        "amplification": ratio
    }


def test_rmsnorm_isa():
    """
    RMSNorm ISA kernel demonstrates batch INVARIANCE.
    
    Uses nisa.dma_copy and nisa.tensor_reduce with skip_middle_end_transformations.
    Despite different HIDDEN_TILE sizes, ISA produces identical results.
    
    Expected: No variance in either float32 or bfloat16
    Reason: ISA-level operations are deterministic regardless of tiling strategy
    
    Returns:
        dict: Test results with float32 and bfloat16 errors (should be 0.0)
    """
    print("Testing RMSNorm batch INVARIANCE (ISA kernel)...")
    device = 'xla'
    hidden_dim = 512
    large_batch = 128
    small_batch = 32
    
    print(f"  hidden_dim={hidden_dim}")
    print(f"    batch_invariant=True:  HIDDEN_TILE=256 (2 chunks, 1 accumulation)")
    print(f"    batch_invariant=False: HIDDEN_TILE=128 (4 chunks, 3 accumulations)")
    print(f"  Note: ISA kernel uses @skip_middle_end_transformations")
    print()
    
    # Create data ONCE in float32
    print("  Creating data in float32...")
    a_large_f32 = torch.linspace(-1, 1, large_batch * hidden_dim, device=device).reshape(large_batch, hidden_dim).to(torch.float32)
    g_f32 = torch.ones(hidden_dim, device=device, dtype=torch.float32)
    
    # Test with float32 FIRST
    print("  Testing with float32:")
    a_small_f32 = a_large_f32[:small_batch, :]
    
    result_small_f32 = nki_rmsnorm_kernel_isa(a_small_f32, g_f32, batch_invariant=True)
    result_large_f32 = nki_rmsnorm_kernel_isa(a_large_f32, g_f32, batch_invariant=False)
    
    diff_f32 = torch.max(torch.abs(result_small_f32 - result_large_f32[:small_batch])).item()
    print(f"    Max difference between HIDDEN_TILE strategies: {diff_f32:.6f}")
    print(f"    Results {'identical' if diff_f32 < 1e-6 else 'differ'}")
    print()
    
    # Cast to bfloat16
    print("  Testing with bfloat16:")
    a_large_bf16 = a_large_f32.to(torch.bfloat16)
    g_bf16 = g_f32.to(torch.bfloat16)
    a_small_bf16 = a_large_bf16[:small_batch, :]
    
    result_small_bf16 = nki_rmsnorm_kernel_isa(a_small_bf16, g_bf16, batch_invariant=True)
    result_large_bf16 = nki_rmsnorm_kernel_isa(a_large_bf16, g_bf16, batch_invariant=False)
    
    diff_bf16 = torch.max(torch.abs(result_small_bf16 - result_large_bf16[:small_batch])).item()
    print(f"    Max difference between HIDDEN_TILE strategies: {diff_bf16:.6f}")
    print(f"    Results {'identical' if diff_bf16 < 1e-6 else 'differ'}")
    print()
    
    if diff_f32 == 0.0 and diff_bf16 == 0.0:
        print(f"  ✓ ISA kernel is BATCH INVARIANT!")
        print(f"    @skip_middle_end_transformations ensures deterministic reduction")
        print(f"    regardless of HIDDEN_TILE size")
        ratio = 0.0
    elif diff_f32 > 0:
        ratio = diff_bf16 / diff_f32 if diff_f32 > 0 else 0.0
        print(f"  Precision impact: bfloat16 error is {ratio:.2f}x {'larger' if diff_bf16 > diff_f32 else 'smaller'} than float32")
    else:
        ratio = 0.0
        print(f"  Precision impact: N/A")
    
    return {
        "kernel": "RMSNorm ISA (nisa.tensor_reduce)",
        "float32_error": diff_f32,
        "bfloat16_error": diff_bf16,
        "amplification": ratio
    }


if __name__ == "__main__":
    import pandas as pd
    
    print("Batch Invariance Test")
    print("=" * 80)
    
    # Run correctness test
    test_matmul_kernel_correctness()
    print("=" * 80)
    
    # Test Lang kernel
    print("\nRunning Lang kernel test...")
    lang_results = test_matmul_lang()
    
    print("=" * 80)
    
    # Test ISA kernel
    print("\nRunning ISA kernel test...")
    isa_results = test_matmul_isa()
    
    print("=" * 80)
    
    # Test RMSNorm Lang kernel
    print("\nRunning RMSNorm Lang kernel test...")
    rmsnorm_lang_results = test_rmsnorm_lang()
    
    print("=" * 80)
    
    # Test RMSNorm ISA kernel
    print("\nRunning RMSNorm ISA kernel test...")
    rmsnorm_isa_results = test_rmsnorm_isa()
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    # Create results dataframe
    print("\nBatch Variance Results:")
    variance_df = pd.DataFrame([lang_results, isa_results, rmsnorm_lang_results, rmsnorm_isa_results])
    print(variance_df.to_string(index=False))
    print()
