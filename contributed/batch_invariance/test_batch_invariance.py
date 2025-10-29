"""
Simple Batch Invariance Test
"""

import torch
import time
import torch_neuronx
import numpy as np
from kernels.rmsnorm_batch_invariant import nki_rmsnorm_kernel
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

def test_rmsnorm_invariant():
    """
    RMSNorm demonstrates batch INVARIANCE with consistent tiling.
    
    When using the same batch_invariant=True setting, results should be
    identical regardless of batch size because each row is computed independently.
    
    Returns:
        dict: Test results showing invariance
    """
    print("Testing RMSNorm batch invariance...")
    
    device = 'xla'
    hidden_dim = 256
    
    # Create a large input with many rows
    large_batch = 128
    a_large = torch.linspace(-1, 1, large_batch * hidden_dim, device=device).reshape(large_batch, hidden_dim).to(torch.bfloat16)
    g = torch.ones(hidden_dim, device=device, dtype=torch.bfloat16)
    
    # Test the SAME 32 rows in different batch contexts
    a_small = a_large[:32, :]
    
    # Process as small batch (32 rows)
    result_small = nki_rmsnorm_kernel(a_small, g, batch_invariant=True)
    
    # Process as part of large batch (128 rows)
    result_large = nki_rmsnorm_kernel(a_large, g, batch_invariant=True)
    
    # Compare the SAME rows
    diff = torch.max(torch.abs(result_small - result_large[:32])).item()
    match = diff < 1e-6
    
    print(f"  First 32 rows: batch=32 vs batch=128: {'MATCH ✓' if match else 'DIFFER ✗'}")
    print(f"  Max difference: {diff:.6f}")
    
    if match:
        print(f"  ✓ RMSNorm is batch-invariant!")
        print(f"    Each row computed independently, reduction is atomic")
        print(f"    Tile size only affects parallelism, not computation order")
    
    return {
        "test": "RMSNorm Invariant",
        "max_difference": diff,
        "is_invariant": match
    }

def test_rmsnorm_variant():
    """
    RMSNorm demonstrates batch VARIANCE with different tiling strategies.
    
    When using different batch_invariant settings (True vs False), results may
    differ due to different HIDDEN_TILE sizes affecting reduction chunking.
    
    Returns:
        dict: Test results showing variance
    """
    print("Testing RMSNorm batch variance...")
    
    device = 'xla'
    hidden_dim = 256
    
    # Create a large input with many rows
    large_batch = 128
    a_large = torch.linspace(-1, 1, large_batch * hidden_dim, device=device).reshape(large_batch, hidden_dim).to(torch.bfloat16)
    g = torch.ones(hidden_dim, device=device, dtype=torch.bfloat16)
    
    # Test the SAME 32 rows in different batch contexts
    a_small = a_large[:32, :]
    
    # Process as small batch (32 rows) with batch_invariant=True
    result_small = nki_rmsnorm_kernel(a_small, g, batch_invariant=True)
    
    # Process as part of large batch (128 rows) with batch_invariant=False
    result_large = nki_rmsnorm_kernel(a_large, g, batch_invariant=False)
    
    diff_bf16 = torch.max(torch.abs(result_small - result_large[:32])).item()
    print(f"  Max difference between HIDDEN_TILE strategies: {diff_bf16:.6f}")
    print(f"  Results {'identical' if diff_bf16 < 1e-6 else 'differ'}")
    
    if diff_bf16 > 1e-6:
        print(f"  ✗ Different HIDDEN_TILE sizes produce different results")
        print(f"    This demonstrates tiling strategy affects reduction order")
    
    return {
        "test": "RMSNorm Variant",
        "max_difference": diff_bf16,
        "is_invariant": diff_bf16 < 1e-6
    }


def test_rmsnorm_accuracy_diff():
    """
    RMSNorm HIDDEN_TILE variance with precision effects.
    
    Tests how different HIDDEN_TILE sizes affect reduction chunking and
    whether precision amplifies these differences.
    
    Returns:
        dict: Test results with float32 and bfloat16 errors
    """
    print("Testing RMSNorm HIDDEN_TILE variance...")
    device = 'xla'
    hidden_dim = 512
    large_batch = 128
    small_batch = 32
    
    print(f"  hidden_dim={hidden_dim}")
    print(f"    batch_invariant=True:  HIDDEN_TILE=256 (2 chunks, 1 accumulation)")
    print(f"    batch_invariant=False: HIDDEN_TILE=128 (4 chunks, 3 accumulations)")
    print()
    
    # Test with bfloat16
    print("  Testing with bfloat16:")
    a_large_bf16 = torch.linspace(-1, 1, large_batch * hidden_dim, device=device).reshape(large_batch, hidden_dim).to(torch.bfloat16)
    g_bf16 = torch.ones(hidden_dim, device=device, dtype=torch.bfloat16)
    
    # Test the SAME 32 rows in different batch contexts
    a_small_bf16 = a_large_bf16[:small_batch, :]
    
    # Process as small batch (32 rows)
    result_small_bf16 = nki_rmsnorm_kernel(a_small_bf16, g_bf16, batch_invariant=True)   # HIDDEN_TILE=256
    
    # Process as part of large batch (128 rows)
    result_large_bf16 = nki_rmsnorm_kernel(a_large_bf16, g_bf16, batch_invariant=False)  # HIDDEN_TILE=128
    
    # Compare the SAME rows
    diff_bf16 = torch.max(torch.abs(result_small_bf16 - result_large_bf16[:small_batch])).item()
    print(f"    Max difference between HIDDEN_TILE strategies: {diff_bf16:.6f}")
    print(f"    Results {'identical' if diff_bf16 < 1e-6 else 'differ'}")
    print()
    
    # Test with float32
    print("  Testing with float32:")
    a_large_f32 = torch.linspace(-1, 1, large_batch * hidden_dim, device=device).reshape(large_batch, hidden_dim).to(torch.float32)
    g_f32 = torch.ones(hidden_dim, device=device, dtype=torch.float32)
    
    # Test the SAME 32 rows in different batch contexts
    a_small_f32 = a_large_f32[:small_batch, :]
    
    # Process as small batch (32 rows)
    result_small_f32 = nki_rmsnorm_kernel(a_small_f32, g_f32, batch_invariant=True)   # HIDDEN_TILE=256
    
    # Process as part of large batch (128 rows)
    result_large_f32 = nki_rmsnorm_kernel(a_large_f32, g_f32, batch_invariant=False)  # HIDDEN_TILE=128
    
    # Compare the SAME rows
    diff_f32 = torch.max(torch.abs(result_small_f32 - result_large_f32[:small_batch])).item()
    print(f"    Max difference between HIDDEN_TILE strategies: {diff_f32:.6f}")
    print(f"    Results {'identical' if diff_f32 < 1e-6 else 'differ'}")
    print()
    
    if diff_f32 > 0:
        ratio = diff_bf16 / diff_f32
        print(f"  Precision impact: bfloat16 error is {ratio:.2f}x {'larger' if diff_bf16 > diff_f32 else 'smaller'} than float32")
    else:
        ratio = 0.0
        print(f"  Precision impact: N/A (no float32 difference detected)")
    
    return {
        "kernel": "RMSNorm (HIDDEN_TILE)",
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
    
    # Test RMSNorm invariance
    print("=" * 80)
    print("\nRunning RMSNorm batch invariance test...")
    rmsnorm_invariant = test_rmsnorm_invariant()
    
    print("=" * 80)
    
    # Test RMSNorm variance
    print("\nRunning RMSNorm batch variance test...")
    rmsnorm_variant = test_rmsnorm_variant()
    
    print("=" * 80)
    
    # Test RMSNorm HIDDEN_TILE precision effects
    print("\nRunning RMSNorm HIDDEN_TILE variance test...")
    rmsnorm_results = test_rmsnorm_accuracy_diff()
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    # Create results dataframes
    print("\nMatMul & RMSNorm Batch Variance Results:")
    variance_df = pd.DataFrame([lang_results, isa_results, rmsnorm_results])
    print(variance_df.to_string(index=False))
    print()
    
    print("\nRMSNorm Invariance vs Variance:")
    invariance_df = pd.DataFrame([rmsnorm_invariant, rmsnorm_variant])
    print(invariance_df.to_string(index=False))
    print()
   
