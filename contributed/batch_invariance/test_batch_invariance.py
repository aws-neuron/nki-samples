"""
Simple Batch Invariance Test
"""

import torch
import torch_neuronx
import numpy as np
from kernels.rmsnorm_batch_invariant import nki_rmsnorm_kernel
from kernels.matmul_batch_invariant import nki_matmul_kernel


def test_matmul():
    """MatMul test showing K_TILE effect and precision impact"""
    print("Testing MatMul batch invariance...")
    device = 'xla'
    K, N = 512, 512
    M_TILE = 128
    large_batch = 256  # 2x M_TILE
    small_batch = 128  # 1x M_TILE
    
    print(f"  K={K} -> batch_invariant=True: K_TILE=128, batch_invariant=False: K_TILE=64")
    print()
    
    # Test with bfloat16
    print("  Testing with bfloat16:")
    a_large_bf16 = torch.linspace(-1, 1, large_batch * K, device=device).reshape(large_batch, K).to(torch.bfloat16)
    b_bf16 = torch.linspace(-1, 1, K * N, device=device).reshape(K, N).to(torch.bfloat16)
    
    # Test the SAME 128 rows in different batch contexts
    a_small_bf16 = a_large_bf16[:small_batch, :]
    
    # Process as small batch (128 rows)
    result_small_bf16 = nki_matmul_kernel(a_small_bf16, b_bf16, batch_invariant=True)
    
    # Process as part of large batch (256 rows)
    result_large_bf16 = nki_matmul_kernel(a_large_bf16, b_bf16, batch_invariant=False)
    
    # Compare the SAME rows
    diff_bf16 = torch.max(torch.abs(result_small_bf16 - result_large_bf16[:small_batch])).item()
    print(f"    Max difference between K_TILE strategies: {diff_bf16:.6f}")
    print(f"    Results {'identical' if diff_bf16 < 1e-6 else 'differ'}")
    print()
    
    # Test with float32
    print("  Testing with float32:")
    a_large_f32 = torch.linspace(-1, 1, large_batch * K, device=device).reshape(large_batch, K).to(torch.float32)
    b_f32 = torch.linspace(-1, 1, K * N, device=device).reshape(K, N).to(torch.float32)
    
    # Test the SAME 128 rows in different batch contexts
    a_small_f32 = a_large_f32[:small_batch, :]
    
    # Process as small batch (128 rows)
    result_small_f32 = nki_matmul_kernel(a_small_f32, b_f32, batch_invariant=True)
    
    # Process as part of large batch (256 rows)
    result_large_f32 = nki_matmul_kernel(a_large_f32, b_f32, batch_invariant=False)
    
    # Compare the SAME rows
    diff_f32 = torch.max(torch.abs(result_small_f32 - result_large_f32[:small_batch])).item()
    print(f"    Max difference between K_TILE strategies: {diff_f32:.6f}")
    print(f"    Results {'identical' if diff_f32 < 1e-6 else 'differ'}")
    print()
    
    print(f"  Precision impact: bfloat16 error is {diff_bf16/diff_f32 if diff_f32 > 0 else 'N/A'}x larger than float32")
    print(f"  This demonstrates how reduced precision amplifies tiling strategy effects")


def test_rmsnorm_invariant():
    """RMSNorm demonstrates batch INVARIANCE (not variance)"""
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
    match = torch.allclose(result_small, result_large[:32], atol=1e-6)
    print(f"  First 32 rows: batch=32 vs batch=128: {'MATCH ✓' if match else 'DIFFER ✗'}")
    
    if match:
        print(f"  ✓ RMSNorm is batch-invariant!")
        print(f"    Each row computed independently, reduction is atomic")
        print(f"    Tile size only affects parallelism, not computation order")

def test_rmsnorm_variant():
    """RMSNorm demonstrates batch INVARIANCE (not variance)"""
    print("Testing RMSNorm batch variance...")
    
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
    result_large = nki_rmsnorm_kernel(a_large, g, batch_invariant=False)
    
    diff_bf16 = torch.max(torch.abs(result_small - result_large[:32])).item()
    print(f"    Max difference between HIDDEN_TILE strategies: {diff_bf16:.6f}")
    print(f"    Results {'identical' if diff_bf16 < 1e-6 else 'differ'}")


def test_rmsnorm_accuracy_diff():
    """RMSNorm with accuracy difference demonstrates bfloat16 vs float32 effects on the result"""
    print("Testing RMSNorm with varying accuracies...")
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
    
    print(f"  Precision impact: bfloat16 error is clear where float32 makes the difference negligible for this test")

if __name__ == "__main__":
    print("Batch Invariance Test")
    print("=" * 80)
    
    test_matmul()
    print()
    print("=" * 80)
    test_rmsnorm_invariant()
    print()
    print("=" * 80)
    test_rmsnorm_variant()
    print()
    print("=" * 80)
    test_rmsnorm_accuracy_diff()
    
    print("\n" + "=" * 80)
    print("SUMMARY:")
    print("  • MatMul: K_TILE variance - different reduction chunking")
    print("  • RMSNorm (standard): Batch-invariant - atomic reduction")
    print("  • RMSNorm (split): HIDDEN_TILE variance - reduction chunking")
    print("\nDone!")
