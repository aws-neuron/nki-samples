"""
Simple Batch Invariance Test
"""

import torch
import torch_neuronx
import numpy as np
from kernels.rmsnorm_batch_invariant import nki_rmsnorm_kernel
from kernels.rmsnorm_split_reduction import nki_rmsnorm_split_reduction
from kernels.matmul_batch_invariant import nki_matmul_kernel as matmul_batch_invariant


def test_matmul():
    """MatMul test showing K_TILE effect and precision impact"""
    print("Testing MatMul batch invariance...")
    
    device = 'xla'
    M, K, N = 128, 512, 512  # K=512 triggers different behavior!
    
    print(f"  K={K} -> batch_invariant=True: K_TILE=128, batch_invariant=False: K_TILE=64")
    print()
    
    # Test with bfloat16
    print("  Testing with bfloat16:")
    a_bf16 = torch.linspace(-1, 1, M * K, device=device).reshape(M, K).to(torch.bfloat16)
    b_bf16 = torch.linspace(-1, 1, K * N, device=device).reshape(K, N).to(torch.bfloat16)
    
    result_inv_bf16 = matmul_batch_invariant(a_bf16, b_bf16, batch_invariant=True)   # K_TILE=128
    result_var_bf16 = matmul_batch_invariant(a_bf16, b_bf16, batch_invariant=False)  # K_TILE=64
    
    diff_bf16 = torch.max(torch.abs(result_inv_bf16 - result_var_bf16)).item()
    print(f"    Max difference between K_TILE strategies: {diff_bf16:.6f}")
    print(f"    Results {'identical' if diff_bf16 < 1e-6 else 'differ'}")
    
    print()
    
    # Test with float32
    print("  Testing with float32:")
    a_f32 = torch.linspace(-1, 1, M * K, device=device).reshape(M, K).to(torch.float32)
    b_f32 = torch.linspace(-1, 1, K * N, device=device).reshape(K, N).to(torch.float32)
    
    result_inv_f32 = matmul_batch_invariant(a_f32, b_f32, batch_invariant=True)   # K_TILE=128
    result_var_f32 = matmul_batch_invariant(a_f32, b_f32, batch_invariant=False)  # K_TILE=64
    
    diff_f32 = torch.max(torch.abs(result_inv_f32 - result_var_f32)).item()
    print(f"    Max difference between K_TILE strategies: {diff_f32:.6f}")
    print(f"    Results {'identical' if diff_f32 < 1e-6 else 'differ'}")
    
    print()
    print(f"  Precision impact: bfloat16 error is {diff_bf16/diff_f32 if diff_f32 > 0 else 'N/A'}x larger than float32")
    print(f"  This demonstrates how reduced precision amplifies tiling strategy effects")
    

def test_rmsnorm():
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


def test_rmsnorm_split_reduction():
    """RMSNorm with SPLIT REDUCTION demonstrates TRUE batch VARIANCE"""
    print("Testing RMSNorm with Split Reduction...")
    print("  (Tiling the HIDDEN dimension creates different accumulation orders)")
    
    device = 'xla'
    hidden_dim = 512  # Use 512 to see clear difference
    batch_size = 64
    
    print(f"  hidden_dim={hidden_dim}")
    print(f"    batch_invariant=True:  HIDDEN_TILE=256 (2 chunks, 1 accumulation)")
    print(f"    batch_invariant=False: HIDDEN_TILE=128 (4 chunks, 3 accumulations)")
    print()
    
    # Test with bfloat16
    print("  Testing with bfloat16:")
    a_bf16 = torch.linspace(-1, 1, batch_size * hidden_dim, device=device).reshape(batch_size, hidden_dim).to(torch.bfloat16)
    g_bf16 = torch.ones(hidden_dim, device=device, dtype=torch.bfloat16)
    
    result_inv_bf16 = nki_rmsnorm_split_reduction(a_bf16, g_bf16, batch_invariant=True)   # HIDDEN_TILE=256
    result_var_bf16 = nki_rmsnorm_split_reduction(a_bf16, g_bf16, batch_invariant=False)  # HIDDEN_TILE=128
    
    diff_bf16 = torch.max(torch.abs(result_inv_bf16 - result_var_bf16)).item()
    print(f"    Max difference between HIDDEN_TILE strategies: {diff_bf16:.6f}")
    print(f"    Results {'identical' if diff_bf16 < 1e-6 else 'differ'}")
    
    print()
    
    # Test with float32
    print("  Testing with float32:")
    a_f32 = torch.linspace(-1, 1, batch_size * hidden_dim, device=device).reshape(batch_size, hidden_dim).to(torch.float32)
    g_f32 = torch.ones(hidden_dim, device=device, dtype=torch.float32)
    
    result_inv_f32 = nki_rmsnorm_split_reduction(a_f32, g_f32, batch_invariant=True)   # HIDDEN_TILE=256
    result_var_f32 = nki_rmsnorm_split_reduction(a_f32, g_f32, batch_invariant=False)  # HIDDEN_TILE=128
    
    diff_f32 = torch.max(torch.abs(result_inv_f32 - result_var_f32)).item()
    print(f"    Max difference between HIDDEN_TILE strategies: {diff_f32:.6f}")
    print(f"    Results {'identical' if diff_f32 < 1e-6 else 'differ'}")
    
    print()
    print(f"  Precision impact: bfloat16 error is {diff_bf16/diff_f32 if diff_f32 > 0 else 'N/A'}x larger than float32")
    print(f"  ✓ Split reduction creates batch variance in BOTH precisions!")
    print(f"    Different hidden tile sizes → different accumulation order")
    print(f"    This is analogous to MatMul's K_TILE effect")


if __name__ == "__main__":
    print("Batch Invariance Test")
    print("=" * 80)
    
    test_matmul()
    print()
    print("=" * 80)
    test_rmsnorm()
    print()
    print("=" * 80)
    test_rmsnorm_split_reduction()
    
    print("\n" + "=" * 80)
    print("SUMMARY:")
    print("  • MatMul: K_TILE variance - different reduction chunking")
    print("  • RMSNorm (standard): Batch-invariant - atomic reduction")
    print("  • RMSNorm (split): HIDDEN_TILE variance - reduction chunking")
    print("\nDone!")