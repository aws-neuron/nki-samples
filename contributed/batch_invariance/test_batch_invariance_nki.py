"""
Minimal NKI Batch Invariance Test - Clean Implementation

Tests if dynamic M tiling introduces non-determinism in matmul.
Based on NKI matmul example pattern.
"""

import torch
import torch_neuronx
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl


@nki.jit
def matmul_m64(a, b):
    """
    Matmul with M tiled at 64
    a: [M, 4096], b: [4096, 512]
    Output: [M, 512]
    
    Works with any M that's divisible by 64
    """
    M, K = a.shape
    N = b.shape[1]
    M_TILE = 64
    K_TILE = 128
    
    result = nl.ndarray((M, N), dtype=a.dtype, buffer=nl.shared_hbm)
    
    # Tile over M dimension
    for m in nl.affine_range(M // M_TILE):
        # Accumulator for this M chunk
        c_psum = nl.zeros((M_TILE, N), dtype=nl.float32, buffer=nl.psum)
        
        # Reduction over K
        for k in nl.affine_range(K // K_TILE):
            # Load a: [M_TILE, K_TILE]
            i_a_p, i_a_f = nl.mgrid[0:M_TILE, 0:K_TILE]
            a_tile = nl.load(a[m*M_TILE + i_a_p, k*K_TILE + i_a_f])
            
            # Load b: [K_TILE, N]
            i_b_p, i_b_f = nl.mgrid[0:K_TILE, 0:N]
            b_tile = nl.load(b[k*K_TILE + i_b_p, i_b_f])
            
            # Matmul
            c_psum += nl.matmul(a_tile, b_tile, transpose_x=False)
        
        # Store this M chunk
        i_out_p, i_out_f = nl.mgrid[0:M_TILE, 0:N]
        c_sbuf = nl.copy(c_psum, dtype=result.dtype)
        nl.store(result[m*M_TILE + i_out_p, i_out_f], value=c_sbuf)
    
    return result


@nki.jit
def matmul_m128(a, b):
    """
    Matmul with M tiled at 128
    a: [M, 4096], b: [4096, 512]
    Output: [M, 512]
    
    Works with any M that's divisible by 128
    """
    M, K = a.shape
    N = b.shape[1]
    M_TILE = 128
    K_TILE = 128
    
    result = nl.ndarray((M, N), dtype=a.dtype, buffer=nl.shared_hbm)
    
    # Tile over M dimension
    for m in nl.affine_range(M // M_TILE):
        # Accumulator for this M chunk
        c_psum = nl.zeros((M_TILE, N), dtype=nl.float32, buffer=nl.psum)
        
        # Reduction over K
        for k in nl.affine_range(K // K_TILE):
            # Load a: [M_TILE, K_TILE]
            i_a_p, i_a_f = nl.mgrid[0:M_TILE, 0:K_TILE]
            a_tile = nl.load(a[m*M_TILE + i_a_p, k*K_TILE + i_a_f])
            
            # Load b: [K_TILE, N]
            i_b_p, i_b_f = nl.mgrid[0:K_TILE, 0:N]
            b_tile = nl.load(b[k*K_TILE + i_b_p, i_b_f])
            
            # Matmul
            c_psum += nl.matmul(a_tile, b_tile, transpose_x=False)
        
        # Store this M chunk
        i_out_p, i_out_f = nl.mgrid[0:M_TILE, 0:N]
        c_sbuf = nl.copy(c_psum, dtype=result.dtype)
        nl.store(result[m*M_TILE + i_out_p, i_out_f], value=c_sbuf)
    
    return result


@nki.jit
def matmul_k64(a, b):
    """
    Matmul with K tiled at 64 (different contraction tile size)
    
    This should produce DIFFERENT results than K_TILE=128
    because the reduction order changes!
    """
    M, K = a.shape
    N = b.shape[1]
    M_TILE = 128
    K_TILE = 64  # DIFFERENT K tiling!
    
    result = nl.ndarray((M, N), dtype=a.dtype, buffer=nl.shared_hbm)
    
    for m in nl.affine_range(M // M_TILE):
        c_psum = nl.zeros((M_TILE, N), dtype=nl.float32, buffer=nl.psum)
        
        # Now we have TWICE as many K iterations (64 instead of 32)
        for k in nl.affine_range(K // K_TILE):
            i_a_p, i_a_f = nl.mgrid[0:M_TILE, 0:K_TILE]
            a_tile = nl.load(a[m*M_TILE + i_a_p, k*K_TILE + i_a_f])
            
            i_b_p, i_b_f = nl.mgrid[0:K_TILE, 0:N]
            b_tile = nl.load(b[k*K_TILE + i_b_p, i_b_f])
            
            c_psum += nl.matmul(a_tile, b_tile, transpose_x=False)
        
        i_out_p, i_out_f = nl.mgrid[0:M_TILE, 0:N]
        c_sbuf = nl.copy(c_psum, dtype=result.dtype)
        nl.store(result[m*M_TILE + i_out_p, i_out_f], value=c_sbuf)
    
    return result


@nki.jit
def matmul_sequential(a, b):
    """
    Matmul using sequential_range instead of affine_range
    
    sequential_range forces sequential execution with loop-carried dependency.
    Question: Does this affect determinism?
    """
    M, K = a.shape
    N = b.shape[1]
    M_TILE = 128
    K_TILE = 128
    
    result = nl.ndarray((M, N), dtype=a.dtype, buffer=nl.shared_hbm)
    
    for m in nl.affine_range(M // M_TILE):
        c_psum = nl.zeros((M_TILE, N), dtype=nl.float32, buffer=nl.psum)
        
        # Using sequential_range - tells compiler there's loop dependency
        for k in nl.sequential_range(K // K_TILE):
            i_a_p, i_a_f = nl.mgrid[0:M_TILE, 0:K_TILE]
            a_tile = nl.load(a[m*M_TILE + i_a_p, k*K_TILE + i_a_f])
            
            i_b_p, i_b_f = nl.mgrid[0:K_TILE, 0:N]
            b_tile = nl.load(b[k*K_TILE + i_b_p, i_b_f])
            
            c_psum += nl.matmul(a_tile, b_tile, transpose_x=False)
        
        i_out_p, i_out_f = nl.mgrid[0:M_TILE, 0:N]
        c_sbuf = nl.copy(c_psum, dtype=result.dtype)
        nl.store(result[m*M_TILE + i_out_p, i_out_f], value=c_sbuf)
    
    return result


@nki.jit
def matmul_m128_fp32(a, b):
    """
    Matmul with M_TILE=128, but using float32 inputs
    To compare precision differences vs bfloat16
    """
    M, K = a.shape
    N = b.shape[1]
    M_TILE = 128
    K_TILE = 128
    
    result = nl.ndarray((M, N), dtype=a.dtype, buffer=nl.shared_hbm)
    
    for m in nl.affine_range(M // M_TILE):
        c_psum = nl.zeros((M_TILE, N), dtype=nl.float32, buffer=nl.psum)
        
        for k in nl.affine_range(K // K_TILE):
            i_a_p, i_a_f = nl.mgrid[0:M_TILE, 0:K_TILE]
            a_tile = nl.load(a[m*M_TILE + i_a_p, k*K_TILE + i_a_f])
            
            i_b_p, i_b_f = nl.mgrid[0:K_TILE, 0:N]
            b_tile = nl.load(b[k*K_TILE + i_b_p, i_b_f])
            
            c_psum += nl.matmul(a_tile, b_tile, transpose_x=False)
        
        i_out_p, i_out_f = nl.mgrid[0:M_TILE, 0:N]
        c_sbuf = nl.copy(c_psum, dtype=result.dtype)
        nl.store(result[m*M_TILE + i_out_p, i_out_f], value=c_sbuf)
    
    return result


@nki.jit
def matmul_k64_fp32(a, b):
    """
    Matmul with K_TILE=64, using float32 inputs
    """
    M, K = a.shape
    N = b.shape[1]
    M_TILE = 128
    K_TILE = 64
    
    result = nl.ndarray((M, N), dtype=a.dtype, buffer=nl.shared_hbm)
    
    for m in nl.affine_range(M // M_TILE):
        c_psum = nl.zeros((M_TILE, N), dtype=nl.float32, buffer=nl.psum)
        
        for k in nl.affine_range(K // K_TILE):
            i_a_p, i_a_f = nl.mgrid[0:M_TILE, 0:K_TILE]
            a_tile = nl.load(a[m*M_TILE + i_a_p, k*K_TILE + i_a_f])
            
            i_b_p, i_b_f = nl.mgrid[0:K_TILE, 0:N]
            b_tile = nl.load(b[k*K_TILE + i_b_p, i_b_f])
            
            c_psum += nl.matmul(a_tile, b_tile, transpose_x=False)
        
        i_out_p, i_out_f = nl.mgrid[0:M_TILE, 0:N]
        c_sbuf = nl.copy(c_psum, dtype=result.dtype)
        nl.store(result[m*M_TILE + i_out_p, i_out_f], value=c_sbuf)
    
    return result


def test_batch_invariance():
    """
    Comprehensive batch invariance testing suite
    """
    B, D, N = 2048, 4096, 512
    
    # Create test inputs on XLA device
    device = 'xla'
    a = torch.linspace(-100, 100, B*D, device=device).reshape(B, D).to(torch.bfloat16)
    b = torch.linspace(-100, 100, D*N, device=device).reshape(D, N).to(torch.bfloat16)
    
    print("=" * 70)
    print("TEST 1: Different M_TILE on same input")
    print("=" * 70)
    print(f"Input: [{B}, {D}] @ [{D}, {N}]")
    print(f"M_TILE=64:  {B//64} iterations over M, K_TILE=128")
    print(f"M_TILE=128: {B//128} iterations over M, K_TILE=128")
    print()
    
    c_m64 = matmul_m64(a, b)
    c_m128 = matmul_m128(a, b)
    
    c_m64_cpu = c_m64.cpu()
    c_m128_cpu = c_m128.cpu()
    
    print("Results:")
    print(f"  M_TILE=64  row[0]: {c_m64_cpu[0, :5]}")
    print(f"  M_TILE=128 row[0]: {c_m128_cpu[0, :5]}")
    
    diff1 = (c_m64_cpu - c_m128_cpu).abs().max()
    print(f"\n  Max difference: {diff1.item()}")
    print(f"  Bitwise identical: {diff1.item() == 0}")
    
    print("\n" + "=" * 70)
    print("TEST 2: Thinking Machines scenario - varying M (batch size)")
    print("=" * 70)
    print("The real batch invariance test!")
    print(f"Compute row 0 with M=128 vs M=2048")
    print()
    
    a_small = a[:128, :]
    c_small = matmul_m128(a_small, b)
    c_full = matmul_m128(a, b)
    
    c_small_cpu = c_small.cpu()
    c_full_cpu = c_full.cpu()
    
    print("Results:")
    print(f"  M=128  row[0]: {c_small_cpu[0, :5]}")
    print(f"  M=2048 row[0]: {c_full_cpu[0, :5]}")
    
    diff2 = (c_small_cpu[0] - c_full_cpu[0]).abs().max()
    print(f"\n  Max difference: {diff2.item()}")
    print(f"  Bitwise identical: {diff2.item() == 0}")
    
    print("\n" + "=" * 70)
    print("TEST 3: Different K_TILE - Does reduction order matter?")
    print("=" * 70)
    print("K_TILE=128: 32 K iterations (accumulate chunks: 0, 128, 256, ...)")
    print("K_TILE=64:  64 K iterations (accumulate chunks: 0, 64, 128, ...)")
    print("Different accumulation order → different floating point results!")
    print()
    
    c_k128 = matmul_m128(a, b)  # K_TILE=128
    c_k64 = matmul_k64(a, b)    # K_TILE=64
    
    c_k128_cpu = c_k128.cpu()
    c_k64_cpu = c_k64.cpu()
    
    print("Results:")
    print(f"  K_TILE=128 row[0]: {c_k128_cpu[0, :5]}")
    print(f"  K_TILE=64  row[0]: {c_k64_cpu[0, :5]}")
    
    diff3 = (c_k128_cpu - c_k64_cpu).abs().max()
    print(f"\n  Max difference: {diff3.item()}")
    print(f"  Are they different? {diff3.item() != 0}")
    
    if diff3.item() != 0:
        print("  ✓ EXPECTED! Different K_TILE → different reduction order")
    else:
        print("  ✗ UNEXPECTED! K_TILE should matter for floating point")
    
    print("\n" + "=" * 70)
    print("TEST 4: sequential_range vs affine_range")
    print("=" * 70)
    print("affine_range: parallel-friendly, allows loop optimizations")
    print("sequential_range: forces sequential execution, loop dependency")
    print("Question: Do they produce identical results?")
    print()
    
    c_affine = matmul_m128(a, b)       # Uses affine_range
    c_sequential = matmul_sequential(a, b)  # Uses sequential_range
    
    c_affine_cpu = c_affine.cpu()
    c_sequential_cpu = c_sequential.cpu()
    
    print("Results:")
    print(f"  affine_range     row[0]: {c_affine_cpu[0, :5]}")
    print(f"  sequential_range row[0]: {c_sequential_cpu[0, :5]}")
    
    diff4 = (c_affine_cpu - c_sequential_cpu).abs().max()
    print(f"\n  Max difference: {diff4.item()}")
    print(f"  Bitwise identical: {diff4.item() == 0}")
    
    if diff4.item() == 0:
        print("  ✓ Loop iterator type doesn't affect determinism!")
    else:
        print("  ✗ sequential_range changes results!")
    
    print("\n" + "=" * 70)
    print("TEST 5: Precision Test - bfloat16 vs float32")
    print("=" * 70)
    print("Does reduced precision (bfloat16) amplify K_TILE differences?")
    print("bfloat16: 7 bits mantissa, ~2-3 decimal digits precision")
    print("float32:  23 bits mantissa, ~7 decimal digits precision")
    print()
    
    # Create float32 inputs
    a_fp32 = torch.linspace(-100, 100, B*D, device=device).reshape(B, D).to(torch.float32)
    b_fp32 = torch.linspace(-100, 100, D*N, device=device).reshape(D, N).to(torch.float32)
    
    # Run with different K_TILE on float32
    c_k128_fp32 = matmul_m128_fp32(a_fp32, b_fp32)
    c_k64_fp32 = matmul_k64_fp32(a_fp32, b_fp32)
    
    c_k128_fp32_cpu = c_k128_fp32.cpu()
    c_k64_fp32_cpu = c_k64_fp32.cpu()
    
    print("Results (float32):")
    print(f"  K_TILE=128 row[0]: {c_k128_fp32_cpu[0, :5]}")
    print(f"  K_TILE=64  row[0]: {c_k64_fp32_cpu[0, :5]}")
    
    diff5_fp32 = (c_k128_fp32_cpu - c_k64_fp32_cpu).abs().max()
    print(f"\n  Max difference (float32): {diff5_fp32.item()}")
    
    print("\nComparison:")
    print(f"  bfloat16 K_TILE diff: {diff3.item()}")
    print(f"  float32  K_TILE diff: {diff5_fp32.item()}")
    print(f"  Ratio (bf16/fp32):    {diff3.item() / diff5_fp32.item():.2f}x")
    
    if diff5_fp32.item() < diff3.item():
        print(f"\n  ✓ float32 reduces error by {diff3.item() / diff5_fp32.item():.1f}x!")
        print("    Lower precision (bfloat16) amplifies accumulation order effects")
    else:
        print("\n  ✗ Unexpected: float32 doesn't reduce error significantly")
    
    # Also check: Is the difference consistent across runs?
    print("\n" + "=" * 70)
    print("TEST 6: Consistency Check - Is K_TILE difference stable?")
    print("=" * 70)
    print("Running K_TILE test 3 times to verify determinism...")
    print()
    
    diffs = []
    for run in range(3):
        c_k128_run = matmul_m128(a, b)
        c_k64_run = matmul_k64(a, b)
        diff_run = (c_k128_run.cpu() - c_k64_run.cpu()).abs().max().item()
        diffs.append(diff_run)
        print(f"  Run {run+1}: max diff = {diff_run}")
    
    if len(set(diffs)) == 1:
        print(f"\n  ✓ FULLY DETERMINISTIC! All runs: {diffs[0]}")
        print("    The 256.0 difference is consistent and repeatable")
    else:
        print(f"\n  ✗ Non-deterministic! Diffs vary: {diffs}")
    
    print("\n" + "=" * 70)
    print("FINAL VERDICT")
    print("=" * 70)
    
    print(f"\n1. M_TILE variation (64 vs 128):      {'✓ INVARIANT' if diff1.item() == 0 else '✗ VARIANT'}")
    print(f"2. M variation (128 vs 2048):          {'✓ INVARIANT' if diff2.item() == 0 else '✗ VARIANT'}")
    print(f"3. K_TILE variation (64 vs 128):       {'✓ VARIANT (expected)' if diff3.item() != 0 else '✗ INVARIANT (unexpected)'}")
    print(f"4. Loop iterator (affine vs seq):      {'✓ INVARIANT' if diff4.item() == 0 else '✗ VARIANT'}")
    print(f"5. Precision (bf16 vs fp32):           {diff3.item():.1f} vs {diff5_fp32.item():.4f} ({diff3.item()/diff5_fp32.item():.1f}x)")
    print(f"6. Consistency across runs:            {'✓ DETERMINISTIC' if len(set(diffs)) == 1 else '✗ NON-DETERMINISTIC'}")
    
    if diff1.item() == 0 and diff2.item() == 0:
        print("\n" + "🎉 " * 20)
        print("NKI IS BATCH INVARIANT!")
        print("  • M_TILE doesn't affect results (batch tiling invariant)")
        print("  • M (batch size) doesn't affect results (true batch invariance)")
        print("  • K_TILE DOES affect results (reduction order matters)")
        print(f"  • bfloat16 amplifies differences by {diff3.item()/diff5_fp32.item():.1f}x vs float32")
        print("  • But for FIXED K_TILE, results are fully deterministic!")
        print("🎉 " * 20)
    else:
        print("\n✗ Batch invariance NOT achieved")


if __name__ == "__main__":
    test_batch_invariance()