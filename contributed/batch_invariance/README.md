# NKI Batch Invariance Study

A study of batch invariance in Neuron Kernel Interface (NKI), replicating and extending [Thinking Machines' "Defeating Nondeterminism in LLM Inference"](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/) research.

## What is Batch Invariance?

Following [Thinking Machines' definition](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/):

**Batch invariance** requires:
1. **Run-to-run determinism**: Same prompt + same model + same inputs + same seed + same runtime config → bitwise-identical outputs across runs
2. **Batching independence**: Changing inference batching behavior (batch size, request packing, continuous batching order) → no output change

A batch-invariant system guarantees that the *way* you batch requests doesn't affect the numerical output—critical for reproducible LLM inference.

## Overview

This project demonstrates how different tile size configurations in NKI kernels can produce varying numerical results due to floating-point non-associativity. We test whether `nki.isa` operations maintain batch invariance when reduction tile sizes change—simulating what happens when a framework dynamically selects tile sizes based on input shape.

### Baselines Used

| Baseline Type | Purpose | Method |
|---------------|---------|--------|
| **CPU Reference** | Numerical parity | NKI kernel output vs PyTorch CPU (`torch.matmul`, manual RMSNorm) |
| **NKI Self-Baseline** | Run-to-run determinism | Same kernel, 1000 iterations, verify bitwise-identical outputs |
| **Tile Configuration Comparison** | Batching independence | Same kernel with different tile sizes (simulating shape-dependent selection) |

## Key Findings

### 1. Run-to-Run Determinism Confirmed

NKI ISA kernels produce bitwise-identical results across 1000 iterations with the same configuration.

### 2. Tile Size Invariance with `nki.isa`

**Critical finding**: `nki.isa` operations produce identical results regardless of tile size configuration in bfloat16 precision.

| Operation | K_TILE=128 vs K_TILE=64 | bfloat16 | float32 |
|-----------|-------------------------|----------|---------|
| **MatMul** | Tile invariant? | ✅ Yes (diff=0.0) | ✗ No (diff=6.1e-05) |
| **RMSNorm** | Tile invariant? | ✅ Yes (diff=0.0) | ✗ No (diff=2.4e-07) |

The bfloat16 invariance is the key result—reduced precision formats are where batch variance is most visible and problematic in practice, and ISA operations eliminate it entirely.

### 3. Historical Note: `nki.lang` Showed Variance

Prior to the NKI beta release, `nki.lang` operations exhibited tile-size-dependent variance:

| Operation | Kernel Type | float32 | bfloat16 | Amplification |
|-----------|-------------|---------|----------|---------------|
| **MatMul** | `nki.lang` | ✗ Variance (4.6e-05) | ✗ Variance (0.0078) | 170x |
| **RMSNorm** | `nki.lang` | ✗ Variance (3.6e-07) | ✗ Variance (0.0078) | 21,845x |

The bfloat16 amplification effect (errors 170-21,845x larger than float32) made variance highly visible in reduced precision formats. This behavior motivated the shift to `nki.isa` operations.

## How Tile Size Selection Can Break Batch Invariance

**The problem**: When reduction dimension tile sizes are selected based on input shape, the accumulation order changes. Due to floating-point non-associativity, different accumulation orders can produce different results:


(a + b) + c ≠ a + (b + c)  in finite precision

**Triton Split-K (Shape-Dependent)**:
python
num_pid_k ← tl.cdiv(k, block_k × split_k)  # Tile count varies with K dimension

**This Study's Simulation**:
Our kernels use a `deterministic` flag to compare two fixed tile configurations, simulating what happens when a framework chooses tile sizes based on input shape:

python
# MatMul kernel
if deterministic:
   K_TILE = 128                              # Fixed strategy
else:
   K_TILE = 64 if K <= 512 else 512          # Shape-dependent strategy

# RMSNorm kernel
HIDDEN_TILE = 128 if deterministic else 64    # Different accumulation granularity

**Why this matters**: If an inference framework selects tile sizes based on batch dimensions, then changing batch size changes accumulation order—potentially breaking batch invariance even though each individual run is deterministic.

## Test Methodology

### What Each Test Validates

| Test | Validates | Method |
|------|-----------|--------|
| `test_determinism()` | Run-to-run determinism | Same config → identical results across 1000 runs |
| `test_tiling_invariance()` | Tile size independence | K_TILE=128 vs K_TILE=64 → same results? |
| `test_matmul_parity()` | Numerical correctness | NKI output matches `torch.matmul` |
| `test_rmsnorm_parity()` | Numerical correctness | NKI output matches PyTorch RMSNorm reference |

### Tile Size Variance Demonstration

python
# Compare deterministic=True (K_TILE=128) vs deterministic=False (K_TILE=64)
out_k128 = nki_matmul_kernel_isa(a, b, deterministic=True)
out_k64  = nki_matmul_kernel_isa(a, b, deterministic=False)

diff = (out_k128 - out_k64).abs().max().item()
# With nki.isa: diff == 0.0 (batch invariant)

## Running the Tests

bash
cd contributed/batch_invariance
python test_batch_invariance.py

### Expected Output

1. **Determinism test**: 1000 iterations produce identical results
2. **Parity tests**: NKI kernels match PyTorch reference within tolerance
3. **Tiling invariance**: Different tile sizes produce identical results (diff=0.0)

## Project Structure


batch_invariance/
├── README.md                           # This document
├── test_batch_invariance.py            # Main test suite
└── kernels/
   ├── init.py
   ├── matmul_batch_invariant.py       # MatMul ISA implementation
   └── rmsnorm_batch_invariant.py      # RMSNorm ISA implementation

## Implications for LLM Inference

### For Deterministic Inference
- **Use `nki.isa` operations** for batch-invariant kernels
- **bfloat16 precision** works reliably with ISA operations
- **Fixed tile sizes** avoid shape-dependent variance (though ISA tolerates variation)

### Why This Matters
Batch invariance ensures that:
- Changing batch size doesn't change model outputs
- Request packing order doesn't affect results
- Continuous batching produces reproducible inference
- Debugging and testing become tractable

## Future Work

1. **Batch Invariant Attention**: Implement attention mechanisms using ISA operations
2. **LLM Integration**: Full forward pass comparison with varying batch configurations
3. **Performance Analysis**: Quantify any performance trade-offs with ISA approach
4. **Extended Precision Study**: Investigate fp16, int8 behavior

## Core Insight

**Batch invariance requires that accumulation order doesn't affect the final result.**

Our tile size comparison (K_TILE=128 vs K_TILE=64) simulates shape-dependent tiling. The finding that `nki.isa` operations produce identical results regardless of tile configuration demonstrates a path to deterministic LLM inference on Neuron hardware—even when batching configurations change.

## References

- [Thinking Machines: Defeating Nondeterminism in LLM Inference](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)
- [Thinking Machines GitHub: Batch Invariant Operations](https://github.com/thinking-machines-lab/batch_invariant_ops)
- [Meta: Triton Split-K Kernel Paper](https://scontent-dfw5-2.xx.fbcdn.net/v/t39.2365-6/418514147_782803483888724_2886980548537654804_n.pdf)
- [AWS Neuron Documentation](https://awsdocs-neuron.readthedocs-hosted.com/)
- [NKI Programming Guide](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/)

## Author

Implementation and analysis by Josh Longenecker, based on foundational work by Thinking Machines Lab.
