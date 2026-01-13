# NKI Batch Invariance Study

A comprehensive study of batch invariance in Neuron Kernel Interface (NKI), replicating and extending [Thinking Machines' "Defeating Nondeterminism in LLM Inference"](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/) research.

## Overview

This project demonstrates how different NKI kernel implementations (`nki.lang` vs `nki.isa`) exhibit varying degrees of batch invariance, particularly when using reduced precision formats like bfloat16.

## Key Findings

### 1. Batch Variance Occurs When Reduction Strategies Are Dynamic

**Confirmed the core hypothesis**: Batch variance emerges when tile sizes for reduction dimensions are determined dynamically based on input shapes, exactly as described in the original paper.

### 2. Precision Choice Dramatically Affects Variance Visibility

Our testing revealed significant amplification effects:
- **MatMul (Lang)**: bfloat16 errors are **170x larger** than float32
- **RMSNorm (Lang)**: bfloat16 errors are **21,845x larger** than float32

### 3. NKI ISA Operations Show Superior Batch Invariance

**Critical Discovery**: `nki.isa` operations demonstrate batch invariance in bfloat16 precision where `nki.lang` operations show variance.

| Operation | Kernel Type | float32 | bfloat16 | Amplification |
|-----------|-------------|---------|----------|---------------|
| **MatMul** | `nki.lang` | ✗ Variance (4.6e-05) | ✗ Variance (0.0078) | 170.7x |
| **MatMul** | `nki.isa` | ✗ Variance (6.1e-05) | ✅ **Invariant** (0.0000) | 0.0x |
| **RMSNorm** | `nki.lang` | ✗ Variance (3.6e-07) | ✗ Variance (0.0078) | 21,845x |
| **RMSNorm** | `nki.isa` | ✗ Variance (3.6e-07) | ✅ **Invariant** (0.0000) | 0.0x |

### 4. NKI Design Patterns Naturally Promote Batch Invariance

NKI best practices emphasize static tile sizes, which inherently avoid batch variance. However, the framework doesn't prevent variance when dynamic strategies are implemented.

## Technical Analysis

### Dynamic vs Static Tiling Strategies

**Triton Split-K Approach** (Dynamic):
```python
num_pid_k ← tl.cdiv(k, block_k × split_k)  # Shape-dependent
```

**NKI Standard Approach** (Static):
```python
# Fixed tile sizes regardless of input shape
TILES_IN_BLOCK_K = 4  # Static configuration
```

### Variance Demonstration

The same kernel with different K-tile configurations produces different results:

```python
# Different K-blocking strategies → different accumulation order
result_1 = nki_matmul(lhs, rhs, TILES_IN_BLOCK_K=4)
result_2 = nki_matmul(lhs, rhs, TILES_IN_BLOCK_K=8)

# Results differ due to floating-point non-associativity
max_diff_bfloat16 = 4.000000    # Significant difference
max_diff_float32 = 0.000244     # Smaller but still present
```

## Experimental Results

### Test Configuration
- **Matrix dimensions**: [256, 512] @ [512, 512] = [256, 512]
- **Precision formats**: float32, bfloat16
- **Kernel variants**: Lang (`nl.matmul`, `nl.sum`) vs ISA (`nisa.nc_matmul`, `nisa.tensor_reduce`)

### Batch Variance Summary

```
                          kernel  float32_error  bfloat16_error  amplification
                Lang (nl.matmul)   4.577637e-05        0.007812     170.666667
            ISA (nisa.nc_matmul)   6.103516e-05        0.000000       0.000000
           RMSNorm Lang (nl.sum)   3.576279e-07        0.007812   21845.333333
RMSNorm ISA (nisa.tensor_reduce)   3.576279e-07        0.000000       0.000000
```

## Implications for LLM Inference

### For Deterministic Inference
- **Use `nki.isa` operations** when batch invariance is critical
- **Choose bfloat16 precision** with ISA kernels for deterministic results
- **Implement static tiling strategies** to avoid shape-dependent variance

### For Performance vs Determinism Trade-offs
- `nki.lang` operations may offer performance benefits but sacrifice determinism
- `nki.isa` operations provide determinism at potential performance cost
- Precision choice significantly impacts the visibility of non-deterministic behavior

## Running the Tests

```bash
cd contributed/batch_invariance
python test_batch_invariance.py
```

### Expected Output
The test will show:
1. **Correctness verification**: Both kernels match PyTorch reference
2. **Batch variance analysis**: Comparison of different tiling strategies
3. **Precision impact**: Amplification effects between float32 and bfloat16

## Project Structure

```
batch_invariance/
├── README.md                           # This document
├── test_batch_invariance.py           # Main test suite
└── kernels/
    ├── __init__.py
    ├── matmul_batch_invariant.py      # MatMul implementations (Lang & ISA)
    └── rmsnorm_batch_invariant.py     # RMSNorm implementations (Lang & ISA)
```

## Future Work

1. **Batch Invariant Attention**: Implement attention mechanisms using ISA operations
2. **LLM Integration**: Compare standard NeuronLlama vs BatchInvariantLlama in full forward pass
3. **Performance Analysis**: Quantify performance trade-offs between Lang and ISA approaches
4. **Extended Precision Study**: Investigate other precision formats (fp16, int8)

## Core Insight

**Batch invariance is fundamentally a design choice, not a framework limitation.** While NKI's design patterns naturally encourage batch-invariant implementations through static tiling, the framework itself doesn't prevent variance when dynamic strategies are employed.

The discovery that `nki.isa` operations maintain batch invariance in bfloat16 precision provides a clear path for deterministic LLM inference on Neuron hardware.

## References

- [Thinking Machines: Defeating Nondeterminism in LLM Inference](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)
- [Thinking Machines GitHub: Batch Invariant Operations](https://github.com/thinking-machines-lab/batch_invariant_ops)
- [Meta: Triton Split-K Kernel Paper](https://scontent-dfw5-2.xx.fbcdn.net/v/t39.2365-6/418514147_782803483888724_2886980548537654804_n.pdf)
- [AWS Neuron Documentation](https://awsdocs-neuron.readthedocs-hosted.com/)
- [NKI Programming Guide](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/)

## Author

Implementation and analysis by Josh Longenecker based on the foundational work by Thinking Machines Lab.
