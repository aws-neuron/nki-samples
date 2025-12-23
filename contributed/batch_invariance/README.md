# NKI Batch Invariance: ISA vs Lang Kernels

Replicating [Thinking Machines' "Defeating Nondeterminism in LLM Inference"](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/) with a key discovery about `nki.isa` operations.

## Key Findings

### 1. Replicated the Paper: Batch Variance with `nki.lang`

The paper showed CUDA operations aren't batch-invariant due to dynamic reduction strategies. **We replicated this in NKI using `nki.lang` kernels:**

- **MatMul** (`nl.matmul`): Batch variance in both float32 and bfloat16
- **RMSNorm**: Batch variance in both float32 and bfloat16

### 2. Discovery: `nki.isa` Shows No Batch Variance in bfloat16

**Using `nki.isa` operations with the same dynamic reduction strategies:**

- **MatMul** (`nisa.nc_matmul`): Variance in float32, but **NO variance in bfloat16**
- **RMSNorm** (ISA operations): Variance in float32, but **NO variance in bfloat16**

## Results

| Operation | Kernel | bfloat16 | float32 |
|-----------|--------|----------|---------|
| **MatMul** | `nki.lang` | ✗ Variance | ✗ Variance |
| **MatMul** | `nki.isa` | ✓ **No Variance** | ✗ Variance |
| **RMSNorm** | `nki.lang` | ✗ Variance | ✗ Variance |
| **RMSNorm** | `nki.isa` | ✓ **No Variance** | ✗ Variance |

**Implication**: Use `nki.isa` operations for deterministic bfloat16 inference.

## Running the Test

```bash
cd contributed/batch_invariance
python test_batch_invariance.py
```

The test compares both kernel types with different K_TILE configurations and reports the differences in float32 vs bfloat16.

## Files

- `kernels/matmul_batch_invariant.py` - MatMul implementations (lang and ISA)
- `test_batch_invariance.py` - Test comparing both kernel types
- `README.md` - This document

## References

- [Thinking Machines: Defeating Nondeterminism in LLM Inference](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)
- [AWS Neuron Documentation](https://awsdocs-neuron.readthedocs-hosted.com/)
- [NKI Programming Guide](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/)
