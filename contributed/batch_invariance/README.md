# NKI Batch Invariance Test

Testing whether NKI's tile size constraints protect against batch-dependent non-determinism in matrix multiplication.

## Hypothesis

**NKI achieves batch invariance by default due to hardware tile constraints.**

Unlike CUDA/PyTorch, where batch size can influence the K-dimension reduction strategy (e.g., switching to split-K for better parallelism when M is small), NKI's hardware constraints enforce fixed tile sizes that decouple batch size from reduction order.

### Key Protection Mechanisms

1. **K is the reduction axis, not the batch axis (M)**
   - Reduction happens over K (contraction dimension)
   - M (batch) loop is outer, K loop is inner
   - Changing M doesn't affect K iteration count

2. **Hardware constraints enforce fixed tile sizes**
   - Tensor Engine limits: P-dim ≤ 128, free-dim ≤ 512
   - Forces compile-time constants (e.g., K_TILE=128)
   - Prevents runtime adaptation based on batch size

3. **Potential vulnerability: Split-K**
   - NKI *could* split along K when M is small (like CUDA does)
   - This would couple M and K reduction strategy
   - Our tests verify this doesn't happen automatically

## Test Design

Replicated [Thinking Machines' batch invariance test](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/):

Instance_type: `inf2.xlarge`
AMI ID: `ami-0ec4ab14b1c5a10f2`
AMI NAME: `Deep Learning AMI Neuron (Ubuntu 22.04) 20250919` 
```python
# CUDA shows non-determinism:
out1 = torch.mm(a[:1], b)      # M=1
out2 = torch.mm(a, b)[:1]      # M=2048
# Result: out1 ≠ out2 (diff: 1669.25)

# NKI test:
out1 = matmul_nki(a[:128], b)[0]   # M=128
out2 = matmul_nki(a, b)[0]         # M=2048
# Result: out1 == out2 (diff: 0.0) ✓
```

## Results

### Test 1: M_TILE Variation (64 vs 128)
```
M_TILE=64  → Result: [9664., 9600., ...]
M_TILE=128 → Result: [9664., 9600., ...]
Max difference: 0.0 ✓ INVARIANT
```
**Conclusion:** Batch tiling strategy doesn't affect results.

### Test 2: M (Batch Size) Variation (128 vs 2048)
```
M=128  → Result: [9664., 9600., ...]
M=2048 → Result: [9664., 9600., ...]
Max difference: 0.0 ✓ INVARIANT
```
**Conclusion:** True batch invariance achieved. Same element produces identical results regardless of batch size.

### Test 3: K_TILE Variation (64 vs 128) - Simulated Dynamic Tiling
```
K_TILE=128 → Result: [9664., 9600., ...]  (32 iterations)
K_TILE=64  → Result: [9664., 9600., ...]  (64 iterations)
Max difference: 256.0 ✓ VARIANT (expected)
```
**Conclusion:** Reduction order matters. Different K_TILE → different accumulation order → different floating-point results. This simulates what CUDA does when it adapts K strategy based on batch size.

### Test 4: Loop Iterator (affine_range vs sequential_range)
```
affine_range     → Result: [9664., 9600., ...]
sequential_range → Result: [9664., 9600., ...]
Max difference: 0.0 ✓ INVARIANT
```
**Conclusion:** Loop iterator type is a compiler hint; doesn't affect numerical output.

### Test 5: Precision Impact (bfloat16 vs float32)
```
bfloat16 K_TILE diff: 256.0    (2.67% relative error)
float32  K_TILE diff: 15.125   (0.091% relative error)
Amplification: 16.9x
```
**Conclusion:** Lower precision amplifies accumulation order effects. bfloat16's 7-bit mantissa shows 17x larger differences than float32's 23-bit mantissa.

### Test 6: Consistency Check
```
Run 1: 256.0
Run 2: 256.0
Run 3: 256.0
✓ FULLY DETERMINISTIC
```
**Conclusion:** The K_TILE difference is consistent and repeatable, not random.

## Key Findings

### ✅ Hypothesis Confirmed

**NKI IS BATCH INVARIANT**
- M_TILE doesn't affect results (batch tiling invariant)
- M (batch size) doesn't affect results (true batch invariance)
- K_TILE DOES affect results (reduction order matters)
- But K_TILE is a compile-time constant → fully deterministic

### 📊 Comparison: NKI vs CUDA

| Aspect | CUDA | NKI |
|--------|------|-----|
| Batch size affects K reduction? | ✗ Yes (split-K adaptation) | ✅ No (fixed K_TILE) |
| Run-to-run deterministic? | ✗ No (varies ~1669) | ✅ Yes (always identical) |
| K_TILE matters? | ✅ Yes | ✅ Yes |
| Tile size constraints? | Flexible | Hardware-enforced (≤128/512) |

### 🔬 Why NKI Wins

1. **M/K decoupling:** Batch loop (M) is outer, reduction loop (K) is inner. Changing batch size doesn't affect K iteration count.

2. **Hardware constraints as a feature:** Tensor Engine limits force compile-time K_TILE constants, preventing runtime adaptation.

3. **No automatic split-K:** NKI doesn't dynamically switch to split-K based on batch size. You'd need to write a separate kernel.

## Implications

**For LLM Inference:**
- Batch-invariant by default (no special kernels needed like Thinking Machines built for CUDA)
- Deterministic sampling at temperature=0 (if K_TILE is fixed)
- True on-policy RL possible (identical numerics between training and inference)

**Caveats:**
- K_TILE variation causes 2.67% relative error in bfloat16 (acceptable for most LLM use cases)
- Must use consistent K_TILE across kernels for bitwise reproducibility
- Lower precision (bfloat16) amplifies accumulation order effects 17x vs float32

## Conclusion

NKI's tile size constraints, enforced by hardware limitations, provide batch invariance as an inherent property rather than requiring specialized implementations. The decoupling of batch size (M) from reduction strategy (K_TILE) ensures that the same element produces identical results regardless of the batch it's computed in.

**Bottom line:** CUDA varies K reduction order *unpredictably* based on batch size. NKI keeps it *fixed* based on compile-time K_TILE. That's the win.
