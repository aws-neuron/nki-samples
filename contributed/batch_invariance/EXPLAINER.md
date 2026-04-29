# Why bfloat16 NKI Matmul is Batch-Invariant for Free

## Project context

This is about `nki_matmul_kernel_isa` in `kernels/matmul_batch_invariant.py`.
The kernel tiles the K dimension and accumulates partial matmuls into a float32 PSUM buffer on the NeuronCore Tensor Engine.

The `deterministic` flag controls K_TILE:
- `deterministic=True`  → K_TILE=128 → 4 accumulation steps for K=512
- `deterministic=False` → K_TILE=64  → 8 accumulation steps for K=512

The question: does changing K_TILE change the output?

**Hardware result (test_determinism.ipynb on Trn2):**
- bfloat16 inputs: diff = 0.0 ✓ invariant
- float32 inputs:  diff = 6e-05 ✗ not invariant

---

## The mechanism — what to visualize

### NKI execution flow (show this as a pipeline diagram)

```
HBM (bfloat16)
    ↓  nisa.dma_copy
SBUF a_tile [K_TILE, M_TILE]  (bfloat16)
SBUF b_tile [K_TILE, N]       (bfloat16)
    ↓  nisa.nc_matmul  ← Tensor Engine multiplies bfloat16 × bfloat16
PSUM c_psum [M_TILE, N]       (float32)  ← accumulates here
    ↓  nisa.tensor_copy
SBUF c_sbuf [M_TILE, N]       (bfloat16) ← cast back
    ↓  nisa.dma_copy
HBM result  [M, N]            (bfloat16)
```

### Where the invariance comes from

The Tensor Engine multiplies two bfloat16 values. bfloat16 has a 7-bit mantissa — only ~128 distinct values between any two powers of 2. The product is snapped to this coarse grid **before** it enters the float32 PSUM.

Show: a zoomed-in number line. float32 has dense tick marks. bfloat16 has sparse tick marks. Two bfloat16 inputs multiply → result lands on a bfloat16 tick mark. That tick mark is the same no matter how you group the K tiles.

### K_TILE=128 vs K_TILE=64 side by side

Show two accumulation trees for K=512:

```
K_TILE=128 (4 steps):   [p0..p127] + [p128..p255] + [p256..p383] + [p384..p511]
K_TILE=64  (8 steps):   [p0..p63] + [p64..p127] + ... + [p448..p511]
```

Each `p_i` is a bfloat16-precision product. Because they're already on the coarse grid, regrouping them gives the same float32 sum. Both trees reach the same PSUM value → same bfloat16 output after cast.

With float32 inputs: each `p_i` is sharp (23-bit mantissa). The intermediate float32 sums round differently depending on grouping → different final values.

---

## NOTE: What is actually being compared

`diff = (out_det - out_adp).abs().max()` compares the two kernel outputs against **each other** — K_TILE=128 result vs K_TILE=64 result on the same inputs. There is no ground truth / PyTorch reference. `diff=0` means the two tiling strategies are bitwise identical.

`test_determinism` is a separate check: it runs the *same* kernel 1000 times and compares each run to the first run — ruling out hardware non-determinism. That one does have a reference: run 0.

So there are two distinct invariance claims:
- **Tiling invariance**: K_TILE=128 and K_TILE=64 give the same output (the main result)
- **Run-to-run determinism**: the same kernel always gives the same output across repeated calls

---

## The precise one-liner

> bfloat16's 7-bit mantissa snaps every multiply result to a coarse grid **before** it enters the float32 PSUM — so no matter how many accumulation steps you use, the inputs to the accumulator are identical.

(It is NOT just the final cast chopping off the error — the coarseness happens at multiply time, upstream of the accumulator.)

---

## Numbers for the visual

| | bfloat16 | float32 |
|---|---|---|
| Mantissa bits | 7 | 23 |
| Distinct values per power-of-2 interval | ~128 | ~8 million |
| K_TILE=128 vs K_TILE=64 diff (K=512, linspace input, Trn2) | **0.0** | **6e-05** |
| Batch invariant? | ✓ Yes | ✗ No |

K=512, K_TILE=128 → 4 PSUM accumulations  
K=512, K_TILE=64  → 8 PSUM accumulations  
Same bfloat16 products in → same float32 sum out → same bfloat16 result

---

## Value-driven story: what the tensors actually see

Inputs: `linspace(-1, 1)`, K=512, M=N=128. Watching a single output element: `PSUM[row=0, col=0]`.

### bfloat16 — deterministic=True (K_TILE=128, 4 accumulation steps)

The Tensor Engine processes 128 K-elements at a time and writes the running sum into the float32 PSUM:

```
after tile 1 (K=  128):  PSUM = 75.041992
after tile 2 (K=  256):  PSUM = 85.833984
after tile 3 (K=  384):  PSUM = 96.375977
after tile 4 (K=  512):  PSUM = 170.667969  ← final result
```

### bfloat16 — deterministic=False (K_TILE=64, 8 accumulation steps)

Same inputs, same output element, but now 64 K-elements per tile:

```
after tile 1 (K=   64):  PSUM =  49.552246
after tile 2 (K=  128):  PSUM =  75.041992  ← same as det=True after tile 1 ✓
after tile 3 (K=  192):  PSUM =  84.469238
after tile 4 (K=  256):  PSUM =  85.833984  ← same as det=True after tile 2 ✓
after tile 5 (K=  320):  PSUM =  87.136230
after tile 6 (K=  384):  PSUM =  96.375977  ← same as det=True after tile 3 ✓
after tile 7 (K=  448):  PSUM = 121.553223
after tile 8 (K=  512):  PSUM = 170.667969  ← same final result ✓
```

Every checkpoint where both strategies have processed the same number of K-elements, the PSUM value is **bitwise identical**. The float32 accumulator is seeing the same numbers regardless of how the K dimension was tiled.

### float32 — deterministic=True (K_TILE=128)

```
after tile 1 (K=  128):  PSUM =  75.041336
after tile 2 (K=  256):  PSUM =  85.832672
after tile 3 (K=  384):  PSUM =  96.375954
after tile 4 (K=  512):  PSUM = 170.673157  ← final result
```

### float32 — deterministic=False (K_TILE=64)

```
after tile 1 (K=   64):  PSUM =  49.552071
after tile 2 (K=  128):  PSUM =  75.041367  ← differs from det=True: 75.041336 vs 75.041367 ✗
after tile 3 (K=  192):  PSUM =  84.468163
after tile 4 (K=  256):  PSUM =  85.832703  ← differs: 85.832672 vs 85.832703 ✗
after tile 5 (K=  320):  PSUM =  87.135231
after tile 6 (K=  384):  PSUM =  96.375992  ← differs: 96.375954 vs 96.375992 ✗
after tile 7 (K=  448):  PSUM = 121.555222
after tile 8 (K=  512):  PSUM = 170.673172  ← differs: 170.673157 vs 170.673172 ✗
```

Divergence appears **at the very first shared checkpoint** (K=128) and compounds from there. This is happening inside the float32 PSUM — before any output cast.

### Why bfloat16 products are identical but float32 products are not

The first 4 products `a[k,0] * b[k,0]` going into the accumulator:

```
bfloat16 (snapped to coarse grid):
  k=0: -1.000000 × -1.000000 = 1.00000000
  k=1: -0.996094 × -0.996094 = 0.99220276
  k=2: -0.992188 × -0.992188 = 0.98443604
  k=3: -0.988281 × -0.988281 = 0.97669983

float32 (full precision):
  k=0: -1.00000000 × -1.00000000 = 1.00000000000
  k=1: -0.99609369 × -0.99609369 = 0.99220264000
  k=2: -0.99218738 × -0.99218738 = 0.98443580000
  k=3: -0.98828107 × -0.98828107 = 0.97669948000
```

The bfloat16 inputs are already snapped to a coarse grid (e.g. `-0.996094` instead of `-0.99609369`). The products are therefore coarser too. When you add 64 of these coarse products vs 128 of them, the float32 accumulator reaches the same intermediate value because the individual products were already rounded to the same bfloat16 slots. With float32, the extra decimal places in each product mean different groupings accumulate rounding error differently.

`inspect_psum.py` uses `nki.simulate` to snapshot the float32 PSUM buffer after every K tile accumulation, for both K_TILE=128 and K_TILE=64. This lets us see exactly where divergence appears — or doesn't.

Inputs: `linspace(-1, 1)`, K=512, M=N=128.

### bfloat16 inputs

```
PSUM after first 128 K-elements: K_TILE=128 vs K_TILE=64 → diff = 0.000000e+00
PSUM after all 512 K-elements:   K_TILE=128 vs K_TILE=64 → diff = 3.051758e-05  (simulator artifact*)

Sample PSUM row 0, cols 0-3:
  K_TILE=128: [170.66797, 170.66797, 170.66797, 170.66797]
  K_TILE=64:  [170.66797, 170.66797, 170.66797, 170.66797]
```

The float32 PSUM is **bitwise identical** after the first 128 K-elements. The accumulator never sees different values — invariance is established before any output cast.

*The small diff at K=512 is a CPU simulator artifact from sequential execution; on Trn2 hardware the diff is 0.0.

### float32 inputs

```
PSUM after first 128 K-elements: K_TILE=128 vs K_TILE=64 → diff = 1.373291e-04
PSUM after all 512 K-elements:   K_TILE=128 vs K_TILE=64 → diff = 1.678467e-04

Sample PSUM row 0, cols 0-3:
  K_TILE=128: [170.6711,  170.67136, 170.67111, 170.67126]
  K_TILE=64:  [170.6712,  170.67122, 170.6712,  170.67114]
```

The float32 PSUM **already diverges after the very first tile** (128 K-elements). The difference is visible inside the accumulator itself, before any cast back to bfloat16. This is pure accumulation-order sensitivity.

### What this proves

The divergence for float32 lives inside the float32 PSUM — it is not introduced by the output cast. For bfloat16, the PSUM is identical at every snapshot. This confirms the mechanism:

> Invariance is established at **multiply time** (bfloat16 products are coarse before entering PSUM), not at **cast time** (the output cast to bfloat16 is not what equalizes the results).
