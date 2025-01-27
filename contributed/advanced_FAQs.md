# Frequently Asked Questions: Advanced NKI Troubleshooting Guide

## Table of Contents
1. [Basic Syntax Questions](#basic-syntax)
2. [Memory and Variable Management](#memory-and-variables) 
3. [Performance Optimization](#performance)
4. [Debugging and Troubleshooting](#debugging)
5. [Common Compiler Errors](#compiler-errors)

## Basic Syntax

### What's the difference between `var[...] = ` and `var = `?

When you want to create a completely new variable, such as loading data directly using `nl.load()` without a predefined buffer, or computing some mathematical function as the result of multiple tiles to create a new tile, then you can use `foo = nl.load()` or `bar = tile_a + tile_b`.

However, sometimes you simply want to work with the data or physical address stored by reference through that variable. This is the case at least twice in examples you've seen.
1.  When you create a buffer on chip, such as with `a_tile = nl.ndarray( etc, buffer = nl.sbuf)`. You can then use `a_tile[...] = nl.load()` to load directly onto that buffer. 
2. When you are performing computation on tiles through indexing, and you want that same indexing logic to carry into the position where your resultant data is stored. This is the case for `a_tile[...] = nl.load(a_vec[m * ROW_CHUNK : (m + 1) * ROW_CHUNK])`.
    
Generally we use the `var[...]` syntax to let you alter the data represented by your variable without you needing to define a new one.

When you have a variable declared with `var = nl.ndarray(etc)`, you should use:
- `var[...] = <nki api>` to assign values to the existing variable
- `var = <nki api>` creates a new variable that replaces the reference to the previous variable of that name

**Performance Note:** Using `var[...]` inside loops can create unnecessary dependencies. If each iteration produces a different `var`, prefer:
```python
for i in range(N):
    var = <nki api>
```
instead of pre-declaring and using `var[...]`.

### When should I use `+=` operator?
The `+=` operator is currently only guaranteed to work reliably in this specific operation - matrix multiplication.

```python
# 1. Initialize a psum buffer with zeros
psum_buf = nl.zeros(..., buffer=nl.psum)

# 2. Use within an affine range loop
for i in nl.affine_range(N):
    # 3. Add matmul results
    psum_buf[...] += nl.matmul(...)
```

Any other use of `+=` may trigger compiler bugs. Please do not use `+=` outside of matrix multiplication.

### What's the difference between `affine_range` and `sequential_range`?
- [`nl.affine_range`](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/api/generated/nki.language.affine_range.html#nki.language.affine_range) creates a sequence of numbers for use as parallel loop iterators in NKI. What this means is that the compiler will unroll your loop, identify the parts that can be parallelized, and then run them in parallel. For this reason, `affine_range` should be the default loop iterator choice when there is no loop-carried dependency. What that means is that you can should only use `nl.affine_range` when there is no depdency between steps in your loops, ie when the steps can be executed at any point in time. Please note that associative reductions are not considered loop carried dependencies in this context. A concrete example of associative reduction is multiple `nl.matmul` or `nisa.nc_matmul` calls accumulating into the same output buffer defined outside of this loop level. `nl.affine_range` allows parallel execution of loop iterations when there are no dependencies
- [`nl.sequential_range`](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/api/generated/nki.language.sequential_range.html) creates a sequence of numbers for use as sequential loop iterators in NKI. sequential_range should be used when there ***is*** a loop carried dependency. `nl.sequential_range` enforces sequential execution, meaning that you want your loop executed exactly in the sequence you identified.
- For initial development, it's helpful to start with `sequential_range` to ensure functionality and accuracy.
- Once your kernel is working, you can switch to `affine_range` for performance optimization

## Memory and Variables

### How do tensor allocations work?
[XLA tensors](https://pytorch.org/xla/master/learn/pytorch-on-xla-devices.html#id3) follow a lazy execution model that differs fundamentally from PyTorch's eager execution. When a tensor is declared, it doesn't immediately allocate memory - instead, it creates a descriptor and records operations in a computation graph. This deferred execution strategy allows XLA to optimize operations, potentially fusing multiple separate operations into a single optimized operation. The actual memory allocation only occurs at execution time when the results are needed, which can lead to more efficient memory usage and better performance through operation optimization.

Memory management and allocation patterns significantly impact performance and resource utilization. When working with tensors in loops, resuing tensors in the loop may create unnecessary dependencies, while declaring them inside the loop may improve memory reuse.  


### What are the constraints on reduction operations?
For operations like `nl.max`:
- The axis must be free dimensions (not partition dimension 0)
- The compiler can automatically reorder free dimensions to meet hardware constraints
- Examples of valid axes: [1], [1,2], [1,2,3], [1,2,3,4]

## Performance Optimization

### How can I optimize memory usage?
1. Avoid unnecessary tensor declarations outside loops, if the tensor is not used outside of the loop. While reusing tensors by declaring them outside loops might seem memory-efficient, it can create complex dependency chains in the computation graph that increase management overhead and limit optimization opportunities.

Compare the two approaches in below:
```python
# Option 1
weights_mat_mul_2 = nl.ndarray((foo, bar, n_tiles_a_o, nl.par_dim(i_pmax), n_tiles_a_i, i_pmax), dtype=W.dtype, buffer=nl.sbuf)

w_temp = nl.ndarray((nl.par_dim(i_pmax), n_tiles_a_i, i_pmax, foo, bar), dtype=W.dtype, buffer=nl.sbuf)

for out_ in nl.sequential_range(n_tiles_a_o):
    w_temp[...] = nl.load(W_reshaped[out_,:,:,:,:,:])
    for fH in nl.affine_range(foo):
        for fW in nl.affine_range(bar):
            for in_ in nl.affine_range(n_tiles_a_i):
                weights_mat_mul_2[fH,fW,out_,:,in_,:] = w_temp[:,in_,:,fH,fW]
```
```python
# Option 2
weights_mat_mul_2 = nl.ndarray((foo, bar, n_tiles_a_o, nl.par_dim(i_pmax), n_tiles_a_i, i_pmax), dtype=W.dtype, buffer=nl.sbuf)

for out_ in nl.sequential_range(n_tiles_a_o):
    w_temp = nl.ndarray((nl.par_dim(i_pmax), n_tiles_a_i, i_pmax, foo, bar), dtype=W.dtype, buffer=nl.sbuf)
    w_temp[...] = nl.load(W_reshaped[out_,:,:,:,:,:])
    for fH in nl.affine_range(foo):
        for fW in nl.affine_range(bar):
            for in_ in nl.affine_range(n_tiles_a_i):
                weights_mat_mul_2[fH,fW,out_,:,in_,:] = w_temp[:,in_,:,fH,fW]

```

Both options can work. But option 2 will be more efficient, as it will turn the nl.sequential_range() back to nl.affine_range() and get better throughput since the loop iterations don't need to wait for a shared chunk of memory to be updated. Tensor `w_temp` is not used outside of the loop so allocating it outside of the loop just adds an unnecessary loop-carried dependency to figure out. 

2. Use appropriate buffer types (SBUF/PSUM). SBUF is on-chip storage on the NeuronCore. In comparison, SBUF is significantly smaller than HBM (24 MiB) but offers much higher bandwidth (~20x than HBM). PSUM is a small, specialized memory (2 MiB) dedicated to holding matrix multiplication results produced by the tensor engine.

3. Consider tile sizes carefully to prevent "too many instructions" errors. The Tensor Engine imposes special layout constraints on the input tiles. First, the partition axis sizes of the `stationary` and `moving` tiles must be both identical and `<=128`, which corresponds to the contraction dimension of the matrix multiplication. Second, the free axis sizes of `stationary` and `moving` tiles must be `<= 128` and `<=512`, respectively. For example, `stationary.shape = (128, 126)`; `moving.shape = (128, 512)` and `nc_matmul(stationary,moving)` returns a tile of `shape = (126, 512)`. For more information about the matmul layout, see [Tensor Engine](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/arch/trainium_inferentia2_arch.html#arch-guide-tensor-engine).

### What should I know about direct allocation?
- Use `@nki.jit(mode='baremetal', disable_allocation_opt=True)` for direct control
- Ensure allocated tensors don't overlap if they need to be alive simultaneously
- Be aware that compiler optimizations may override allocation decisions

## Debugging

### How can I debug intermediate values?
Current options:
1. Use `nl.device_print` in simulation mode. This is available on CPU only, and with bare metal input arrays. See the docs on this topic [here](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/api/generated/nki.simulate_kernel.html).
2. Return intermediate tensors as kernel outputs for host inspection. This means that, just like with normal Python code, you can end your function early to return a value, then print it out and see what's happening with it.
3. For hardware runs, you'll need to modify the kernel to expose intermediate values. It means you can return intermediate values when you are testing on the hardware, just like with CPU. Simply return the intermediate value of interest as the end of your debug kernel.

### What should I do if my kernel works in simulation but fails on hardware?
1. Check for memory allocation conflicts
2. Verify tensor shapes and access patterns
3. Consider using sequential_range initially
4. Open a support ticket if the issue persists

## Compiler Errors

### Common Error Messages and Solutions

#### "Allocated memory out of bound"
```
[NLA001] Unhandled exception: Allocated memory out of bound...
```
**Solution:**
- Check tensor dimensions against hardware limits
- Verify tile sizes are within SBUF partition size
- Consider breaking large tensors into smaller tiles

#### "Unexpected output dependencies"
```
SyntaxError: Unexpected output dependencies, missing indices...
```
**Solution:**
- Verify all loop indices are properly used in tensor accesses
- Check for missing dimension specifications
- Ensure loop ranges match tensor dimensions

#### "Too many instructions" Error
**Common causes:**
- Using very small tiles for DMA operations
- Complex nested loops with small tile sizes
- Excessive unrolling

**Solutions:**
- Increase tile sizes where possible
- Consolidate operations
- Review loop structure for optimization opportunities

### Best Practices for Avoiding Compiler Errors

1. **Code Organization**
   - Keep tile sizes reasonable, not too big and not too small.
   - Use consistent variable naming
   - Document tensor shapes and memory requirements

2. **Memory Management**
   - Explicitly declare buffer types (SBUF/PSUM)
   - Be careful with tensor reuse, try to only declare a tensor inside of a loop where possible.
   - Verify memory alignment requirements

3. **Performance Considerations**
   - Start with working code using `sequential_range`
   - Profile before optimizing
   - Test with various input sizes

For more best practices and recommendations, please view [NKI Performance Guide](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/nki_perf_guide.html).

## Resource Links

### Documentation
- [NKI Programming Guide](link)
- [API Reference](link)
- [Known Limitations](link)
- [Troubleshooting Guide](link)

### Getting Help
1. Check the error code reference: https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/api/nki.errors.html
2. Review the [Known Issues](link) page
3. Submit detailed bug reports including:
   - Full error message
   - Minimal code example
   - Input shapes and types
   - Expected vs actual behavior

### Common Tools
- [Performance Profiler](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/neuron_profile_for_nki.html)
- [Simulation Environment](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/api/generated/nki.simulate_kernel.html)

---
**Note:** This FAQ is maintained based on user feedback and common issues. If you encounter problems not covered here, please report them through the appropriate channels for inclusion in future updates.
