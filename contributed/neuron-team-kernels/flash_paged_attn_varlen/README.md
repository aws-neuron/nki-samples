NKI FlashPagedAttention on Variable-length Inputs
======================================

## Kernel Usage
```python
# Unroll loop to reduce on-chip while overhead
dynamic_loop_unrolling_size = 8

# prepare plan based on sequence lengths
nki_kernel_runner = NKIFlashPagedAttentionRunner(
    query_lens,
    context_lens,
    large_q_tile_size,
    large_kv_tile_size,
    block_size,
    dynamic_loop_unrolling_size=dynamic_loop_unrolling_size,
)

# Prepare context token tile plan inputs
nki_kernel_runner.prepare_tile_plan_inputs(
    block_tables=block_table,
    max_kv_cache_size=k_cache.shape[0],
)

# pad inputs and change to layout used by kernel
num_padded_queries = nki_kernel_runner.get_num_active_tokens_after_padding()

# code to pad query dimension to `num_padded_queries`
# ...

# execute kernel
output_nki = nki_kernel_runner(
    query,
    k_cache,
    v_cache,
    k_active,
    v_active,
    use_mixed_precision,
)
```

Check out `kernel_runner.py` for example.


## Unit Test
Run full kernel test. We support two modes: torch_xla and nki baremetal.
```bash
mode=xla  # or baremetal
TEST_EXEC_MODE=$mode pytest test_kernel.py -xv
```

We also support parallel testing using multiple Neuron Cores with `pytest-xdist`.
```bash
nworker=8
TEST_EXEC_MODE=$mode pytest test_kernel.py -xv -n$nworker
```


### Branch hint
On-chip control flow benefit from branch hint. One can use env variable
`ENABLE_BRANCH_HINT=1` to enable branch hint. However, this might trigger
compiler bugs on Trn2 and kernel might hang and time out.
