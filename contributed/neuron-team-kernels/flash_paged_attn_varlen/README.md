NKI FlashAttention with BlockSparse Execution Plan
======================================

## Unit Test
Run full kernel test. We support three mode: torch_xla, nki baremetal, nki simulation.
```bash
mode=xla
#mode=baremetal
#mode=simulation
TEST_EXEC_MODE=$mode PYTHONPATH=. pytest test_kernel.py -xv
```

Note: simulation mode may fail.

We also support parallel testing using multiple Neuron Cores with `pytest-xdist`.
```bash
nworker=8
TEST_EXEC_MODE=$mode PYTHONPATH=. pytest test_kernel.py -xv -n$nworker
```