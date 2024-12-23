Tests under this folder are unit tests for the kernels in `src/nki_samples`. 

To execute the tests, we need to include `src/nki_samples` in the `PYTHONPATH`.

For example, 

PYTHONPATH=$PYTHONPATH:/home/ubuntu/nki-samples/src/ pytest test_flash_attn_fwd.py