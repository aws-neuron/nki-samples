"""
Copyright (C) 2026, Amazon.com. All Rights Reserved

PyTorch implementation for tensor addition NKI tutorial.

"""
# NKI_EXAMPLE_29_BEGIN
import torch
import torch_neuronx
# NKI_EXAMPLE_29_END

from tensor_addition_nki_kernels import nki_tensor_add


# NKI_EXAMPLE_29_BEGIN
if __name__ == "__main__":
  a = torch.rand((256, 1024), dtype=torch.float32)
  b = torch.rand((256, 1024), dtype=torch.float32)

  trace = torch_neuronx.trace(nki_tensor_add, (a, b))
  output_nki = trace(a, b)
  print(f"output_nki={output_nki}")

  output_torch = a + b
  print(f"output_torch={output_torch}")

  allclose = torch.allclose(output_torch, output_nki.cpu(), atol=1e-4, rtol=1e-2)
  if allclose:
    print("NKI and Torch match")
  else:
    print("NKI and Torch differ")

  assert allclose
  # NKI_EXAMPLE_29_END
