"""
Copyright (C) 2026, Amazon.com. All Rights Reserved

PyTorch implementation for transpose2d NKI tutorial.
"""

# NKI_EXAMPLE_34_BEGIN
import torch
import torch_xla
# NKI_EXAMPLE_34_END

from transpose2d_nki_kernels import tensor_transpose2D_kernel_


# NKI_EXAMPLE_34_BEGIN
if __name__ == "__main__":
  device = torch_xla.device()

  P, X, Y = 5, 3, 4
  a = torch.arange(P*X*Y, dtype=torch.float32).reshape((P, X*Y)).to(device=device)
  a_t_nki = tensor_transpose2D_kernel_(a, (X, Y))

  a_t_torch = torch.transpose(a.reshape(P, X, Y), 1, 2).reshape(P, X * Y)

  print(a, a_t_nki, a_t_torch)

  allclose = torch.allclose(a_t_torch, a_t_nki)
  if allclose:
    print("NKI and PyTorch match")
  else:
    print("NKI and PyTorch differ")

  assert allclose
  # NKI_EXAMPLE_34_END
