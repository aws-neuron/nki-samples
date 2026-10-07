"""
Copyright (C) 2026, Amazon.com. All Rights Reserved

PyTorch implementation for tensor indirection NKI tutorial.
"""

# NKI_EXAMPLE_43_BEGIN
import torch
import torch_neuronx
# NKI_EXAMPLE_43_END

from tensor_indirection_nki_kernels import tensor_indirection_kernel

DATA_P = 128
DATA_F = 64
INDEX_F = 32


def tensor_indirection_torch(index_tensor, data_tensor):
  """Reference: out = data[:, index[0]]."""
  return data_tensor[:, index_tensor[0].to(torch.int64)]


# NKI_EXAMPLE_43_BEGIN
class TensorIndirectionModule(torch.nn.Module):
  def forward(self, index_tensor, data_tensor):
    return tensor_indirection_kernel(index_tensor, data_tensor)


if __name__ == "__main__":
  index_tensor = torch.randint(0, DATA_F, (1, INDEX_F), dtype=torch.uint16)
  data_tensor = torch.rand((DATA_P, DATA_F), dtype=torch.float32)

  model = torch_neuronx.trace(TensorIndirectionModule(),
                              (index_tensor, data_tensor))

  out_nki = model(index_tensor, data_tensor)
  out_torch = tensor_indirection_torch(index_tensor, data_tensor)

  print(out_nki, out_torch)

  allclose = torch.allclose(out_torch, out_nki)
  if allclose:
    print("NKI and PyTorch match")
  else:
    print("NKI and PyTorch differ")

  assert allclose
  # NKI_EXAMPLE_43_END