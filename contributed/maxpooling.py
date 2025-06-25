import math

from neuronxcc import nki
import neuronxcc.nki.language as nl
import numpy as np
import torch
import torch.nn.functional as F


@nki.jit
def max_pooling_2d_stride_1(in_tensor: nki.tensor, pool_size: int) -> nki.tensor:
    """
    Performs 2D max pooling with stride 1 on a 2D tensor.
    
    Args:
        in_tensor: Input tensor with shape [height, width]
        pool_size: Size of the pooling window (pool_size x pool_size)
        
    Returns:
        Output tensor with shape [height-(pool_size-1), width-(pool_size-1)]
    """
    k = pool_size
    h_in, w_in = in_tensor.shape
    h_out, w_out = h_in - (k-1), w_in - (k-1)
    out_tensor = nl.ndarray((h_out, w_out), dtype=in_tensor.dtype, buffer=nl.shared_hbm)

    h_tiles_count = math.ceil(h_in / nl.tile_size.pmax)
    for h_tile_idx in nl.affine_range(h_tiles_count):
        in_tile = nl.ndarray((nl.par_dim(nl.tile_size.pmax), k, w_in), dtype=in_tensor.dtype, buffer=nl.sbuf)
        i_h, i_kh, i_w = nl.mgrid[0:nl.tile_size.pmax, 0:k, 0:w_in]
        i_h = h_tile_idx * nl.tile_size.pmax + i_h
        in_tile = nl.load(in_tensor[i_h + i_kh, i_w], mask=(i_h < (h_in - (k-1))))
        i_h, i_kh, i_w, i_kw = nl.mgrid[0:nl.tile_size.pmax, 0:k, 0:(w_in - (k-1)), 0:k]
        out_tile = nl.max(in_tile[i_h, i_kh, i_w + i_kw], axis=[1, 3], mask=(h_tile_idx * nl.tile_size.pmax + i_h < h_in))
        i_h_out, i_w_out = nl.mgrid[0:nl.tile_size.pmax, 0:(w_in - (k-1))]
        i_h_out = h_tile_idx * nl.tile_size.pmax + i_h_out
        nl.store(out_tensor[i_h_out, i_w_out], value=out_tile, mask=(i_h_out < h_out))

    return out_tensor


def check_correct():
    H, W = 448, 448
    pool_size = 3
    
    input_tensor_torch = torch.rand(H, W, dtype=torch.float32)
    input_tensor_np = input_tensor_torch.numpy()
    
    # Run max pooling on the baremetal device
    baremetal_func = nki.baremetal()(max_pooling_2d_stride_1)
    output_np = baremetal_func(input_tensor_np, pool_size)
    output_torch = torch.from_numpy(output_np)[None, None, :, :]
    
    # Compute the reference result using PyTorch's max_pool2d
    expected_output = F.max_pool2d(
        input_tensor_torch.reshape(1, 1, H, W), 
        kernel_size=pool_size,
        stride=1,
        padding=0
    )
    
    print("Is close?", torch.allclose(output_torch, expected_output, rtol=1.3e-6, atol=1e-5))
    

def benchmark_kernel():
    H, W = 448, 448
    pool_size = 3
    
    input_tensor = np.random.random_sample([H, W]).astype(np.float32)
    
    benchmark_func = nki.benchmark(warmup=5, iters=10, save_neff_name="max_pooling.neff")(max_pooling_2d_stride_1)
    benchmark_func(input_tensor, pool_size)
    

def main():
  check_correct()
  benchmark_kernel()


if __name__ == "__main__":
  main()
