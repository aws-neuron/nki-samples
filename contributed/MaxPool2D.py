"""
MaxPool2d NKI with benchmark

Pretty similar to https://github.com/aws-neuron/nki-samples/blob/main/src/nki_samples/tutorials/average_pool2d/average_pool2d_nki_kernels.py


"""

import torch
import torch.nn.functional as F
import numpy as np
import time
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
from neuronxcc.nki.typing import tensor

@nki.jit
def maxpool_kernel(in_tensor, kernel_size, padding=0, stride=None):
    """NKI kernel to compute a 2D max-pool operation
    
    Args:
        in_tensor: Input tensor of shape (C, H, W)
        kernel_size: Size of pooling window
        padding: Integer padding size.  Defaults to zero
        stride: Stride of pooling window. If None, defaults to pool_size
        pad should be at most half of effective kernel size, 
        Only handles integer pad at this point

        Todo:  add an assert for the pad/kernel size like pytorch does
            Also, consider a (1,2) type format for padding in addition to an integer.  
            Maybe the same thing for kernel size?
            Add a batch dimension
    """
    if stride is None:
        stride = kernel_size
        
    # Get input/output dimensions
    sz_cin, sz_hin, sz_win = in_tensor.shape
    sz_hout = (sz_hin + 2*padding - kernel_size) // stride + 1
    sz_wout = (sz_win + 2*padding - kernel_size) // stride + 1
    
        
    # Create output tensor
    out_tensor = nl.ndarray((sz_cin, sz_hout, sz_wout), dtype=in_tensor.dtype,
                           buffer=nl.shared_hbm)

    # Set relevant sizes
    sz_p = sz_cin

    # Generate pool index patterns with stride
    i0 = nl.arange(sz_p)[:, None, None, None, None]  # Channel dim
    i1 = nl.arange(sz_hout)[None, :, None, None, None]  # Output height
    i2 = nl.arange(sz_wout)[None, None, :, None, None]  # Output width
    i3 = nl.arange(kernel_size)[None, None, None, :, None]  # Pool height
    i4 = nl.arange(kernel_size)[None, None, None, None, :]  # Pool width

    # Load input data
    in_tile: tensor[sz_p, sz_hin, sz_win] = nl.load(in_tensor)

    # Regular maxpool for non-padded region
    regular_out = nl.max(
        in_tile[i0, 
               stride*i1-padding + i3,  # Apply stride to height, accounting for padding
               stride*i2-padding + i4], # Apply stride to width, accounting for padding
        axis=[3, 4],  # Reduce over pool dimensions
        #account for the edges in the mask.  Since we aren't adding a column and we adjusted our indices, 
        #we just have to make sure we aren't processing the rows and columns that aren't there with the mask
        mask = (stride*i1 -padding +i3 >=0) & (stride*i2 -padding +i4 >=0) & (stride*i1-padding + i3 < sz_hin) & (stride*i2-padding + i4 < sz_win)
    )
    nl.store(out_tensor, value=regular_out)

   
    return out_tensor

def benchmark_maxpool():
    # Test parameters
    batch_size = 1
    channels = 64
    height = 224
    width = 224
    kernel_size = 4
    stride = 2
    padding = 1
    
    # Create example input
    x_np = np.random.randn(channels, height, width).astype(np.int8)
    x_torch = torch.from_numpy(x_np).unsqueeze(0)  # For reference comparison
    
    print("Running inference...")
    
    benchmark_func = nki.benchmark(
        maxpool_kernel,
        warmup=10,
        iters=100,
        save_neff_name='file.neff',
        save_trace_name='maxpool.ntff'
    )
    output_nki = benchmark_func(x_np, kernel_size, stride, padding)
    #reassigning because the benchmark output doesn't match the kernel output.  Leaving it in here as an illustration.
    output_nki = maxpool_kernel(x_np, kernel_size=kernel_size, stride=stride, padding=padding)


    # Print benchmark results
    # Thank you Amazon Q!
    metrics = benchmark_func.benchmark_result.nc_latency
    print("\nBenchmark Results:")
    print(f"P50 Latency: {metrics.get_latency_percentile(50):>8.2f} us")
    print(f"P90 Latency: {metrics.get_latency_percentile(90):>8.2f} us")
    print(f"P99 Latency: {metrics.get_latency_percentile(99):>8.2f} us")

    # Verify shapes
    print(f"\nShape verification:")
    print(f"Input shape: {x_np.shape}")
    print(f"Output shape: {output_nki.shape}")
    
    # Calculate expected output shape
    expected_h = ((height + 2 * padding - kernel_size) // stride + 1)
    expected_w = ((width + 2 * padding - kernel_size) // stride + 1)
    print(f"Expected output shape: ({channels}, {expected_h}, {expected_w})")
    
    # Verify against CPU reference
    print("\nVerifying against CPU reference...")
    with torch.no_grad():
        ref_output = F.max_pool2d(x_torch, 
                                 kernel_size=kernel_size,
                                 stride=stride,
                                 padding=padding)
    
    ref_output = ref_output.squeeze(0).numpy()
    max_diff = np.max(np.abs(output_nki - ref_output))
    print(f"Maximum difference from reference: {max_diff}")

if __name__ == "__main__":
    benchmark_maxpool()
