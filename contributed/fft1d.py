"""
1D Fast Fourier Transform (FFT) - NKI Implementation

This module provides a hardware-accelerated 1D FFT implementation using the
AWS Neuron Kernel Interface (NKI), optimized for AWS Trainium and Inferentia processors.

Author:  Jim Burtoft

Algorithm:
    Radix-2 Cooley-Tukey FFT with 128-point DFT base case
    - Base Case (width=128): Direct DFT matrix multiplication using Tensor Engine
    - Recursive Case (width>128): Radix-2 decomposition with twiddle factors
    - Height Handling: Automatic tiling with hardware-accelerated masking

Hardware Optimization:
    - Tensor Engine: Used for 128x128 DFT matrix multiplication (~90% of compute)
    - SBUF Management: Efficient on-chip memory usage with minimal HBM traffic
    - Computational Masking: Hardware-accelerated masking for arbitrary heights

Supported Input Sizes:
    - Height: 1 to 128 (any value in this range)
    - Width: Powers of 2 from 128 to 4096 (128, 256, 512, 1024, 2048, 4096)

Accuracy:
    - Typical relative error: < 0.003% compared to NumPy reference
    - Validated across 50+ size combinations on Trainium hardware

Limitations:
    - Height must be <= 128 (hardware constraint from tile size)
    - Width must be a power of 2 and >= 128
    - Only forward FFT (no inverse FFT)
    - No normalization options
    - Requires AWS Neuron device (Trainium/Inferentia)

Usage Example:
    import torch
    from fft1d import fft1d
    
    # Create input tensor
    x = torch.randn(64, 512, dtype=torch.float32)
    
    # Compute 1D FFT along rows (last dimension)
    y = fft1d(x)
    
    print(f"Input shape: {x.shape}")   # torch.Size([64, 512])
    print(f"Output shape: {y.shape}")  # torch.Size([64, 512])
    print(f"Output dtype: {y.dtype}")  # torch.complex64
"""

import torch
import numpy as np
import torch_xla.core.xla_model as xm
import os

from neuronxcc import nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa

# Disable auto-casting for precise control over data types
os.environ['NEURON_CC_FLAGS'] = '--auto-cast=none --enable-mixed-precision-accumulation'


def _compute_dft_matrix(N: int):
    """
    Compute the DFT (Discrete Fourier Transform) matrix for N-point FFT.
    
    The DFT matrix W is defined as:
        W[k,n] = exp(-2πi * k * n / N)
    
    We separate this into real and imaginary components:
        W_real[k,n] = cos(-2π * k * n / N)
        W_imag[k,n] = sin(-2π * k * n / N)
    
    Args:
        N: FFT size (must be 128 for this implementation)
    
    Returns:
        Tuple of (W_real, W_imag) as numpy arrays of shape [N, N]
    """
    k = np.arange(N, dtype=np.float32).reshape(N, 1)
    n = np.arange(N, dtype=np.float32).reshape(1, N)
    angles = -2.0 * np.pi * k * n / N
    W_real = np.cos(angles).astype(np.float32)
    W_imag = np.sin(angles).astype(np.float32)
    return W_real, W_imag


def _compute_twiddle_factors(N: int, H: int):
    """
    Compute twiddle factors for radix-2 FFT butterfly operations.
    
    Twiddle factors are complex exponentials used in the Cooley-Tukey algorithm:
        W_N^k = exp(-2πi * k / N)
    
    For radix-2, we need N/2 twiddle factors.
    
    Args:
        N: FFT size
        H: Height (batch size) - twiddle factors are broadcast across height
    
    Returns:
        Tuple of (twiddle_real, twiddle_imag) as numpy arrays of shape [H, N/2]
    """
    k = np.arange(N // 2, dtype=np.float32)
    angles = -2.0 * np.pi * k / N
    twiddle_real_1d = np.cos(angles).astype(np.float32)
    twiddle_imag_1d = np.sin(angles).astype(np.float32)
    
    # Broadcast to [H, N/2] for efficient batch processing
    twiddle_real = np.tile(twiddle_real_1d, (H, 1))
    twiddle_imag = np.tile(twiddle_imag_1d, (H, 1))
    
    return twiddle_real, twiddle_imag


def _fft1d_matmul_isa(X_real, X_imag, Y_real, Y_imag, W_real, W_imag, axis: int):
    """
    Perform 1D FFT using matrix multiplication via the Tensor Engine.
    
    This function implements the DFT as a matrix-vector product:
        Y = W @ X
    
    For complex numbers, this expands to:
        Y_real = W_real @ X_real - W_imag @ X_imag
        Y_imag = W_real @ X_imag + W_imag @ X_real
    
    The Tensor Engine requires the stationary matrix to be transposed.
    
    Args:
        X_real, X_imag: Input tiles (real and imaginary parts)
        Y_real, Y_imag: Output tiles (real and imaginary parts)
        W_real, W_imag: DFT matrix (real and imaginary parts)
        axis: Axis along which to perform FFT (1 for row FFT)
    """
    if axis == 1:  # Row FFT
        # Transpose DFT matrix for Tensor Engine (stationary matrix requirement)
        W_real_T = nisa.nc_transpose(W_real)
        W_imag_T = nisa.nc_transpose(W_imag)
        
        # Transpose input for proper matrix multiplication orientation
        X_real_T = nisa.nc_transpose(X_real)
        X_imag_T = nisa.nc_transpose(X_imag)
        
        H, W_size = X_real.shape
        
        # Allocate PSUM buffers for Tensor Engine output
        term1_psum = nl.ndarray((H, W_size), dtype=nl.float32, buffer=nl.psum)
        term2_psum = nl.ndarray((H, W_size), dtype=nl.float32, buffer=nl.psum)
        term3_psum = nl.ndarray((H, W_size), dtype=nl.float32, buffer=nl.psum)
        term4_psum = nl.ndarray((H, W_size), dtype=nl.float32, buffer=nl.psum)
        
        # Perform matrix multiplications using Tensor Engine
        # These operations leverage the 128x128 systolic array for high throughput
        term1_psum[...] = nisa.nc_matmul(X_real_T, W_real_T)  # Real * Real
        term2_psum[...] = nisa.nc_matmul(X_imag_T, W_imag_T)  # Imag * Imag
        term3_psum[...] = nisa.nc_matmul(X_real_T, W_imag_T)  # Real * Imag
        term4_psum[...] = nisa.nc_matmul(X_imag_T, W_real_T)  # Imag * Real
        
        # Copy from PSUM to SBUF for further operations
        term1 = nl.ndarray((H, W_size), dtype=nl.float32, buffer=nl.sbuf)
        term2 = nl.ndarray((H, W_size), dtype=nl.float32, buffer=nl.sbuf)
        term3 = nl.ndarray((H, W_size), dtype=nl.float32, buffer=nl.sbuf)
        term4 = nl.ndarray((H, W_size), dtype=nl.float32, buffer=nl.sbuf)
        
        term1[...] = nl.copy(term1_psum[...])
        term2[...] = nl.copy(term2_psum[...])
        term3[...] = nl.copy(term3_psum[...])
        term4[...] = nl.copy(term4_psum[...])
        
        # Combine terms according to complex multiplication rules
        Y_real[...] = term1 - term2  # Real part
        Y_imag[...] = term3 + term4  # Imaginary part


@nki.jit
def _fft1d_dft_128_masked(X_real_hbm, X_imag_hbm,
                          W_128_real_hbm, W_128_imag_hbm,
                          actual_height):
    """
    Base case: 128-point FFT using direct DFT matrix multiplication with height masking.
    
    This kernel handles the base case of the recursive FFT algorithm. It performs
    a direct DFT using matrix multiplication, which is efficient for the 128-point
    size that matches the Tensor Engine's 128x128 systolic array.
    
    Computational masking is used to handle arbitrary heights efficiently:
    - Input is padded to 128 rows if needed
    - Mask ensures only valid rows are processed
    - Hardware-accelerated masking has minimal overhead
    
    Args:
        X_real_hbm, X_imag_hbm: Input tensors in HBM [128, 128] (padded)
        W_128_real_hbm, W_128_imag_hbm: DFT matrix in HBM [128, 128]
        actual_height: Actual number of valid rows (may be < 128)
    
    Returns:
        Tuple of (Y_real_hbm, Y_imag_hbm): Output tensors in HBM [128, 128]
    """
    TILE_H = 128
    N = 128
    
    # Allocate output in HBM
    Y_real_hbm = nl.ndarray((TILE_H, N), dtype=nl.float32, buffer=nl.shared_hbm)
    Y_imag_hbm = nl.ndarray((TILE_H, N), dtype=nl.float32, buffer=nl.shared_hbm)
    
    # Create computational mask using index grids
    # This mask controls which elements are processed by the hardware
    i_h, i_w = nl.mgrid[0:TILE_H, 0:N]
    mask = (i_h < actual_height)
    
    # Load input from HBM to SBUF with masking
    # Masked elements are not loaded, saving memory bandwidth
    X_real = nl.ndarray((TILE_H, N), dtype=nl.float32, buffer=nl.sbuf)
    X_imag = nl.ndarray((TILE_H, N), dtype=nl.float32, buffer=nl.sbuf)
    X_real[...] = nl.load(X_real_hbm[i_h, i_w], mask=mask)
    X_imag[...] = nl.load(X_imag_hbm[i_h, i_w], mask=mask)
    
    # Load DFT matrix (no masking needed - always full size)
    W_real = nl.ndarray((N, N), dtype=nl.float32, buffer=nl.sbuf)
    W_imag = nl.ndarray((N, N), dtype=nl.float32, buffer=nl.sbuf)
    W_real[...] = nl.load(W_128_real_hbm)
    W_imag[...] = nl.load(W_128_imag_hbm)
    
    # Allocate output tiles in SBUF
    Y_real = nl.ndarray((TILE_H, N), dtype=nl.float32, buffer=nl.sbuf)
    Y_imag = nl.ndarray((TILE_H, N), dtype=nl.float32, buffer=nl.sbuf)
    
    # Perform DFT using Tensor Engine
    _fft1d_matmul_isa(X_real, X_imag, Y_real, Y_imag,
                      W_real, W_imag, axis=1)
    
    # Store result back to HBM with masking
    nl.store(Y_real_hbm[i_h, i_w], value=Y_real, mask=mask)
    nl.store(Y_imag_hbm[i_h, i_w], value=Y_imag, mask=mask)
    
    return Y_real_hbm, Y_imag_hbm


@nki.jit
def _fft1d_radix2_256_masked(X_real_hbm, X_imag_hbm,
                              W_128_real_hbm, W_128_imag_hbm,
                              twiddle_real_hbm, twiddle_imag_hbm,
                              actual_height):
    """
    256-point FFT using radix-2 Cooley-Tukey algorithm with height masking.
    
    This kernel implements one level of radix-2 decomposition:
    1. Split input into even and odd indexed elements
    2. Compute 128-point FFT on each half
    3. Apply twiddle factors to odd half
    4. Combine results using butterfly operations
    
    The radix-2 algorithm reduces a 256-point FFT to two 128-point FFTs,
    which can be efficiently computed using the base case kernel.
    
    Args:
        X_real_hbm, X_imag_hbm: Input tensors in HBM [128, 256] (padded height)
        W_128_real_hbm, W_128_imag_hbm: 128-point DFT matrix in HBM [128, 128]
        twiddle_real_hbm, twiddle_imag_hbm: Twiddle factors in HBM [128, 128]
        actual_height: Actual number of valid rows (may be < 128)
    
    Returns:
        Tuple of (Y_real_hbm, Y_imag_hbm): Output tensors in HBM [128, 256]
    """
    TILE_H = 128
    N = 256
    N_half = 128
    
    # Allocate output in HBM
    Y_real_hbm = nl.ndarray((TILE_H, N), dtype=nl.float32, buffer=nl.shared_hbm)
    Y_imag_hbm = nl.ndarray((TILE_H, N), dtype=nl.float32, buffer=nl.shared_hbm)
    
    # Create computational mask
    i_h, i_w = nl.mgrid[0:TILE_H, 0:N]
    mask = (i_h < actual_height)
    
    # Load input from HBM to SBUF
    X_real = nl.ndarray((TILE_H, N), dtype=nl.float32, buffer=nl.sbuf)
    X_imag = nl.ndarray((TILE_H, N), dtype=nl.float32, buffer=nl.sbuf)
    X_real[...] = nl.load(X_real_hbm[i_h, i_w], mask=mask)
    X_imag[...] = nl.load(X_imag_hbm[i_h, i_w], mask=mask)
    
    # Load DFT matrix and twiddle factors
    W_128_real = nl.ndarray((N_half, N_half), dtype=nl.float32, buffer=nl.sbuf)
    W_128_imag = nl.ndarray((N_half, N_half), dtype=nl.float32, buffer=nl.sbuf)
    W_128_real[...] = nl.load(W_128_real_hbm)
    W_128_imag[...] = nl.load(W_128_imag_hbm)
    
    twiddle_real = nl.ndarray((TILE_H, N_half), dtype=nl.float32, buffer=nl.sbuf)
    twiddle_imag = nl.ndarray((TILE_H, N_half), dtype=nl.float32, buffer=nl.sbuf)
    twiddle_real[...] = nl.load(twiddle_real_hbm)
    twiddle_imag[...] = nl.load(twiddle_imag_hbm)
    
    # Split input into even and odd indexed elements
    # Even: X[0], X[2], X[4], ...
    # Odd:  X[1], X[3], X[5], ...
    X_even_real = nl.ndarray((TILE_H, N_half), dtype=nl.float32, buffer=nl.sbuf)
    X_even_imag = nl.ndarray((TILE_H, N_half), dtype=nl.float32, buffer=nl.sbuf)
    X_odd_real = nl.ndarray((TILE_H, N_half), dtype=nl.float32, buffer=nl.sbuf)
    X_odd_imag = nl.ndarray((TILE_H, N_half), dtype=nl.float32, buffer=nl.sbuf)
    
    X_even_real[...] = X_real[:, 0::2]
    X_even_imag[...] = X_imag[:, 0::2]
    X_odd_real[...] = X_real[:, 1::2]
    X_odd_imag[...] = X_imag[:, 1::2]
    
    # Compute 128-point FFT on even and odd halves
    X_even_fft_real = nl.ndarray((TILE_H, N_half), dtype=nl.float32, buffer=nl.sbuf)
    X_even_fft_imag = nl.ndarray((TILE_H, N_half), dtype=nl.float32, buffer=nl.sbuf)
    X_odd_fft_real = nl.ndarray((TILE_H, N_half), dtype=nl.float32, buffer=nl.sbuf)
    X_odd_fft_imag = nl.ndarray((TILE_H, N_half), dtype=nl.float32, buffer=nl.sbuf)
    
    _fft1d_matmul_isa(X_even_real, X_even_imag, X_even_fft_real, X_even_fft_imag,
                      W_128_real, W_128_imag, axis=1)
    _fft1d_matmul_isa(X_odd_real, X_odd_imag, X_odd_fft_real, X_odd_fft_imag,
                      W_128_real, W_128_imag, axis=1)
    
    # Apply twiddle factors to odd half
    # Twiddle multiplication: (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
    X_odd_tw_real = nl.ndarray((TILE_H, N_half), dtype=nl.float32, buffer=nl.sbuf)
    X_odd_tw_imag = nl.ndarray((TILE_H, N_half), dtype=nl.float32, buffer=nl.sbuf)
    
    X_odd_tw_real[...] = X_odd_fft_real * twiddle_real - X_odd_fft_imag * twiddle_imag
    X_odd_tw_imag[...] = X_odd_fft_real * twiddle_imag + X_odd_fft_imag * twiddle_real
    
    # Combine using butterfly operations
    # First half:  Y[k] = Even[k] + Twiddle[k] * Odd[k]
    # Second half: Y[k + N/2] = Even[k] - Twiddle[k] * Odd[k]
    Y_combined_real = nl.ndarray((TILE_H, N), dtype=nl.float32, buffer=nl.sbuf)
    Y_combined_imag = nl.ndarray((TILE_H, N), dtype=nl.float32, buffer=nl.sbuf)
    
    Y_combined_real[:, 0:N_half] = X_even_fft_real + X_odd_tw_real
    Y_combined_imag[:, 0:N_half] = X_even_fft_imag + X_odd_tw_imag
    Y_combined_real[:, N_half:N] = X_even_fft_real - X_odd_tw_real
    Y_combined_imag[:, N_half:N] = X_even_fft_imag - X_odd_tw_imag
    
    # Store result back to HBM with masking
    nl.store(Y_real_hbm[i_h, i_w], value=Y_combined_real, mask=mask)
    nl.store(Y_imag_hbm[i_h, i_w], value=Y_combined_imag, mask=mask)
    
    return Y_real_hbm, Y_imag_hbm


def fft1d(X: torch.Tensor, dtype=torch.float32) -> torch.Tensor:
    """
    Compute 1D FFT along the last dimension (rows) using NKI hardware acceleration.
    
    This function provides a high-level interface to the NKI FFT implementation.
    It handles input validation, padding, device management, and recursive FFT
    computation automatically.
    
    Algorithm Flow:
        1. Validate input dimensions and data type
        2. Separate complex input into real/imaginary components
        3. Pad height to 128 if needed (for masking)
        4. Transfer data to Neuron device
        5. Recursively compute FFT using radix-2 decomposition
        6. Extract valid results and combine into complex tensor
        7. Return result on CPU
    
    Args:
        X: Input tensor of shape [H, W]
           - H: 1 to 128 (height/batch size)
           - W: Power of 2, >= 128 (128, 256, 512, 1024, 2048, 4096)
        dtype: Data type for computation (default: torch.float32)
    
    Returns:
        Complex tensor of shape [H, W] containing the FFT result
        Output dtype is torch.complex64 (float32 real + float32 imag)
    
    Raises:
        AssertionError: If width is not a power of 2 >= 128
    
    Example:
        >>> import torch
        >>> from fft1d import fft1d
        >>> 
        >>> # Real input
        >>> x = torch.randn(64, 512)
        >>> y = fft1d(x)
        >>> print(y.shape)  # torch.Size([64, 512])
        >>> print(y.dtype)  # torch.complex64
        >>> 
        >>> # Complex input
        >>> x_complex = torch.randn(64, 512, dtype=torch.complex64)
        >>> y = fft1d(x_complex)
    """
    H, W = X.shape
    
    # Validate width is power of 2 and >= 128
    assert W >= 128 and (W & (W - 1)) == 0, \
        f"Width must be power of 2 >= 128, got {W}"
    
    # Convert to real/imaginary components
    if torch.is_complex(X):
        x_real = torch.real(X).contiguous()
        x_imag = torch.imag(X).contiguous()
    else:
        x_real = X.contiguous()
        x_imag = torch.zeros_like(X)
    
    x_real = x_real.to(dtype)
    x_imag = x_imag.to(dtype)
    
    # Transfer to Neuron device
    device = xm.xla_device()
    x_real = x_real.to(device)
    x_imag = x_imag.to(device)
    
    # Prepare 128-point DFT matrix (base case)
    W_128_real_np, W_128_imag_np = _compute_dft_matrix(128)
    W_128_real = torch.from_numpy(W_128_real_np).to(device)
    W_128_imag = torch.from_numpy(W_128_imag_np).to(device)
    
    # Pad height to 128 if needed (for masking support)
    # Note: Heights > 128 are not currently supported by the kernels
    if H > 128:
        raise ValueError(f"Height {H} exceeds maximum supported height of 128")
    
    H_padded = 128
    if H < 128:
        x_real_padded = torch.zeros(128, W, dtype=dtype, device=device)
        x_imag_padded = torch.zeros(128, W, dtype=dtype, device=device)
        x_real_padded[:H, :] = x_real
        x_imag_padded[:H, :] = x_imag
        x_real = x_real_padded
        x_imag = x_imag_padded
    
    # Recursively compute FFT
    y_real, y_imag = _fft_recursive(
        x_real, x_imag, W, H, W_128_real, W_128_imag, device
    )
    
    # Extract valid rows (remove padding)
    y_real = y_real[:H, :]
    y_imag = y_imag[:H, :]
    
    # Transfer back to CPU and combine into complex tensor
    y_real_cpu = y_real.cpu()
    y_imag_cpu = y_imag.cpu()
    y = torch.complex(y_real_cpu, y_imag_cpu)
    
    return y


def _fft_recursive(x_real, x_imag, W, H, W_128_real, W_128_imag, device):
    """
    Recursively apply radix-2 FFT decomposition with height masking.
    
    This function implements the recursive structure of the Cooley-Tukey algorithm:
    - Base case (W=128): Use direct DFT matrix multiplication
    - Single-level (W=256): Use optimized single-level radix-2 kernel
    - Recursive case (W>256): Split, recurse, and combine
    
    The recursion tree has depth log2(W/128), with each level performing
    O(W) operations, giving overall O(W log W) complexity.
    
    Args:
        x_real, x_imag: Input tensors on device [H_padded, W]
        W: Current FFT width
        H: Actual height (for masking)
        W_128_real, W_128_imag: 128-point DFT matrix on device
        device: Neuron device
    
    Returns:
        Tuple of (y_real, y_imag): Output tensors on device
    """
    if W == 128:
        # Base case: Direct 128-point DFT with masking
        return _fft1d_dft_128_masked(
            x_real, x_imag,
            W_128_real, W_128_imag,
            H
        )
    elif W == 256:
        # Optimized single-level radix-2 with masking
        twiddle_real_np, twiddle_imag_np = _compute_twiddle_factors(W, 128)
        twiddle_real = torch.from_numpy(twiddle_real_np).to(device)
        twiddle_imag = torch.from_numpy(twiddle_imag_np).to(device)
        
        return _fft1d_radix2_256_masked(
            x_real, x_imag,
            W_128_real, W_128_imag,
            twiddle_real, twiddle_imag,
            H
        )
    else:
        # Recursive case: Split, recurse, combine
        W_half = W // 2
        
        # Split into even and odd indexed elements
        x_even_real = x_real[:, 0::2]
        x_even_imag = x_imag[:, 0::2]
        x_odd_real = x_real[:, 1::2]
        x_odd_imag = x_imag[:, 1::2]
        
        # Recursively compute FFT on each half
        y_even_real, y_even_imag = _fft_recursive(
            x_even_real, x_even_imag, W_half, H, W_128_real, W_128_imag, device
        )
        y_odd_real, y_odd_imag = _fft_recursive(
            x_odd_real, x_odd_imag, W_half, H, W_128_real, W_128_imag, device
        )
        
        # Compute twiddle factors for this level
        actual_H = y_even_real.shape[0]
        twiddle_real_np, twiddle_imag_np = _compute_twiddle_factors(W, actual_H)
        twiddle_real = torch.from_numpy(twiddle_real_np).to(device)
        twiddle_imag = torch.from_numpy(twiddle_imag_np).to(device)
        
        # Allocate output
        y_real = torch.zeros(actual_H, W, dtype=torch.float32, device=device)
        y_imag = torch.zeros(actual_H, W, dtype=torch.float32, device=device)
        
        # Apply twiddle factors to odd half
        tw_odd_real = twiddle_real * y_odd_real - twiddle_imag * y_odd_imag
        tw_odd_imag = twiddle_real * y_odd_imag + twiddle_imag * y_odd_real
        
        # Combine using butterfly operations
        y_real[:, :W_half] = y_even_real + tw_odd_real
        y_imag[:, :W_half] = y_even_imag + tw_odd_imag
        y_real[:, W_half:] = y_even_real - tw_odd_real
        y_imag[:, W_half:] = y_even_imag - tw_odd_imag
        
        return y_real, y_imag
