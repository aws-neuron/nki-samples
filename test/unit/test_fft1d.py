"""

Unit tests for 1D FFT kernel
"""
import pytest
import numpy as np
import torch
import sys
import os

# Add contributed directory to path for importing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../contributed'))

from fft1d import fft1d


def cpu_golden_result_numpy(input_tensor):
    """
    Pure NumPy reference implementation for 1D FFT
    """
    return np.fft.fft(input_tensor, axis=1)


class TestFFT1D:
    
    @pytest.mark.parametrize("H, W", [
        # Execution test cases with various input sizes
        [64, 128],
        [128, 256],
        [100, 512],
        [128, 1024],
        [32, 2048],
        [50, 4096],
    ])
    def test_fft1d_execution(self, H, W):
        """Test that FFT executes successfully on various input sizes"""
        # Generate random input
        torch.manual_seed(42)
        input_tensor = torch.randn(H, W, dtype=torch.float32)
        
        # Run FFT - should complete without errors
        output = fft1d(input_tensor)
        
        # Verify output shape
        assert output.shape == (H, W), f"Output shape {output.shape} doesn't match input {(H, W)}"
        assert output.dtype == torch.complex64, f"Output dtype {output.dtype} should be complex64"
    
    @pytest.mark.parametrize("H, W", [
        # Numerical accuracy test cases
        [64, 128],
        [128, 256],
        [100, 512],
        [128, 1024],
        [32, 2048],
        [50, 4096],
        [1, 128],  # Minimum height
        [128, 128],  # Maximum height
    ])
    def test_fft1d_numerical(self, H, W):
        """Numerical accuracy test for 1D FFT"""
        # Generate random input
        torch.manual_seed(42)
        input_tensor = torch.randn(H, W, dtype=torch.float32)
        
        # Compute reference using NumPy
        input_np = input_tensor.numpy()
        expected_output = cpu_golden_result_numpy(input_np)
        
        # Run NKI kernel (fft1d handles device management internally)
        actual_output = fft1d(input_tensor)
        
        # Convert to numpy for comparison
        actual_np = actual_output.numpy()
        
        # Check numerical accuracy
        max_error = np.max(np.abs(actual_np - expected_output))
        rel_error = max_error / (np.max(np.abs(expected_output)) + 1e-10)
        
        assert rel_error < 0.01, f"Relative error {rel_error:.6%} exceeds 1% threshold"
    
    @pytest.mark.parametrize("H, W", [
        # Complex input test cases
        [64, 256],
        [128, 512],
    ])
    def test_fft1d_complex_input(self, H, W):
        """Test FFT with complex input"""
        # Generate complex input
        torch.manual_seed(42)
        real_part = torch.randn(H, W)
        imag_part = torch.randn(H, W)
        input_tensor = torch.complex(real_part, imag_part)
        
        # Compute reference
        input_np = input_tensor.numpy()
        expected_output = cpu_golden_result_numpy(input_np)
        
        # Run NKI kernel
        actual_output = fft1d(input_tensor)
        
        # Check accuracy
        actual_np = actual_output.numpy()
        max_error = np.max(np.abs(actual_np - expected_output))
        rel_error = max_error / (np.max(np.abs(expected_output)) + 1e-10)
        
        assert rel_error < 0.01, f"Relative error {rel_error:.6%} exceeds 1% threshold"
    
    def test_fft1d_invalid_width(self):
        """Test that invalid widths are rejected"""
        # Width not power of 2
        with pytest.raises(AssertionError):
            input_tensor = torch.randn(128, 100)
            fft1d(input_tensor)
        
        # Width too small
        with pytest.raises(AssertionError):
            input_tensor = torch.randn(128, 64)
            fft1d(input_tensor)
