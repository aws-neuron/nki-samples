import numpy as np
from typing import NamedTuple

from neuronxcc import nki
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
import neuronxcc.nki.typing as nt
from neuronxcc.nki.language.constants import sizeinbytes
from .rmsnorm_quant_tile_info import RMSNormQuantTileInfo


#
#
# Primary tuple that holds miscellaneous constants required by the kernel
#
class RMSNormQuantConstants(NamedTuple):
  num_hw_psum_banks: int
  # Compute data type used for matmuls, etc.
  compute_data_type: np.dtype
  # Data type used for quantizing the inputs along with its range
  quant_data_type: np.dtype
  quant_data_type_range: int
  # The number of elements in the output tensor used to store each dequantizing scale factor
  dequant_scale_size: int
  # A small constant used to put an upper bound on the quantization scales for numerical stability.  This number is a minimum
  # applied to the dequantization scales and serves our purpose since quantization scales are computed as 1 / dequantization scale.
  min_dequant_scale_value: float
  # Bias vector of zeros used for activation functions
  outer_dim_tile_zero_bias_vector_sbuf: nt.tensor
  # Epsilon vector used for numerical stability in RMS norm
  rmsn_eps_bias_sbuf: nt.tensor
  # Ones vector for broadcasting via the PE
  pe_broadcast_ones_vector_sbuf: nt.tensor
  # The dimension we use for processing
  outer_dim_size: int
  proc_dim_size: int


# Factory method
# ONLY CONSTRUCT THIS USING THE FACTORY METHOD BELOW


@nki.jit(
  mode='trace'
)  # TODO: Remove when NKI-917 is fixed (range() loops unrolled when loops in functions beyond the top level of the kernel)
def build_constants(
  tile_info: RMSNormQuantTileInfo, eps: float, processing_shape: tuple[int, int]
) -> 'RMSNormQuantConstants':
  # TODO: Get this constant from the NKI API once it is available
  num_hw_psum_banks = 8
  # Data types
  compute_data_type = nl.bfloat16
  quant_data_type = nl.float8_e4m3
  # Range calculation: 3 bits of fraction, max exponent is (2^4 - 1) - 2^3 = 7...  2 ^ 7 * (1 + (2^7 / 2^8)) = 240
  quant_data_type_range = 240
  # The dequantizing scale factors are added to the end of each processed dimension.  The unit of this constant
  # is in output tensor elements.  Each dequantizing scale factor is a 32-bit float and the element size of
  # quant_data_type (i.e. the output type) is 8 bits.  Therefore, it takes 4 elements to store each dequantizing factor.
  # We do the math here for illustrative purposes plus the assert as a sanity check.
  assert sizeinbytes(np.float32) % sizeinbytes(quant_data_type) == 0
  dequant_scale_size = sizeinbytes(np.float32) // sizeinbytes(quant_data_type)
  # Used for numerical stability for quantizing
  min_dequant_scale_value = 1e-6
  # We need a zero bias vector for activations to work around a runtime issue when no bias vector is supplied to the activation method
  bias_vector_sbuf = nl.zeros((tile_info.outer_dim_tile.tile_size, 1), nl.float32, buffer=nl.sbuf)
  # Epsilon is added to values across the outer dimension tile for RMS norm
  rmsn_eps_bias_sbuf = nisa.memset((tile_info.outer_dim_tile.tile_size, 1), value=eps, dtype=compute_data_type)
  # Used for broadcasting using the PE array
  ones_vector_sbuf = nl.ones((1, nl.tile_size.pmax), dtype=compute_data_type, buffer=nl.sbuf)

  return RMSNormQuantConstants(
    num_hw_psum_banks,
    compute_data_type,
    quant_data_type,
    quant_data_type_range,
    dequant_scale_size,
    min_dequant_scale_value,
    bias_vector_sbuf,
    rmsn_eps_bias_sbuf,
    ones_vector_sbuf,
    processing_shape[0],
    processing_shape[1],
  )
