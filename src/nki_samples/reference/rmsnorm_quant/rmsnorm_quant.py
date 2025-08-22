import logging
from dataclasses import dataclass
from math import prod

from neuronxcc import nki
import neuronxcc.nki.compiler as ncc
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
import neuronxcc.nki.typing as nt

from ..util.common_types import NormType
from ..util.kernel_helpers import is_launched_as_spmd, is_rms_normalization
from .rmsnorm_quant_constants import RMSNormQuantConstants, build_constants
from .rmsnorm_quant_tile_info import RMSNormQuantTileInfo

"""
8-bit Quantization and RMS Normalization Kernel

This kernel performs quantization down to 8 bits along the last dimension of the input tensor.  What this means is that for each
given last dimension vector, the maximum absolute value is found and that is used in conjunction with the maximum values that can be
represented in the 8-bit data type to scale the values of that vector to fit within the range of the 8-bit data type.  The scale
factors for converting each vector back to 16 bits (i.e. dequantizing) are provided in the output tensor along with the quantized
vectors.

RMS normalization is an optional step when calling this kernel.  If it is desired, the RMS normalization is computed along each
of the last dimension vectors.  This happens prior to quantization.

This kernel accepts an input tensor of at minimum 2 dimensions.  We establish 2 terms here:
processing dimension - The dimension along which the kernel calculates normalization and quantization.
                       This is the last dimension of the input tensor.
outer dimension      - This dimension is comprised of all input tensor dimensions excluding the last dimension.

For example, if our input tensor has shape [W, X, Y, Z], we reshape the tensor to be [W * X * Y, Z] where
the processing dimension is Z and the outer dimension is W * X * Y.  This is done for a few reasons:
1. Since the code processes along the last dimension, there is no reason to require specific criteria for the other
   dimensions other than that there has to be at least one other dimension (i.e. the input tensor itself must be at least 2D).
   We then naturally support input tensors like ones of shape [batch size, sequence length, hidden size] without requiring
   such specific shapes.
2. When we collapse all major dimensions into this outer dimension and have one loop, we only have one boundary condition at the
   very end of the loop where compute is less efficient if the outer dimension is not a multiple of the tile size for that dimension.
   Contrast that with an example where we have a loop for batch size and a nested loop for sequence length.  If the batch size is
   greater than 1 and the sequence length is not a multiple of the tile size, we end up with the inefficient boundary condition
   for each batch loop.

To keep comments more abbreviated, we establish the following acronyms when describing things like shapes:
  OD  - Outer dimension
  PD  - Processing dimension
  ODT - Outer dimension tile
  PDT - Processing dimension tile
  PDS - Processing dimension with dequantizing scale
"""


@dataclass(frozen=True)
class RmsNormQuantKernelArgs:
  """RMS Norm Quantization Kernel arguments.

  Args:
    lower_bound (float): Non-negative float used for clipping input values and scale.
    norm_type (NormType): Normalization type to use [RMS_NORM, NO_NORM]
    eps (float): Epsilon value for numerical stability, model hyperparameter

  Raises:
    NotImplementedError: Raised when unsupported inputs are used.
    ValueError: Raised when invalid inputs are used.
  """

  lower_bound: float
  norm_type: NormType = NormType.RMS_NORM
  eps: float = 1e-6

  def __post_init__(self):
    if self.norm_type not in (NormType.NO_NORM, NormType.RMS_NORM):
      raise NotImplementedError(f"{self.norm_type.name} normalization is not supported")
    if self.lower_bound < 0:
      raise ValueError(f"Lower bound must be positive but got {self.lower_bound}")
    if self.eps < 0:
      raise ValueError(f"Epsilon must be positive but got {self.lower_bound}")

  def needs_rms_normalization(self) -> bool:
    return is_rms_normalization(self.norm_type)

  def has_lower_bound(self) -> bool:
    return self.lower_bound is not None and self.lower_bound > 0


# FIXME: add an optional argument to respect autocast options passed in by the user
@nki.compiler.enable_stack_allocator(log_level=logging.INFO)
@nki.compiler.skip_middle_end_transformations
@nki.jit
def rmsnorm_quant_kernel(hidden: nt.tensor, ln_w: nt.tensor, kargs: RmsNormQuantKernelArgs):
  """Entrypoint NKI kernel that performs one of the following:
      (1) perform RMSNorm and quantize the normalized hidden over the hidden dimension (H, or axis=-1).
      (2) quantize hidden over dimension H.

  The kernel supports no specialization, or specialization along 1 dimension (1D SPMD grid).

  Args:
    hidden (nl.ndarray): Input hidden states in [B, S, H] layout
    ln_w (nl.ndarray): Gamma multiplicative bias vector with [H] or [1, H] layout
    kargs (RmsNormQuantKernelArgs): See docstring for arguments

  Returns:
    Output tensor of shape [B, S, H + 4] on HBM. out[:, :, :H] of shape [B, S, H] stores the possibly
      normalized, and quantized tensor. out[:, :, H:] of shape [B, S, 4] stores 4 fp8 floats (for each unique
      batch and sequence length index) which can be reinterpreted as a fp32 dequantization scale.

  NOTE:
      The autocast argument may NOT be respected properly.
  """
  # Either we aren't sharding (grid_ndim == 0) or we can shard on a single dimension (grid_ndim == 1)
  grid_ndim = nl.program_ndim()
  assert grid_ndim == 0 or grid_ndim == 1, (
    f"RMSNorm quantization kernel only supports no specialization, or specialization along one axis."
  )
  # We need at least 2 dimensions on the input.  If there are N dimensions, the outer dimensions are considered to be
  # the first N - 1 dimensions and the dimension we process on is the Nth dimension (i.e. the last dimension).
  if len(hidden.shape) < 2:
    raise ValueError(f"Rank of hidden must be at least 2 but got {len(hidden.shape)}")

  # We can handle any input by processing along its last dimension and reshaping to collapse all other dimensions into one.
  # This way, we don't have to be so specific that we require inputs with shape [B, S, H] for example.
  tsr_proc_shape = _collapse_shape_major_dimensions(hidden.shape)
  # Build data structures with info that we need throughout the kernel
  tile_info = RMSNormQuantTileInfo.build(tsr_proc_shape)
  constants = build_constants(tile_info, kargs.eps, tsr_proc_shape)

  if kargs.needs_rms_normalization():
    # The shape of the RMS norm gamma tensor needs to either be [N] or [1, N] where
    # N is the size of the processing dimension.
    if len(ln_w.shape) not in (1, 2):
      raise ValueError(f"Rank of ln_w must be 1 or 2 but got {len(ln_w.shape)}")
    if ln_w.shape[-1] != constants.proc_dim_size:
      raise ValueError(f"ln_w vector length must equal {constants.proc_dim_size} but got {ln_w.shape[-1]}")

  # Create the output tensor with the same shape as the input tensor but with the
  # innermost dimension extended to hold the dequantizing scale factors.
  out_tsr_shape = hidden.shape[:-1] + (hidden.shape[-1] + constants.dequant_scale_size,)
  out_tsr_proc_shape = _collapse_shape_major_dimensions(out_tsr_shape)
  out_tsr_hbm = nl.ndarray(out_tsr_shape, dtype=constants.quant_data_type, buffer=nl.shared_hbm)

  # Set the input and output shapes up based on an outer dimension and the dimension that we process along
  in_tsr_hbm_view = hidden.reshape(tsr_proc_shape)
  out_tsr_hbm_view = out_tsr_hbm.reshape(out_tsr_proc_shape)

  # Check if we were launched for single program multiple data and if so, call the code to do the appropriate sharding.
  # Otherwise, just do a single invocation of the kernel to process all the data.
  if is_launched_as_spmd():
    _rmsnorm_quant_sharded_kernel(kargs, in_tsr_hbm_view, ln_w, out_tsr_hbm_view)
  else:
    _rmsnorm_quant_single_core_kernel(kargs, tile_info, constants, in_tsr_hbm_view, ln_w, out_tsr_hbm_view)

  return out_tsr_hbm


#
# Local private helpers


def _collapse_shape_major_dimensions(shape: tuple[int, ...]) -> tuple[int, int]:
  """
  We process along the last dimension only so just multiply all dimensions leading up
  to the last one together.  Our processing loop can just traverse this resulting outer
  dimension and process each vector (i.e. the last dimension of the input).
  """
  return (prod(shape[:-1]), shape[-1])


@nki.jit(
  mode='trace'
)  # TODO: Remove when NKI-917 is fixed (range() loops unrolled when loops in functions beyond the top level of the kernel)
def _load_input_tensor_tile(
  tile_info: RMSNormQuantTileInfo,
  constants: RMSNormQuantConstants,
  in_tsr_hbm: nt.tensor,
  outer_dim_tile_num: int,
  output_tile_sbuf: nt.tensor,
):
  """
  Load a single tile from the input tensor in HBM into SBUF ensuring that we use a mask for the
  case where the outer dimension is not a multiple of the tile size.  The shapes are:
    in_tsr_hbm       - [OD, PD]
    output_tile_sbuf - [ODT, PD]
  NOTE: We don't need a mask for the processing dimension here because we aren't tiling on that
        dimension when loading.
  """
  p_idx, f_idx = nl.mgrid[0 : tile_info.outer_dim_tile.tile_size, 0 : constants.proc_dim_size]
  nisa.dma_copy(
    dst=output_tile_sbuf[p_idx, f_idx],
    src=in_tsr_hbm[tile_info.outer_dim_tile.get_tile_indices(outer_dim_tile_num, p_idx), f_idx],
    mask=tile_info.outer_dim_tile.get_tile_mask(outer_dim_tile_num, p_idx, max_exclusive=constants.outer_dim_size),
  )


@nki.jit(
  mode='trace'
)  # TODO: Remove when NKI-917 is fixed (range() loops unrolled when loops in functions beyond the top level of the kernel)
def _store_tensor_tile_and_dequant_scales(
  tile_info: RMSNormQuantTileInfo,
  constants: RMSNormQuantConstants,
  outer_dim_tile_num: int,
  in_tile_sbuf: nt.tensor,
  dequant_scales_tile_sbuf: nt.tensor,
  out_tsr_hbm: nt.tensor,
):
  """
  Store a single computed tile into the output tensor in HBM from SBUF ensuring that we use a mask for the
  case where the outer dimension is not a multiple of the tile size.  Each of the last dimensions in the
  output tensor is structured like this:
  -----------------------------------
  | ... Computed elements ... | DQS |
  -----------------------------------
  | ... Computed elements ... | DQS |
  -----------------------------------
  |              .                  |
                 .
  |              .                  |
  -----------------------------------
  | ... Computed elements ... | DQS |
  -----------------------------------

  Where DQS is the dequantizing scale for the computed elements in a given last dimension vector.

  The shapes are:
  in_tile_sbuf             - [ODT, PD]
  dequant_scales_tile_sbuf - [ODT, 1]
  out_tsr_hbm              - [OD, PDS]

  NOTE: We don't need a mask for the processing dimension here because we aren't tiling on that
        dimension when storing.
  """
  # Store the quantized data
  p_e_idx, f_e_idx = nl.mgrid[0 : tile_info.outer_dim_tile.tile_size, 0 : constants.proc_dim_size]
  nisa.dma_copy(
    dst=out_tsr_hbm[tile_info.outer_dim_tile.get_tile_indices(outer_dim_tile_num, p_e_idx), f_e_idx],
    src=in_tile_sbuf[p_e_idx, f_e_idx],
    mask=tile_info.outer_dim_tile.get_tile_mask(outer_dim_tile_num, p_e_idx, max_exclusive=constants.outer_dim_size),
  )
  # Store the dequantization weights
  p_s_idx, f_s_idx = nl.mgrid[0 : tile_info.outer_dim_tile.tile_size, 0 : constants.dequant_scale_size]
  nisa.dma_copy(
    dst=out_tsr_hbm[
      tile_info.outer_dim_tile.get_tile_indices(outer_dim_tile_num, p_s_idx), constants.proc_dim_size + f_s_idx
    ],
    src=dequant_scales_tile_sbuf.view(nl.float8_e4m3)[p_s_idx, f_s_idx],
    mask=tile_info.outer_dim_tile.get_tile_mask(outer_dim_tile_num, p_s_idx, max_exclusive=constants.outer_dim_size),
  )


@nki.jit(
  mode='trace'
)  # TODO: Remove when NKI-917 is fixed (range() loops unrolled when loops in functions beyond the top level of the kernel)
def _rms_normalize_tile(
  tile_info: RMSNormQuantTileInfo,
  constants: RMSNormQuantConstants,
  outer_dim_tile_num: int,
  in_tile_sbuf: nt.tensor,
  gamma_hbm: nt.tensor,
  squared_in_tsr_sbuf: nt.tensor,
  broadcasted_gamma_sbuf: nt.tensor,
  inverse_rms_scale_sbuf: nt.tensor,
):
  """
  Compute RMS normalization along the last dimension for a tile.  The shapes are:
    in_tile_sbuf           - [ODT, PD]
    gamma_hbm              - [1, PD]
    squared_in_tsr_sbuf    - [ODT, PD]
    broadcasted_gamma_sbuf - [ODT, PDT]
    inverse_rms_scale_sbuf - [ODT, 1]

  NOTE: The computation is stored in place in in_tile_sbuf.
  """
  # Alias these to cut down on the verbosity in the code
  outer_dim_tile_size = tile_info.outer_dim_tile.tile_size
  proc_dim_tile_size = tile_info.proc_dim_tile.tile_size

  # We need a number of indices for the various computations below
  # Input index
  p_in_idx, f_in_idx = nl.mgrid[0:outer_dim_tile_size, 0 : constants.proc_dim_size]
  # General purpose column vector index of shape [outer_dim_tile_size, 1]
  p_cvec_idx, f_cvec_idx = nl.mgrid[0:outer_dim_tile_size, 0:1]
  # Processing dimension tile index
  p_pd_t_idx, f_pd_t_idx = nl.mgrid[0:outer_dim_tile_size, 0:proc_dim_tile_size]
  # Gamma index
  p_gam_idx, f_gam_idx = nl.mgrid[0:1, 0:proc_dim_tile_size]
  # Ones vector index
  p_ones_idx, f_ones_idx = nl.mgrid[
    0 : constants.pe_broadcast_ones_vector_sbuf.shape[0], 0 : constants.pe_broadcast_ones_vector_sbuf.shape[1]
  ]

  # Find the sum of the squares of all elements in the processing dimension and store that in inverse_rms_scale_sbuf.
  # NOTE: squared_in_tsr_sbuf is each individual element squared.  We don't use that result data but have to provide storage for it.
  squared_in_tsr_sbuf[p_in_idx, f_in_idx] = nisa.activation_reduce(
    op=nl.square,
    data=in_tile_sbuf[p_in_idx, f_in_idx],
    reduce_op=nl.add,
    reduce_res=inverse_rms_scale_sbuf[p_cvec_idx, f_cvec_idx],
    bias=constants.outer_dim_tile_zero_bias_vector_sbuf[p_cvec_idx, f_cvec_idx],
    scale=1.0,
    mask=tile_info.outer_dim_tile.get_tile_mask(outer_dim_tile_num, p_in_idx, max_exclusive=constants.outer_dim_size),
  )
  # Calculate the reciprocal of the RMS being sure to add epsilon for numerical stability.  Store the result back in inverse_rms_scale_sbuf.
  inverse_rms_scale_sbuf[p_cvec_idx, f_cvec_idx] = nisa.activation(
    op=nl.rsqrt,
    data=inverse_rms_scale_sbuf[p_cvec_idx, f_cvec_idx],
    bias=constants.rmsn_eps_bias_sbuf[p_cvec_idx, f_cvec_idx],
    scale=1 / constants.proc_dim_size,
    mask=tile_info.outer_dim_tile.get_tile_mask(outer_dim_tile_num, p_cvec_idx, max_exclusive=constants.outer_dim_size),
  )

  # Load the gamma values
  p_gamma_idx, f_gamma_idx = nl.mgrid[0 : gamma_hbm.shape[0], 0 : gamma_hbm.shape[1]]
  broadcasted_gamma_sbuf = nl.ndarray((gamma_hbm.shape[0], gamma_hbm.shape[1]), gamma_hbm.dtype, buffer=nl.sbuf)
  nisa.dma_copy(
    dst=broadcasted_gamma_sbuf[p_gamma_idx, f_gamma_idx],
    src=gamma_hbm[p_gamma_idx, f_gamma_idx],
  )

  # Loop and process all the tiles in the processing dimension
  for proc_dim_tile_num in nl.sequential_range(
    tile_info.proc_dim_tile.tile_count, directives=ncc.multi_buffer(constants.num_hw_psum_banks)
  ):
    proc_dim_idx = tile_info.proc_dim_tile.get_tile_indices(proc_dim_tile_num, f_pd_t_idx)
    outer_dim_mask = tile_info.outer_dim_tile.get_tile_mask(
      outer_dim_tile_num, p_pd_t_idx, max_exclusive=constants.outer_dim_size
    )
    proc_dim_mask = tile_info.proc_dim_tile.get_tile_mask(
      proc_dim_tile_num, f_pd_t_idx, max_exclusive=constants.proc_dim_size
    )
    agg_mask = outer_dim_mask & proc_dim_mask
    gam_proc_dim_idx = tile_info.proc_dim_tile.get_tile_indices(proc_dim_tile_num, f_gam_idx)
    gam_mask = tile_info.proc_dim_tile.get_tile_mask(
      proc_dim_tile_num, f_gam_idx, max_exclusive=constants.proc_dim_size
    )

    # Matrix multiply the ones vector by the gamma values to get the values broadcast across all partitions.  We use the PE to do this because
    # it allows us to utilize this engine here while saving cycles on the vector engine for other work.  We can overlap PE computations with
    # vector engine and activation engine computations.
    broadcasted_gamma_psum = nl.ndarray((outer_dim_tile_size, proc_dim_tile_size), nl.float32, buffer=nl.psum)
    broadcasted_gamma_psum[p_pd_t_idx, f_pd_t_idx] = nisa.nc_matmul(
      constants.pe_broadcast_ones_vector_sbuf[p_ones_idx, f_ones_idx],
      broadcasted_gamma_sbuf[p_gam_idx, gam_proc_dim_idx][gam_mask],
    )

    # Apply gamma and inverse_rms_scale
    in_tile_sbuf[p_pd_t_idx, proc_dim_idx] = nisa.scalar_tensor_tensor(
      data=in_tile_sbuf[p_pd_t_idx, proc_dim_idx],
      op0=nl.multiply,
      operand0=inverse_rms_scale_sbuf[p_cvec_idx, f_cvec_idx],
      op1=nl.multiply,
      operand1=broadcasted_gamma_psum[p_pd_t_idx, f_pd_t_idx],
      mask=agg_mask,
    )


@nki.jit(
  mode='trace'
)  # TODO: Remove when NKI-917 is fixed (range() loops unrolled when loops in functions beyond the top level of the kernel)
def _quantize_tile(
  kargs: RmsNormQuantKernelArgs,
  tile_info: RMSNormQuantTileInfo,
  constants: RMSNormQuantConstants,
  outer_dim_tile_num: int,
  in_tile_sbuf: nt.tensor,
  out_tile_sbuf: nt.tensor,
  out_dequant_scales_sbuf: nt.tensor,
):
  """
  Compute quantization and dequantization scales along the last dimension for a tile.  The shapes are:
    in_tile_sbuf            - [ODT, PD]
    out_tile_sbuf           - [ODT, PD]
    out_dequant_scales_sbuf - [ODT, 1]
  """
  # Alias these to cut down on the verbosity in the code
  outer_dim_tile_size = tile_info.outer_dim_tile.tile_size

  p_in_idx, f_in_idx = nl.mgrid[0:outer_dim_tile_size, 0 : constants.proc_dim_size]
  p_vec_idx, f_vec_idx = nl.mgrid[0:outer_dim_tile_size, 0:1]

  # We need these masks for the case where the outer dimension is not a multiple of the tile size
  in_mask = tile_info.outer_dim_tile.get_tile_mask(outer_dim_tile_num, p_in_idx, max_exclusive=constants.outer_dim_size)
  vec_mask = tile_info.outer_dim_tile.get_tile_mask(
    outer_dim_tile_num, p_vec_idx, max_exclusive=constants.outer_dim_size
  )

  # abs_tile_sbuf stores the absolute values of the tile being processed.  We don't end up using these values but need a place to store them.
  # out_dequant_scales_sbuf stores abs_group reduced over abs_group's last dimension (i.e. the processing dimension)
  # to get the maximum absolute value.
  abs_tile_sbuf = nl.ndarray(in_tile_sbuf.shape, dtype=constants.compute_data_type, buffer=nl.sbuf)
  abs_tile_sbuf[p_in_idx, f_in_idx] = nisa.tensor_scalar_reduce(
    data=in_tile_sbuf[p_in_idx, f_in_idx],
    op0=nl.abs,
    operand0=0.0,
    reduce_op=nl.max,
    reduce_res=out_dequant_scales_sbuf[p_vec_idx, f_vec_idx],
    mask=in_mask,
  )

  if kargs.has_lower_bound():
    # Clip out_dequant_scales_sbuf to the range [0, lower_bound].
    out_dequant_scales_sbuf[p_vec_idx, f_vec_idx] = nisa.tensor_scalar(
      data=out_dequant_scales_sbuf[p_vec_idx, f_vec_idx], op0=nl.minimum, operand0=kargs.lower_bound, mask=vec_mask
    )

    # Clip the tile being processed in range [-lower_bound, lower_bound].
    in_tile_sbuf[p_in_idx, f_in_idx] = nisa.tensor_scalar(
      data=in_tile_sbuf[p_in_idx, f_in_idx],
      op0=nl.minimum,
      operand0=kargs.lower_bound,
      op1=nl.maximum,
      operand1=-kargs.lower_bound,
      mask=in_mask,
    )

  # Compute absolute maximum / _FP8_RANGE along each processing dimension to get the dequantization scales.
  out_dequant_scales_sbuf[p_vec_idx, f_vec_idx] = nisa.activation(
    op=nl.copy,
    data=out_dequant_scales_sbuf[p_vec_idx, f_vec_idx],
    scale=1 / constants.quant_data_type_range,
    bias=constants.outer_dim_tile_zero_bias_vector_sbuf[p_vec_idx, f_vec_idx],
    mask=vec_mask,
  )

  # Clamp out_dequant_scales_sbuf to _MIN_DEQUANT_SCALE_VAL for numerical stability reasons.  Basically, it keeps tiny
  # values from exploding in size when we take the reciprocal of them.
  out_dequant_scales_sbuf[p_vec_idx, f_vec_idx] = nisa.tensor_scalar(
    data=out_dequant_scales_sbuf[p_vec_idx, f_vec_idx],
    op0=nl.maximum,
    operand0=constants.min_dequant_scale_value,
    mask=vec_mask,
  )

  # Get the reciprocol of our dequantization scales to get the quantization scales
  quant_scales_sbuf = nl.ndarray(out_dequant_scales_sbuf.shape, dtype=nl.float32, buffer=nl.sbuf)
  quant_scales_sbuf[p_vec_idx, f_vec_idx] = nisa.reciprocal(
    out_dequant_scales_sbuf[p_vec_idx, f_vec_idx], mask=vec_mask
  )

  # Apply quantization scales to get the quantized result.
  out_tile_sbuf[p_in_idx, f_in_idx] = nisa.tensor_scalar(
    data=in_tile_sbuf[p_in_idx, f_in_idx],
    op0=nl.multiply,
    operand0=quant_scales_sbuf[p_vec_idx, f_vec_idx],
    mask=in_mask,
  )


@nki.jit(
  mode='trace'
)  # TODO: Remove when NKI-917 is fixed (range() loops unrolled when loops in functions beyond the top level of the kernel)
def _rmsnorm_quant_single_core_kernel(
  kargs: RmsNormQuantKernelArgs,
  tile_info: RMSNormQuantTileInfo,
  constants: RMSNormQuantConstants,
  in_tsr_hbm: nt.tensor,
  rmsn_gamma_hbm: nt.tensor,
  out_tsr_hbm: nt.tensor,
):
  """
  Process all tiles along the outer dimension by applying optional RMS normalization, quantizing the data, and storing the results.
  The shapes are:
    in_tsr_hbm     - [OD, PD]
    rmsn_gamma_hbm - [1, PD] or [PD]
    out_tsr_hbm    - [OD, PDS]

    Where DIM1 - DIMN are the major dimensions that ultimately are combined to be the outer dimension in this kernel.
  """
  if kargs.needs_rms_normalization():
    # Ensure the shape that the code requires
    rmsn_gamma_hbm_view = rmsn_gamma_hbm.reshape((1, constants.proc_dim_size))

  # Loop over all the tiles in the outer dimension and process them one at a time
  for outer_tile_num in range(tile_info.outer_dim_tile.tile_count):
    if kargs.needs_rms_normalization():
      # Declare some allocations required throughout RMS norm calculations
      squared_in_tsr_sbuf = nl.ndarray(
        (tile_info.outer_dim_tile.tile_size, constants.proc_dim_size), dtype=nl.float32, buffer=nl.sbuf
      )
      broadcasted_gamma_sbuf = nl.ndarray(
        (tile_info.outer_dim_tile.tile_size, tile_info.proc_dim_tile.tile_size), dtype=nl.bfloat16, buffer=nl.sbuf
      )
      inverse_rms_scale_sbuf = nl.ndarray((tile_info.outer_dim_tile.tile_size, 1), dtype=nl.float32, buffer=nl.sbuf)

    quant_tile_sbuf = nl.ndarray(
      (tile_info.outer_dim_tile.tile_size, constants.proc_dim_size), dtype=constants.quant_data_type, buffer=nl.sbuf
    )
    in_tile_sbuf = nl.ndarray(
      (tile_info.outer_dim_tile.tile_size, constants.proc_dim_size), dtype=constants.compute_data_type, buffer=nl.sbuf
    )
    dequant_scales_tile_sbuf = nl.ndarray((tile_info.outer_dim_tile.tile_size, 1), dtype=nl.float32, buffer=nl.sbuf)

    # Conceptually, our loop is simple:
    #   - Load a tile into SBUF
    #   - Apply RMS normalization if the caller has elected to do so
    #   - Quantize the tile
    #   - Store the resulting quantized tile data along with the associated dequantization scales into HBM
    _load_input_tensor_tile(tile_info, constants, in_tsr_hbm, outer_tile_num, in_tile_sbuf)

    if kargs.needs_rms_normalization():
      _rms_normalize_tile(
        tile_info,
        constants,
        outer_tile_num,
        in_tile_sbuf,
        rmsn_gamma_hbm_view,
        squared_in_tsr_sbuf,
        broadcasted_gamma_sbuf,
        inverse_rms_scale_sbuf,
      )

    _quantize_tile(kargs, tile_info, constants, outer_tile_num, in_tile_sbuf, quant_tile_sbuf, dequant_scales_tile_sbuf)

    _store_tensor_tile_and_dequant_scales(
      tile_info, constants, outer_tile_num, quant_tile_sbuf, dequant_scales_tile_sbuf, out_tsr_hbm
    )


@nki.jit(
  mode='trace'
)  # TODO: Remove when NKI-917 is fixed (range() loops unrolled when loops in functions beyond the top level of the kernel)
def _rmsnorm_quant_sharded_kernel(
  kargs: RmsNormQuantKernelArgs, in_tsr_hbm: nt.tensor, rmsn_gamma_hbm: nt.tensor, out_tsr_hbm: nt.tensor
):
  """
  We support a 1D launch grid only (or no launch grid at all).  If there is a dimension in the launch grid then however many programs
  there are in that dimension dictates our number of shards since we will launch the kernel once per program and each launch of the
  kernel needs a shard to process.

  We shard along the outer dimension since each processing dimension vector can be processed independently.  We calculate references
  to the original input and output tensors based on the sharding and then pass those references into the core of the kernel.
  """
  outer_dim, _ = in_tsr_hbm.shape
  num_shards = nl.num_programs(axes=0)
  nominal_shard_size = outer_dim // num_shards
  shard_id = nl.program_id(axis=0)

  # Allow outer dimensions that are not a multiple of the number of shards.  For the last shard,
  # we have to calculate the remaining size for the shard.  Otherwise, it is just the
  # nominal size.
  if shard_id == num_shards - 1:
    shard_size = outer_dim - nominal_shard_size * (num_shards - 1)
  else:
    shard_size = nominal_shard_size
  shard_offset = shard_id * nominal_shard_size

  outer_dim_idx = nl.ds(shard_offset, shard_size)
  in_tile_shard_hbm = in_tsr_hbm[outer_dim_idx, :]
  out_tile_shard_hbm = out_tsr_hbm[outer_dim_idx, :]
  # Figure out the shape in terms of [outer dim, processing dim]
  tsr_proc_shape = _collapse_shape_major_dimensions(in_tile_shard_hbm.shape)
  # Build data structures with info that we need throughout the kernel
  tile_info = RMSNormQuantTileInfo.build(tsr_proc_shape)
  constants = build_constants(tile_info, kargs.eps, tsr_proc_shape)

  return _rmsnorm_quant_single_core_kernel(
    kargs, tile_info, constants, in_tile_shard_hbm, rmsn_gamma_hbm, out_tile_shard_hbm
  )
