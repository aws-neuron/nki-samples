from typing import Callable

import neuronxcc.nki.language as nl
from .common_types import NormType
from neuronxcc.nki.compiler.backends.neuron.tensor import local_tile

def get_ceil_quotient(numerator: int, denominator: int) -> int:
  return (numerator + denominator - 1) // denominator


def get_ceil_aligned_size(size: int, alignment_multiple: int) -> int:
  return get_ceil_quotient(size, alignment_multiple) * alignment_multiple


def is_launched_as_spmd() -> bool:
  return nl.program_ndim() != 0 and nl.num_programs(axes=0) > 1


def is_rms_normalization(norm_type: NormType) -> bool:
  return norm_type in [NormType.RMS_NORM, NormType.RMS_NORM_SKIP_GAMMA]


def normalization_uses_weights(norm_type: NormType) -> bool:
  return norm_type in [NormType.RMS_NORM, NormType.LAYER_NORM]
