"""
Copyright (c) 2025, Amazon.com. All Rights Reserved

Kernels written by the AWS Neuron.

WARNING: These kernels:
   - Are tested only against internal nightly builds
   - May not be compatible with public NeuronSDK releases
   - Have not been extensively tested across all input configurations
   - Carry no compatibility guarantees
   - The behavior of these kernels may be modified without prior notice

Common helper functions used by NKI kernels

"""
from collections import deque, Counter
from functools import lru_cache, reduce
from itertools import chain, combinations
from math import gcd
from operator import mul
from typing import Dict, Any, Iterable, Callable, Tuple, List, TypeVar
import numpy as np

def div_ceil(n, d):
  return (n + d - 1) // d

def normalize_dim(idx, rank):
  return idx if idx >= 0 else (rank + idx)

def n_elts(shape):
  return reduce(mul, shape, 1)

def simplify_permute(src_shape, permute):
  """
  Simplifies a permutation of a tensor by grouping contiguous dimensions.
  Returns a tuple containing the simplified permutation and source shape.
  Eg shape = [2, 4, 8, 16, 32]
     permute = [0, 1, 4, 2, 3]

     simple_permute = [0, 2, 1]
     simple_src_shape = [2*4, 8*16, 32]
  """
  assert len(src_shape) == len(permute)
  assert list(range(len(permute))) == sorted(permute)
  grouped_permute = []
  que = deque([permute[0]])
  for i in range(1, len(permute)):
    if permute[i] != que[-1] + 1:
      grouped_permute.append(tuple(que))
      que.clear()
    que.append(permute[i])
  grouped_permute.append(tuple(que))

  def argsort(seq):
    return sorted(range(len(seq)), key=seq.__getitem__)

  simple_permute = argsort([dims[0] for dims in grouped_permute])
  simple_permuted_shape = [int(np.prod([src_shape[dim] for dim in dims])) for dims in grouped_permute]
  simple_original_shape = [simple_permuted_shape[i] for i in simple_permute]
  return simple_permute, simple_original_shape