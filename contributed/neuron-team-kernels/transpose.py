"""
Copyright (c) 2025, Amazon.com. All Rights Reserved

Kernels written by the AWS Neuron.

WARNING: These kernels:
   - Are tested only against internal nightly builds
   - May not be compatible with public NeuronSDK releases
   - Have not been extensively tested across all input configurations
   - Carry no compatibility guarantees
   - The behavior of these kernels may be modified without prior notice

Transposes kernels

"""
import numpy as np
import neuronxcc.nki as nki
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
from neuronxcc.nki.language import par_dim
from common import div_ceil, simplify_permute
from nki.language.constants import sizeinbytes

# helper method to invoke the transpose kernel
def transpose_to_last_dim(src, dim, dst=None):
  if dst is None:
    new_shape = get_3d_shape(src, dim)
    transposed_shape = (new_shape[0], new_shape[2], new_shape[1])
    dst = nl.ndarray(shape=transposed_shape, buffer=nl.hbm, dtype=src.dtype)
  transpose_to_last_dim_kernel(src, dim, dst)
  return dst


# return a 3d shape for transpose
# the second dim is the dim to sort and to be transposed
# the first and the thrid dim are flatten by the rest of the dims in btw
def get_3d_shape(ref, dim):
  new_shape = [int(np.prod(ref.shape[:dim])),
                ref.shape[dim],
                int(np.prod(ref.shape[dim+1:]))]
  return new_shape


# transpose the tensor's given dim to the last dim
def transpose_to_last_dim_kernel(ref, dim, dst):
  assert len(ref.shape) >= 2
  assert dim != len(ref.shape) - 1

  ref = ref.reshape(get_3d_shape(ref, dim))
  transposed_shape = (ref.shape[0], ref.shape[2], ref.shape[1])
  transpose_nonlocal = dst.reshape(transposed_shape)

  D0, B, N = ref.shape
  B_tile_size = min(128, B)
  N_tile_size = min(128, N)
  B_num_tiles = div_ceil(B, B_tile_size)
  N_num_tiles = div_ceil(N, N_tile_size)
  for d0 in nl.affine_range(D0):
    for b_out_tile in nl.affine_range(B_num_tiles):
      for n_out_tile in nl.affine_range(N_num_tiles):
        _local = nl.ndarray(shape=(B_tile_size, N_tile_size), 
                                      dtype=ref.dtype, buffer=nl.sbuf, name='local')
        transposed_local = nl.ndarray(shape=(par_dim(N_tile_size), B_tile_size), 
                                      dtype=ref.dtype, buffer=nl.sbuf, name='transposed_local')
        i = nl.arange(0, B_tile_size)[:, None]
        j = nl.arange(0, N_tile_size)[None, :]
        mask = (b_out_tile * B_tile_size + i < B) & (n_out_tile * N_tile_size + j < N)
        #TODO: maybe better performance by refetching the ref tensor
        _local[i, j] = nl.load(ref[d0, b_out_tile * B_tile_size + i, n_out_tile * N_tile_size + j], mask=mask)

        p = nl.arange(0, N_tile_size)[:, None]
        q = nl.arange(0, B_tile_size)[None, :]
        transposed_local[p, q] = nisa.nc_transpose(_local[i, j], mask=mask)

        mask = (b_out_tile * B_tile_size + q < B) & (n_out_tile * N_tile_size + p < N)
        nl.store(transpose_nonlocal[d0, n_out_tile * N_tile_size + p, b_out_tile * B_tile_size + q], transposed_local[p, q], mask=mask)


def _tiled_dve_transpose_10(img_ref):
  """
  Transposes the last two dims of the input tensor.
  [..., I, J] --> [..., J, I]

  Args:
      img_ref (_type_): Input HBM tensor with dims >= 2

  Returns:
      transposed HBM tensor
  """
  # Steps
  # 1. Allocate a output tile, temp, that fits the entirety of I dim and as much J dim that will fit in one partition, the dims are [P, j_tile_size, I]
  # 2. Load a tile of img_ref that fits the entirety of J dim and as much I dim that will fit in one partition, the dims are [P, i_tile_size, J]
  # 3. DVE (tensor copy) transpose on the part of img_tile that intersects with temp, store in temp.
  #     Now, a [j_tile_size, i_tile_size] tile of temp has correct data.
  # 4. Repeat steps 2. and 3. until the entirety of temp is correct
  # 5. Store temp on HBM
  # 6. Repeat with next output tile

  # TODO: Use nki.language.constants when bytes per partition is added.
  # Also this should be higher than the current 160k, but setting it higher
  # will cause an allocation error, which should not happen.
  MAX_F_SIZE = (100000 // sizeinbytes(img_ref.dtype)) // 2
  img_shape = img_ref.shape
  assert len(img_shape) >= 2, 'At least 2 dimensions needed for transpose'

  I = img_shape[-2]
  J = img_shape[-1]
  assert I < MAX_F_SIZE and J < MAX_F_SIZE, f"I, J larger than {MAX_F_SIZE} elems is not supported yet"

  i_tile_size = MAX_F_SIZE // J
  if i_tile_size >= I:
      i_tile_size = I
  num_i_tiles = div_ceil(I, i_tile_size)

  j_tile_size = MAX_F_SIZE // I
  if j_tile_size >= J:
      j_tile_size = J
  num_j_tiles = div_ceil(J, j_tile_size)

  # Flatten batch dimensions and put on partitions
  flattened_B_dim = int(np.prod([1] + list(img_shape[:-2])))
  MAX_P = nl.tile_size.pmax
  num_p_tiles = div_ceil(flattened_B_dim, MAX_P)

  img_ref = img_ref.reshape((flattened_B_dim, I, J))
  out = nl.ndarray(shape=(flattened_B_dim, J, I), dtype=img_ref.dtype, buffer=nl.hbm, name='transposed_img_ref')

  for p_tile in nl.affine_range(num_p_tiles):
      for j_tile in nl.affine_range(num_j_tiles):
          temp = nl.ndarray(shape=(MAX_P, j_tile_size * I), dtype=img_ref.dtype, buffer=nl.sbuf, name='transpose_temp')

          for i_tile in nl.affine_range(num_i_tiles):
              i_P, i_I, i_J = nl.mgrid[0:MAX_P, 0:i_tile_size, 0:J]
              img_tile = nl.load(img_ref[nl.ds(p_tile * MAX_P, MAX_P), nl.ds(i_tile * i_tile_size, i_tile_size), :],
                              mask=(p_tile*MAX_P+i_P < flattened_B_dim)&(i_tile*i_tile_size+i_I < I)).reshape((MAX_P, i_tile_size*J))

              i_P, i_I, i_J = nl.mgrid[0:MAX_P, 0:i_tile_size, 0:j_tile_size]
              temp[i_P, i_J*I + (i_I + i_tile*i_tile_size)] = nl.copy(img_tile[i_P, i_I*J + (i_J + j_tile*j_tile_size)],
                                                                      mask=(i_tile_size*i_tile+i_I < I) & \
                                                                          (j_tile_size*j_tile+i_J < J) & \
                                                                              (MAX_P*p_tile+i_P < flattened_B_dim))
          i_P, i_J, i_I = nl.mgrid[0:MAX_P, 0:j_tile_size, 0:I]
          nl.store(out[nl.ds(p_tile * MAX_P, MAX_P), nl.ds(j_tile * j_tile_size, j_tile_size)], temp.reshape((MAX_P, j_tile_size, I)), 
                  mask=(j_tile_size*j_tile+i_J < J) & (MAX_P*p_tile+i_P < flattened_B_dim))

  return out.reshape(list(img_shape[:-2]) + [J, I])


@nki.jit
def tiled_dve_transpose_210(img_ref, **kwargs):
  """
  Transposes the last dim and third last dim of the input tensor.

  Args:
      img_ref (_type_): Input HBM tensor with dims >= 3 and I, J, K are all F dims

  Returns:
      transposed HBM tensor
  """
  # TODO: Use nki.language.constants when bytes per partition is added.
  MAX_F_SIZE = (100000 // sizeinbytes(img_ref.dtype)) // 2

  img_shape = img_ref.shape
  assert len(img_shape) >= 3, 'At least 3 dimensions needed for transpose'

  I = img_shape[-3]
  J = img_shape[-2]
  K = img_shape[-1]
  assert I*J < MAX_F_SIZE and K*J < MAX_F_SIZE, f"This kernel does not support I*J or J*K larger than {MAX_F_SIZE}."

  i_tile_size = MAX_F_SIZE // (J * K)
  if i_tile_size >= I:
      i_tile_size = I
  num_i_tiles = div_ceil(I, i_tile_size)

  k_tile_size = MAX_F_SIZE // (I * J)
  if k_tile_size >= K:
      k_tile_size = K
  num_k_tiles = div_ceil(K, k_tile_size)

  # Flatten batch dimensions and put on partitions
  flattened_B_dim = int(np.prod([1] + list(img_shape[:-3])))
  MAX_P = nl.tile_size.pmax
  num_p_tiles = div_ceil(flattened_B_dim, MAX_P)

  img_ref = img_ref.reshape((flattened_B_dim, I, J, K))
  out = nl.ndarray(shape=(flattened_B_dim, K, J, I), dtype=img_ref.dtype, buffer=nl.hbm, name='transposed_img_ref')

  for p_tile in nl.affine_range(num_p_tiles):
    for k_tile in nl.affine_range(num_k_tiles):
      temp = nl.ndarray(shape=(MAX_P, k_tile_size * I * J), dtype=img_ref.dtype, buffer=nl.sbuf, name='transpose_temp')

      for i_tile in nl.affine_range(num_i_tiles):
        i_P, i_I, i_J, i_K = nl.mgrid[0:MAX_P, 0:i_tile_size, 0:J, 0:K]
        img_tile = nl.load(img_ref[nl.ds(p_tile * MAX_P, MAX_P), nl.ds(i_tile * i_tile_size, i_tile_size), :, :],
                        mask=(p_tile*MAX_P+i_P < flattened_B_dim)&(i_tile*i_tile_size+i_I < I)).reshape((MAX_P, i_tile_size*J*K))

        i_P, i_I, i_J, i_K = nl.mgrid[0:MAX_P, 0:i_tile_size, 0:J, 0:k_tile_size]
        temp[i_P, i_K*I*J + i_J*I + (i_I + i_tile*i_tile_size)] = nl.copy(img_tile[i_P, i_I*J*K + i_J*K + (i_K + k_tile*k_tile_size)],
                                                                mask=(i_tile_size*i_tile+i_I < I) & \
                                                                    (k_tile_size*k_tile+i_K < K) & \
                                                                        (MAX_P*p_tile+i_P < flattened_B_dim))

      i_P, i_K, i_J, i_I = nl.mgrid[0:MAX_P, 0:k_tile_size, 0:J, 0:I]
      nl.store(out[nl.ds(p_tile * MAX_P, MAX_P), nl.ds(k_tile * k_tile_size, k_tile_size)], temp.reshape((MAX_P, k_tile_size, J, I)), 
              mask=(k_tile_size*k_tile+i_K < K) & (MAX_P*p_tile+i_P < flattened_B_dim))

  return out.reshape(list(img_shape[:-3]) + [K, J, I])


@nki.jit
def tiled_dve_transpose_10(img_ref, permutation=None):
  assert permutation
  original_shape = img_ref.shape
  simple_perm, simple_shape = simplify_permute(original_shape, permutation)
  supported_perm = list(range(len(simple_perm)))
  supported_perm[-1], supported_perm[-2] = supported_perm[-2], supported_perm[-1]
  assert tuple(simple_perm) == tuple(supported_perm), 'This kernel only supports transposing last 2 dims'

  img_ref = img_ref.reshape(simple_shape)
  out = _tiled_dve_transpose_10(img_ref)
  out = out.reshape([original_shape[dim] for dim in permutation])
  return out
