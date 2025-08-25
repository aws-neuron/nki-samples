"""
Copyright (c) 2025, Amazon.com. All Rights Reserved

Kernels written by the AWS Neuron Team.

WARNING: These kernels:
   - Are tested only against internal nightly compiler builds
   - May rely on internal compiler feature/flags and not be compatible with public NeuronSDK
   - Have not been extensively tested across all input configurations
   - Carry no compatibility guarantees
   - The behavior of these kernels may be modified without prior notice

Flash Paged Attention kernels with variable-length sequence inputs.

"""

from collections import defaultdict

import numpy as np
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa
from neuronxcc.nki.language import par_dim
from neuronxcc.nki.isa.constants import oob_mode

from constants import B_P_SIZE


def ceil_div(a, b):
    return (a + b - 1) // b


def pad_to_multiple(a, b):
    return ceil_div(a, b) * b


def is_power_of_2(x):
    return x > 0 and (x & (x - 1)) == 0


def load_indices(indices_hbm):
    """
    Load a 2D indices array of shape [num_tiles, num_indices] from HBM to SBUF

    To map num_tiles to SBUF partition dimension, this function automatically partitions num_tiles
    with partition_size set to min(num_tiles, B_P_SIZE=128)

    Output SBUF tensor shape:
      [par_dim(partition_size), ceil_div(num_tiles, partition_size), num_indices]
    """
    num_tiles, num_indices = indices_hbm.shape
    partition_size = min(B_P_SIZE, num_tiles)
    num_partitions = ceil_div(num_tiles, partition_size)
    indices_sbuf = nl.zeros(
        (par_dim(partition_size), num_partitions, num_indices),
        dtype=indices_hbm.dtype,
    )
    for i in nl.affine_range(num_partitions):
        i_p = nl.arange(partition_size)[:, None]
        i_f = nl.arange(num_indices)[None, :]
        indices_sbuf[i_p, i, i_f] = nl.load(
            indices_hbm[i_p + i * partition_size, i_f],
            mask=(i_p + i * partition_size < num_tiles),
        )
    return indices_sbuf


def load_indices_for_loop_step(
    indices_hbm,
    loop_index,
    step_size,
    partition_size=None,
    partition_size_iota=None,
    partition_offset_iota=None,
):
    """
    Load a 2D indices array with dim 0 range [loop_index * step_size,
    (loop_index + 1) * size) from HBM with start offset to SBUF

    To map num_tiles to SBUF partition dimension, this function automatically
    partitions num_tiles with partition_size set to min(step_size, B_P_SIZE=128)

    Output SBUF tensor shape:
    [par_dim(partition_size), ceil_div(size, partition_size), num_indices]
    """

    _, num_indices = indices_hbm.shape
    if partition_size is None:
        partition_size = min(B_P_SIZE, step_size)
    else:
        assert (
            partition_size <= B_P_SIZE
        ), f"Expect {partition_size=} <= {B_P_SIZE=}"
    assert (
        step_size % partition_size == 0
    ), f"Expect {step_size=} % {partition_size=} == 0"
    num_partitions = step_size // partition_size

    indices_sbuf = nl.ndarray(
        (par_dim(partition_size), num_partitions, num_indices),
        dtype=indices_hbm.dtype,
    )
    if partition_size_iota is None:
        partition_size_iota = nisa.iota(
            nl.arange(partition_size)[None, :],
            dtype=nl.uint32,
        )
    base_offsets = nisa.tensor_tensor(
        partition_size_iota,
        nisa.activation(
            nl.copy,
            loop_index,
            scale=float(step_size),
            dtype=nl.uint32,
        ),
        nl.add,
        engine=nisa.vector_engine,
    )
    base_offsets_t = nl.ndarray((partition_size, 1), dtype=nl.uint32)
    PF_transpose_with_PE(src=base_offsets, out=base_offsets_t)
    if num_partitions > 1:
        if partition_offset_iota is None:
            partition_offset_iota = nisa.iota(
                nl.arange(num_partitions)[None, :] * partition_size,
                dtype=nl.uint32,
            )

        if nisa.get_nc_version() == nisa.nc_version.gen3:
            partition_offsets_br = nl.broadcast_to(
                partition_offset_iota,
                shape=(partition_size, num_partitions),
            )
        else:
            partition_offsets_br = nl.ndarray(
                (partition_size, num_partitions),
                dtype=nl.uint32,
            )
            broadcast_partition_with_PE(
                src=partition_offset_iota,
                out=partition_offsets_br,
                out_in_psum=False,
            )
        offsets = nisa.tensor_tensor(
            base_offsets_t,
            partition_offsets_br,
            nl.add,
            engine=nisa.vector_engine,
        )
    else:
        offsets = base_offsets_t
    for i in nl.affine_range(num_partitions):
        i_p = nl.arange(partition_size)[:, None]
        i_f = nl.arange(num_indices)[None, :]
        indices_sbuf[i_p, i, i_f] = nl.load(
            indices_hbm[offsets[i_p, i], i_f],
            mode=oob_mode.skip,
        )
    return indices_sbuf


def transform_to_vector_dge_layout(
    indices_in, indices_out, partition_size=None, identity_for_transpose=None
):
    """
    Transpose an tile of shape [tile_size, num_indices] so that num_indices is mapped to partition
    dimension and perform partition with partition_size=min(num_indices, B_P_SIZE=128)

    indices_in:
      [par_dim(tile_size), num_indices]
    indices_out:
      [par_dim(partition_size), ceil_div(num_indices, partition_size), tile_size]
    """
    tile_size, num_indices = indices_in.shape
    if partition_size is None:
        partition_size = min(num_indices, B_P_SIZE)
    else:
        assert (
            partition_size <= B_P_SIZE
        ), f"Expect {partition_size=} <= {B_P_SIZE=}"
    num_partitions = ceil_div(num_indices, partition_size)
    assert indices_out.shape == (
        partition_size,
        num_partitions,
        tile_size,
    )
    for i in nl.affine_range(num_partitions):
        PF_transpose_with_PE(
            indices_in[:, nl.ds(i * partition_size, partition_size)],
            indices_out[:, i, :],
            identity_for_transpose=identity_for_transpose,
        )


def PF_transpose_with_PE_integer(src, out):
    """
    Perform int32 P-F Transpose with PE. Lower into 4 uint8 P-F transpose with reinterpret cast
    """
    assert nisa.get_nc_version() == nisa.nc_version.gen2
    # lower as 1/2/4 uint8 matmul
    assert src.dtype == out.dtype
    assert src.dtype == nl.int32 or src.dtype == nl.uint32
    p, f = src.shape
    nbytes = src.itemsize
    if nbytes > 1:
        src_copy = nl.copy(src)
    else:
        src_copy = src
    src_reinterpreted = src_copy.view(nl.uint8)
    out_reinterpreted = out.view(nl.uint8)
    for i in nl.affine_range(nbytes):
        out_psum = nl.ndarray((par_dim(f), p), dtype=nl.int32, buffer=nl.psum)
        i_p = nl.arange(p)[:, None]
        i_f = nl.arange(f)[None, :] * nbytes + i
        out_psum[:, :] = nisa.nc_transpose(
            src_reinterpreted[i_p, i_f],
            engine=nisa.tensor_engine,
        )
        i_p = nl.arange(f)[:, None]
        i_f = nl.arange(p)[None, :]
        out_reinterpreted[i_p, i_f * nbytes + i] = nl.copy(
            out_psum[i_p, i_f],
            dtype=nl.uint8,
        )


def get_move_dtype(src):
    itemsize = src.itemsize
    assert itemsize <= 4, f"{src.dtype=} has itemsize > 4"
    if itemsize == 1:
        return (
            nl.uint8
            if nisa.get_nc_version() == nisa.nc_version.gen2
            else nl.float8_e5m2
        )
    elif itemsize == 2:
        return nl.bfloat16
    else:
        return nl.float32


def create_identity_for_transpose(src_dtype, *sizes, force=False):
    if not force and src_dtype != nl.float32:
        return tuple([None for _ in sizes])
    identities = []
    for size in sizes:
        assert size > 0
        if size == 1:
            identity = nl.ones((1, 1), dtype=src_dtype)
        else:
            identity_hbm = nl.shared_constant(
                np.identity(n=size, dtype=np.uint8),
                dtype=src_dtype,
            )
            identity = nl.load(identity_hbm)
        identities.append(identity)
    return tuple(identities)


class IdentityStore:
    def __init__(self, *dtype_size):
        if nisa.get_nc_version() == nisa.nc_version.gen2:
            # XXX: work around an accuracy issue on Trn1
            self._manual_transpose_dtypes = [nl.float32]
        else:
            self._manual_transpose_dtypes = []
        self.cache = defaultdict(dict)
        for dtype, size in dtype_size:
            if dtype in self._manual_transpose_dtypes:
                if not size in self.cache[dtype]:
                    (identity,) = create_identity_for_transpose(dtype, size)
                    self.cache[dtype][size] = identity

    def force_add(self, dtype, size):
        if not size in self.cache[dtype]:
            (identity,) = create_identity_for_transpose(dtype, size, force=True)
            self.cache[dtype][size] = identity

    def print(self):
        for dtype in self.cache:
            for size in self.cache[dtype]:
                identity = self.cache[dtype][size]
                print(dtype, size, identity.dtype, identity.shape)

    def get(self, *dtype_size):
        out = []
        for dtype, size in dtype_size:
            if not dtype in self.cache:
                identity = None
            else:
                identity = self.cache[dtype].get(size, None)
            assert (
                identity is not None
                or dtype not in self._manual_transpose_dtypes
            ), f"Missing identity for {dtype} {size}"
            out.append(identity)
        return tuple(out)


def PF_transpose_with_PE(
    src,
    out,
    identity_for_transpose=None,
    out_in_psum=False,
):
    """
    Perform P-F Transpose with PE.
    """
    p, f = src.shape
    assert p <= B_P_SIZE and f <= B_P_SIZE
    assert out.shape == (f, p), f"{src.shape=} {out.shape=}"
    is_nc_gen2 = nisa.get_nc_version() == nisa.nc_version.gen2
    if out_in_psum:
        if is_nc_gen2 and src.dtype == nl.float32:
            # XXX: work around an accuracy issue on Trn1
            # When src and out dtype is float32, using nc_transpose
            # leads to result mismatch.
            # Not sure why nc_matmul does not have this issue on Trn1.
            assert out.dtype == nl.float32
            assert (
                identity_for_transpose is not None
                and identity_for_transpose.dtype == nl.float32
            )
            out[...] = nisa.nc_matmul(
                src,
                identity_for_transpose,
                is_moving_onezero=True,
                is_transpose=True,
            )
        else:
            psum_dtype = nl.float32 if is_nc_gen2 else src.dtype
            assert psum_dtype == out.dtype
            out[...] = nisa.nc_transpose(src, engine=nisa.tensor_engine)
    else:
        if src.dtype in (nl.int32, nl.uint32, nl.uint8):
            assert src.dtype == out.dtype
            if is_nc_gen2:
                PF_transpose_with_PE_integer(src, out)
            else:
                move_dtype = get_move_dtype(src)
                src_reinterpreted = src.view(move_dtype)
                out_reinterpreted = out.view(move_dtype)
                out_psum = nl.ndarray(
                    out.shape,
                    dtype=move_dtype,
                    buffer=nl.psum,
                )
                out_psum[...] = nisa.nc_transpose(
                    src_reinterpreted,
                    engine=nisa.tensor_engine,
                )
                out_reinterpreted[...] = nl.copy(out_psum)
        elif src.dtype == out.dtype == nl.float32:
            out_psum = nl.ndarray(out.shape, dtype=nl.float32, buffer=nl.psum)
            if is_nc_gen2:
                # XXX: work around an accuracy issue on Trn1
                # When src and out dtype is float32, using nc_transpose
                # leads to result mismatch.
                # Not sure why nc_matmul does not have this issue on Trn1.
                assert (
                    identity_for_transpose is not None
                    and identity_for_transpose.dtype == nl.float32
                )
                out_psum[...] = nisa.nc_matmul(
                    src,
                    identity_for_transpose,
                    is_moving_onezero=True,
                    is_transpose=True,
                )
            else:
                out_psum[...] = nisa.nc_transpose(
                    src,
                    engine=nisa.tensor_engine,
                )
            out[...] = nl.copy(out_psum, dtype=out.dtype)
        else:
            assert src.dtype in (nl.bfloat16, nl.float16, nl.float32), src.dtype
            if is_nc_gen2:
                out_psum = nl.ndarray(
                    out.shape,
                    dtype=nl.float32,
                    buffer=nl.psum,
                )
            else:
                out_psum = nl.ndarray(
                    out.shape,
                    dtype=src.dtype,
                    buffer=nl.psum,
                )
            out_psum[...] = nisa.nc_transpose(src, engine=nisa.tensor_engine)
            out[...] = nl.copy(out_psum, dtype=out.dtype)


def broadcast_partition_with_PE(
    src,
    out,
    src_one_zero=False,
    out_in_psum=False,
):
    """
    Perform Partition Dimension Broadcast with PE rather than vector engine.
    """
    assert (
        src.dtype != nl.int32
    ), f"{src.dtype=} may produce wrong results if input has negative values"
    assert (
        src.dtype not in (nl.uint32, nl.uint8)
        or nisa.get_nc_version() == nisa.nc_version.gen2
    )
    out_shape = out.shape
    assert len(src.shape) == 2 and len(out_shape) == 2
    assert src.shape[0] == 1 and src.shape[1] == out_shape[1]
    move_dtype = get_move_dtype(src)

    src_reinterpreted = src.view(move_dtype)
    ones = nl.ones((1, out_shape[0]), dtype=move_dtype)
    if out_in_psum:
        out_psum = out
    else:
        psum_dtype = nl.int32 if move_dtype == nl.uint8 else nl.float32
        out_psum = nl.ndarray(out_shape, dtype=psum_dtype, buffer=nl.psum)
    out_psum[:, :] = nisa.nc_matmul(
        ones,
        src_reinterpreted,
        is_stationary_onezero=True,
        is_moving_onezero=src_one_zero,
    )
    if out_in_psum:
        assert out.dtype == nl.float32
    elif src.dtype == move_dtype:
        out[...] = nl.copy(out_psum, dtype=out.dtype)
    else:
        if src.dtype == out.dtype:
            out_reinterpreted = out.view(move_dtype)
            out_reinterpreted[...] = nl.copy(out_psum, dtype=move_dtype)
        else:
            out_tmp = nl.ndarray(out.shape, dtype=src.dtype)
            out_tmp_reinterpreted = out_tmp.view(move_dtype)
            out_tmp_reinterpreted[...] = nl.copy(out_psum, dtype=move_dtype)
            out[...] = nl.copy(out_tmp, dtype=out.dtype)
