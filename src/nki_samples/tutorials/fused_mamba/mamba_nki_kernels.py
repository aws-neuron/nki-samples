"""
Copyright (C) 2026, Amazon.com. All Rights Reserved

Mamba-v1 NKI kernel implementation.

"""
# NKI_EXAMPLE_25_BEGIN
import nki
import nki.language as nl
import nki.isa as nisa
import numpy as np
# NKI_EXAMPLE_25_END
import argparse
import itertools

# NKI_EXAMPLE_25_BEGIN
@nki.jit
def mamba_v1(delta, u, A, B, C):
    """Computes the SSM operation in the Mamba model.

    :param delta: (batch_size, channels, seq_len)
    :param u: (batch_size, channels, seq_len)
    :param A: (channels, state_size)
    :param B: (batch_size, state_size, seq_len)
    :param C: (batch_size, state_size, seq_len)
    :return: (batch_size, channels, seq_len)
    """
    batch_size, channels, seq_len = delta.shape
    output = nl.ndarray((batch_size, channels, seq_len), dtype=delta.dtype,
                        buffer=nl.shared_hbm)

    _, state_size = A.shape

    # We can relax this using mask paramters in all the NKI API calls
    assert channels % 128 == 0

    # Map channels to the partition dimension
    # Tile channels to comply with NKI tile size constraints
    channel_psize = nl.tile_size.pmax
    n_channel_tile = channels // channel_psize

    # Most outer loop with batch_size, parallel_for
    for i_batch in nl.affine_range(batch_size):
        # Inner loop: tiling channels
        for i_channel_tile in nl.affine_range(n_channel_tile):
            channel_start = i_channel_tile * channel_psize

            # partial accumulated scanC result with processed states
            scanC_accum = nl.zeros((channel_psize, seq_len), dtype=delta.dtype)

            # Second outer loop with state_size, partial parallel
            for i_state in nl.affine_range(state_size):

                # Load the relevant tile from delta and A
                delta_slice = delta[i_batch, channel_start:channel_start+channel_psize, 0:seq_len]
                delta_i = nl.ndarray(delta_slice.shape, dtype=delta_slice.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=delta_i, src=delta_slice)
                A_slice = A[channel_start:channel_start+channel_psize, i_state:i_state+1]
                A_i = nl.ndarray(A_slice.shape, dtype=A_slice.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=A_i, src=A_slice)

                # Step 1&2: Element-wise multiplication of delta_i and A_i and then exponential
                deltaA = nl.ndarray((channel_psize, seq_len), dtype=delta.dtype, buffer=nl.sbuf)
                nisa.activation(dst=deltaA, op=nl.exp, data=delta_i, scale=A_i)

                # Load the relevant tile from u and B
                u_slice = u[i_batch, channel_start:channel_start+channel_psize, 0:seq_len]
                u_i = nl.ndarray(u_slice.shape, dtype=u_slice.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=u_i, src=u_slice)
                B_slice = B[i_batch, i_state:i_state+1, 0:seq_len]
                B_i = nl.ndarray(B_slice.shape, dtype=B_slice.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=B_i, src=B_slice)

                # Step 3: Element-wise multiplication of delta_i, B_i and u_i
                deltaU = nl.ndarray((channel_psize, seq_len), dtype=delta.dtype, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=deltaU, data1=delta_i, data2=u_i, op=nl.multiply)
                B_i_bcast = nl.broadcast_to(B_i, (channel_psize, seq_len))
                deltaBu = nl.ndarray((channel_psize, seq_len), dtype=delta.dtype, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=deltaBu, data1=deltaU, data2=B_i_bcast, op=nl.multiply)

                # Step 4: Associative scan between deltaA and deltaBu
                scan_res = nl.ndarray((channel_psize, seq_len), dtype=delta.dtype, buffer=nl.sbuf)
                nisa.tensor_tensor_scan(dst=scan_res, data0=deltaA, data1=deltaBu, initial=0.0,
                        op0=nl.multiply, op1=nl.add)

                # Load the relevant tile from C
                C_slice = C[i_batch, i_state:i_state+1, 0:seq_len]
                C_i = nl.ndarray(C_slice.shape, dtype=C_slice.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=C_i, src=C_slice)

                # Step 5: Element-wise multiplication of scan_res and C_i
                C_i_bcast = nl.broadcast_to(C_i, (channel_psize, seq_len))
                scanC = nl.ndarray((channel_psize, seq_len), dtype=delta.dtype, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=scanC, data1=scan_res, data2=C_i_bcast, op=nl.multiply)

                # Step 6: Accumulation of scanC along state_size dimension
                nisa.tensor_tensor(dst=scanC_accum, data1=scanC_accum, data2=scanC, op=nl.add)

            # Store scanC_accum for a single batch/channel tile to output
            nisa.dma_copy(dst=output[i_batch, channel_start:channel_start+channel_psize, 0:seq_len],
                    src=scanC_accum)

    return output
# NKI_EXAMPLE_25_END

# NKI_EXAMPLE_26_BEGIN
@nki.jit
def mamba_v2(delta, u, A, B, C):
    """Computes the SSM operation in the Mamba model.

    :param delta: (batch_size, channels, seq_len)
    :param u: (batch_size, channels, seq_len)
    :param A: (channels, state_size)
    :param B: (batch_size, state_size, seq_len)
    :param C: (batch_size, state_size, seq_len)
    :return: (batch_size, channels, seq_len)
    """
    batch_size, channels, seq_len = delta.shape
    output = nl.ndarray((batch_size, channels, seq_len), dtype=delta.dtype,
                        buffer=nl.shared_hbm)
    _, state_size = A.shape

    assert channels % 128 == 0

    # Map channels to the partition dimension
    # Tile channels to comply with NKI tile size constraints
    channel_psize = nl.tile_size.pmax
    n_channel_tile = channels // channel_psize

    # Most outer loop with batch_size, parallel_for
    for i_batch in nl.affine_range(batch_size):

        # Second outer loop: tiling channels
        for i_channel_tile in nl.affine_range(n_channel_tile):
            channel_start = i_channel_tile * channel_psize

            # partial accumulated scanC result with processed states
            scanC_accum = nl.zeros((channel_psize, seq_len), dtype=delta.dtype)

            # Load delta/u once to be reused across states
            delta_slice = delta[i_batch, channel_start:channel_start+channel_psize, 0:seq_len]
            delta_i = nl.ndarray(delta_slice.shape, dtype=delta_slice.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=delta_i, src=delta_slice)
            u_slice = u[i_batch, channel_start:channel_start+channel_psize, 0:seq_len]
            u_i = nl.ndarray(u_slice.shape, dtype=u_slice.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=u_i, src=u_slice)

            # Inner loop with state_size, partial parallel
            for i_state in nl.affine_range(state_size):
                # Load the relevant tile from A
                A_slice = A[channel_start:channel_start+channel_psize, i_state:i_state+1]
                A_i = nl.ndarray(A_slice.shape, dtype=A_slice.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=A_i, src=A_slice)

                # Step 1&2: Element-wise multiplication of delta_i and A_i and then exponential
                deltaA = nl.ndarray((channel_psize, seq_len), dtype=delta.dtype, buffer=nl.sbuf)
                nisa.activation(dst=deltaA, op=nl.exp, data=delta_i, scale=A_i)

                # Load the relevant tile from B
                B_slice = B[i_batch, i_state:i_state+1, 0:seq_len]
                B_i = nl.ndarray(B_slice.shape, dtype=B_slice.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=B_i, src=B_slice)

                # Step 3: Element-wise multiplication of delta_i, B_i and u_i
                deltaU = nl.ndarray((channel_psize, seq_len), dtype=delta.dtype, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=deltaU, data1=delta_i, data2=u_i, op=nl.multiply)
                B_i_bcast = nl.broadcast_to(B_i, (channel_psize, seq_len))
                deltaBu = nl.ndarray((channel_psize, seq_len), dtype=delta.dtype, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=deltaBu, data1=deltaU, data2=B_i_bcast, op=nl.multiply)

                # Step 4: Associative scan between deltaA and deltaBu
                scan_res = nl.ndarray((channel_psize, seq_len), dtype=delta.dtype, buffer=nl.sbuf)
                nisa.tensor_tensor_scan(dst=scan_res, data0=deltaA, data1=deltaBu, initial=0.0,
                        op0=nl.multiply, op1=nl.add)

                # Load the relevant tile from C
                C_slice = C[i_batch, i_state:i_state+1, 0:seq_len]
                C_i = nl.ndarray(C_slice.shape, dtype=C_slice.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=C_i, src=C_slice)

                # Step 5: Element-wise multiplication of scan_res and C_i
                C_i_bcast = nl.broadcast_to(C_i, (channel_psize, seq_len))
                scanC = nl.ndarray((channel_psize, seq_len), dtype=delta.dtype, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=scanC, data1=scan_res, data2=C_i_bcast, op=nl.multiply)

                # Step 6: Accumulation of scanC along state_size dimension
                nisa.tensor_tensor(dst=scanC_accum, data1=scanC_accum, data2=scanC, op=nl.add)

            # Store scanC_accum for a single batch to output
            nisa.dma_copy(dst=output[i_batch, channel_start:channel_start+channel_psize, 0:seq_len],
                    src=scanC_accum[0:channel_psize, 0:seq_len])

    return output
# NKI_EXAMPLE_26_END


@nki.jit
def mamba_v3(delta, u, A, B, C):
    """Computes the SSM operation in the Mamba model.

    :param delta: (batch_size, channels, seq_len)
    :param u: (batch_size, channels, seq_len)
    :param A: (channels, state_size)
    :param B: (batch_size, state_size, seq_len)
    :param C: (batch_size, state_size, seq_len)
    :return: (batch_size, channels, seq_len)
    """
    batch_size, channels, seq_len = delta.shape
    output = nl.ndarray((batch_size, channels, seq_len), dtype=delta.dtype,
                        buffer=nl.shared_hbm)
    _, state_size = A.shape

    # Map channels to the partition dimension
    # Tile channels to comply with NKI tile size constraints
    channel_psize = nl.tile_size.pmax
    n_channel_tile = channels // channel_psize

    # Magic number, decided through empirical profiling data
    seq_len_fsize = 512
    n_seq_len_tile = seq_len // seq_len_fsize

    # Fix this later with mask
    assert channels % channel_psize == 0
    assert seq_len % seq_len_fsize == 0

    # Most outer loop with batch_size, parallel_for
    for i_batch in nl.affine_range(batch_size):

        # Second outer loop: tiling channels
        for i_channel_tile in nl.affine_range(n_channel_tile):
            channel_start = i_channel_tile * channel_psize

            # partial accumulated scanC result with processed states
            scanC_accum = nl.zeros((channel_psize, seq_len), dtype=delta.dtype)

            # Load delta/u once to be reused across states
            delta_slice = delta[i_batch, channel_start:channel_start+channel_psize, 0:seq_len]
            delta_i = nl.ndarray(delta_slice.shape, dtype=delta_slice.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=delta_i, src=delta_slice)
            u_slice = u[i_batch, channel_start:channel_start+channel_psize, 0:seq_len]
            u_i = nl.ndarray(u_slice.shape, dtype=u_slice.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=u_i, src=u_slice)

            # Inner loop with state_size, partial parallel
            for i_state in nl.affine_range(state_size):
                # Load the relevant tile from A
                A_slice = A[channel_start:channel_start+channel_psize, i_state:i_state+1]
                A_i = nl.ndarray(A_slice.shape, dtype=A_slice.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=A_i, src=A_slice)

                # Last scan result
                scan_init = nl.zeros((channel_psize, 1), dtype=delta_i.dtype)
                # FIXME: sequential_range gives incorrect answer and also much worse perf than static_range
                # for i_seq_len_tile in nl.sequential_range(n_seq_len_tile):
                for i_seq_len_tile in nl.static_range(n_seq_len_tile):
                    seq_len_start = i_seq_len_tile * seq_len_fsize

                    # Step 1&2: Element-wise multiplication of delta_i and A_i and then exponential
                    deltaA = nl.ndarray((channel_psize, seq_len_fsize), dtype=delta.dtype, buffer=nl.sbuf)
                    nisa.activation(dst=deltaA, op=nl.exp,
                            data=delta_i[0:channel_psize, seq_len_start:seq_len_start+seq_len_fsize],
                            scale=A_i)

                    # Load the relevant tile from B
                    B_slice = B[i_batch, i_state:i_state+1, seq_len_start:seq_len_start+seq_len_fsize]
                    B_i = nl.ndarray(B_slice.shape, dtype=B_slice.dtype, buffer=nl.sbuf)
                    nisa.dma_copy(dst=B_i, src=B_slice)

                    # Step 3: Element-wise multiplication of delta_i, B_i and u_i
                    deltaU = nl.ndarray((channel_psize, seq_len_fsize), dtype=delta.dtype, buffer=nl.sbuf)
                    nisa.tensor_tensor(dst=deltaU,
                            data1=delta_i[0:channel_psize, seq_len_start:seq_len_start+seq_len_fsize],
                            data2=u_i[0:channel_psize, seq_len_start:seq_len_start+seq_len_fsize],
                            op=nl.multiply)
                    B_i_bcast = nl.broadcast_to(B_i, (channel_psize, seq_len_fsize))
                    deltaBu = nl.ndarray((channel_psize, seq_len_fsize), dtype=delta.dtype, buffer=nl.sbuf)
                    nisa.tensor_tensor(dst=deltaBu, data1=deltaU, data2=B_i_bcast, op=nl.multiply)

                    # Step 4: Associative scan between deltaA and deltaBu
                    scan_res = nl.ndarray((channel_psize, seq_len_fsize), dtype=delta.dtype, buffer=nl.sbuf)
                    nisa.tensor_tensor_scan(dst=scan_res, data0=deltaA, data1=deltaBu, initial=scan_init,
                            op0=nl.multiply, op1=nl.add)
                    nisa.tensor_copy(dst=scan_init, src=scan_res[0:channel_psize, seq_len_fsize-1:seq_len_fsize])

                    # Load the relevant tile from C
                    C_slice = C[i_batch, i_state:i_state+1, seq_len_start:seq_len_start+seq_len_fsize]
                    C_i = nl.ndarray(C_slice.shape, dtype=C_slice.dtype, buffer=nl.sbuf)
                    nisa.dma_copy(dst=C_i, src=C_slice)

                    # Step 5: Element-wise multiplication of scan_res and C_i
                    C_i_bcast = nl.broadcast_to(C_i, (channel_psize, seq_len_fsize))
                    scanC = nl.ndarray((channel_psize, seq_len_fsize), dtype=delta.dtype, buffer=nl.sbuf)
                    nisa.tensor_tensor(dst=scanC, data1=scan_res, data2=C_i_bcast, op=nl.multiply)

                    # Step 6: Accumulation of scanC along state_size dimension
                    nisa.tensor_tensor(dst=scanC_accum[0:channel_psize, seq_len_start:seq_len_start+seq_len_fsize],
                            data1=scanC_accum[0:channel_psize, seq_len_start:seq_len_start+seq_len_fsize],
                            data2=scanC, op=nl.add)

            # Store scanC_accum for a single batch to output
            nisa.dma_copy(dst=output[i_batch, channel_start:channel_start+channel_psize, 0:seq_len],
                    src=scanC_accum[0:channel_psize, 0:seq_len])
    return output


def parse_args():
    parser = argparse.ArgumentParser("Run Mamba NKI kernels.")
    parser.add_argument("--version",
            nargs='+',
            default=["v1", "v2", "v3"],
            choices=["v1", "v2", "v3"],
            help="Test versions")

    parser.add_argument("--batch",
            nargs='+',
            default=[1],
            help="Batch size.")
    parser.add_argument("--seq_len",
            nargs='+',
            default=[2048],
            help="Sequence length.")
    parser.add_argument("--channels",
            nargs='+',
            default=[256],
            help="Number of channels.")
    parser.add_argument("--state_size",
            nargs='+',
            default=[16],
            help="State size.")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    # Small test to ensure numerical correctness
    arr_batch = [int(_) for _ in args.batch]
    arr_seq_len = [int(_) for _ in args.seq_len]
    arr_channels = [int(_) for _ in args.channels]
    arr_state_size = [int(_) for _ in args.state_size]

    configs = itertools.product(arr_batch, arr_seq_len, arr_channels, arr_state_size)

    for config in configs:
        batch, seq_len, channels, state_size = config
        print(f">>> batch={batch}, seq_len={seq_len}, channels={channels}, state_size={state_size}")

        # Set up input tensors
        dtype = np.float32
        delta = np.ones((batch, channels, seq_len), dtype=dtype)
        u = np.ones((batch, channels, seq_len), dtype=dtype)
        A = -np.ones((channels, state_size), dtype=dtype)
        B = np.ones((batch, state_size, seq_len), dtype=dtype)
        C = np.ones((batch, state_size, seq_len), dtype=dtype)

        func_dict = {"v1": mamba_v1,
                     "v2": mamba_v2,
                     "v3": mamba_v3,
                    }

        # v1: reference kernel
        print(f">>>> Running v1 (reference).")
        nki_out_v1 = mamba_v1(delta, u, A, B, C)

        for version in args.version:
            if version == "v1":
                # already run, continue
                continue

            print(f">>>> Running version {version}.")
            func = func_dict[version]
            nki_out_test = func(delta, u, A, B, C)
            print(f">>>> mamba {version} matches?", np.all(nki_out_test == nki_out_v1))
            assert np.all(nki_out_test == nki_out_v1)