import math
import nki
import nki.isa as nisa
import nki.language as nl


@nki.jit
def nki_rmsnorm_kernel_isa(a, g, deterministic=True):
    out_tensor = nl.ndarray(a.shape, dtype=a.dtype, buffer=nl.shared_hbm)

    num_rows, hidden_dim = a.shape[0], a.shape[1]
    BATCH_TILE = 128
    HIDDEN_TILE = 128 if deterministic else 64

    g = g.reshape((1, hidden_dim))

    ones_vec = nl.ndarray((1, BATCH_TILE), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=ones_vec, value=1.0)

    zero_bias = nl.ndarray((BATCH_TILE, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=zero_bias, value=0.0)

    for i in nl.affine_range(math.ceil(num_rows / BATCH_TILE)):
        b_start = i * BATCH_TILE
        b_end = min(num_rows, b_start + BATCH_TILE)
        b_size = b_end - b_start

        sum_sq = nl.ndarray((BATCH_TILE, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=sum_sq, value=0.0)

        # Pass 1: Compute sum of squares
        for h in nl.affine_range(math.ceil(hidden_dim / HIDDEN_TILE)):
            h_start = h * HIDDEN_TILE
            h_end = min(hidden_dim, h_start + HIDDEN_TILE)
            h_size = h_end - h_start

            x = nl.ndarray((BATCH_TILE, HIDDEN_TILE), dtype=a.dtype, buffer=nl.sbuf)
            nisa.dma_copy(
                dst=x[0:b_size, 0:h_size], src=a[b_start:b_end, h_start:h_end]
            )

            x_sq = nl.ndarray(
                (BATCH_TILE, HIDDEN_TILE), dtype=nl.float32, buffer=nl.sbuf
            )
            tile_sum = nl.ndarray((BATCH_TILE, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation_reduce(
                dst=x_sq,
                op=nl.square,
                data=x,
                reduce_op=nl.add,
                reduce_res=tile_sum,
                bias=zero_bias,
                scale=1.0,
            )

            nisa.tensor_tensor(dst=sum_sq, data1=sum_sq, data2=tile_sum, op=nl.add)

        rms_inv = nl.ndarray((BATCH_TILE, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(
            dst=rms_inv,
            op=nl.rsqrt,
            data=sum_sq,
            scale=1.0 / hidden_dim,
            bias=zero_bias,
        )

        # Pass 2: Normalize and apply weight
        for h in nl.affine_range(math.ceil(hidden_dim / HIDDEN_TILE)):
            h_start = h * HIDDEN_TILE
            h_end = min(hidden_dim, h_start + HIDDEN_TILE)
            h_size = h_end - h_start

            x = nl.ndarray((BATCH_TILE, HIDDEN_TILE), dtype=a.dtype, buffer=nl.sbuf)
            nisa.dma_copy(
                dst=x[0:b_size, 0:h_size], src=a[b_start:b_end, h_start:h_end]
            )

            g_tile = nl.ndarray((1, HIDDEN_TILE), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=g_tile[0:1, 0:h_size], src=g[0:1, h_start:h_end])

            g_bcast = nl.ndarray(
                (BATCH_TILE, HIDDEN_TILE), dtype=nl.float32, buffer=nl.psum
            )
            nisa.nc_matmul(dst=g_bcast, stationary=ones_vec, moving=g_tile)

            x_out = nl.ndarray((BATCH_TILE, HIDDEN_TILE), dtype=a.dtype, buffer=nl.sbuf)
            nisa.scalar_tensor_tensor(
                dst=x_out,
                data=x,
                op0=nl.multiply,
                operand0=rms_inv,
                op1=nl.multiply,
                operand1=g_bcast,
            )

            nisa.dma_copy(
                dst=out_tensor[b_start:b_end, h_start:h_end],
                src=x_out[0:b_size, 0:h_size],
            )

    return out_tensor
