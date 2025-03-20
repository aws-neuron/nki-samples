"""
Copyright (c) 2025, Amazon.com. All Rights Reserved

Kernels written by the AWS Neuron.

WARNING: These kernels:
   - Are tested only against internal nightly builds
   - May not be compatible with public NeuronSDK releases
   - Have not been extensively tested across all input configurations
   - Carry no compatibility guarantees
   - The behavior of these kernels may be modified without prior notice

"""

import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa
from neuronxcc.nki.nccl.collectives import collective_permute_implicit
from neuronxcc.nki._private.private_api import cc_div
from neuronxcc.nki.isa.neuron_isa import dma_copy

def generate_replica_groups(tp_degree, num_groups):
    num_replicas = tp_degree * num_groups
    replicas_per_group = num_replicas // num_groups
 
    replica_groups = [
        list(range(i * replicas_per_group, (i + 1) * replicas_per_group))
        for i in range(num_groups)
    ]
 
    return replica_groups
 
def collective_matmul(
        lhs,
        rhs,
        input_token,
        result,
        output_token,
        rhs_contracting_dim=0,
        tp_degree=8,
        use_sb_to_sb=False,
        arch_target=nisa.nc_version.gen2,
        num_groups=1,
        has_token=False
    ):
      """NKI kernel to compute a collective matmul dot(all-gather(x), y) pattern.
         Kernel is double buffered and supports multi channel cc permute.
         Logical Neuron Core (LNC) is supported.
      Args:
          lhs: An input tensor of shape [M,K] or [Batch,M,K].
          rhs: An input tensor of shape [K,N] or a FusedQKV Weight of shape [3,K,Heads,Hidden_Per_Head].
          input_token: An input scalar used for scheduling of collectives.
          result: The resulting output tensor of shape [M,N] or [Batch,M,N] or a FusedQKV output of [OptionalBatch,3,M,Heads,Hidden_Per_Head].
          output_token: The output token memcopy of the input token. Used by next collectives.
          rhs_contracting_dim: The dimension on RHS that is contracting with the innermost dimension of LHS.
          tp_degree: The number of tp workers running in this kernel.
          use_sb_to_sb: Whether to use state buffer to state buffer cc permutes.
          arch_target: Whether this kernel is running on Gen2 or Gen3 hardware.
          num_groups: Number of replica groups.
          has_token: Whether this kernel has token input and output.
      """
      #--------------------------------- Setup Start ---------------------------------------
      # This is hard-coded and may not be right in every situation
      NUM_LNC = 1 if arch_target == nisa.nc_version.gen2 else 2
 
      if NUM_LNC > 1:
        # if in LNC mode - always use SB to SB
        use_sb_to_sb = True
        assert NUM_LNC == 2, "Only LNC=2 is supported in CollectiveMatmul Kernel."
      
      original_result_shape = result.shape
      if len(lhs.shape) == 3:
        # Support batch dimension. 
        # Just reshape batch into sequence.
        batch, seq, hidden = lhs.shape
        lhs = lhs.reshape((batch*seq, hidden))
      
      M, K = lhs.shape
      original_fused_qkv_shape = None
      original_rhs_shape = None
 
      #---------Special Case Embedding Table Handling------------------
      transpose_rhs = False
      if len(rhs.shape)==2 and rhs_contracting_dim==1:
        # Perform real transpose when loading into SBUF
        transpose_rhs = True
        free, hidden = rhs.shape
        original_rhs_shape = rhs.shape
        rhs = rhs.reshape((hidden, free))
 
      #--------------Special Case FusedQKV Handling---------------
      if len(rhs.shape)==4:
        # Only support something like (3,4096,4,128) for now.        
        three, hidden, heads, h_per_head = rhs.shape
        original_fused_qkv_shape = rhs.shape
        rhs = rhs.reshape((hidden, three*heads*h_per_head))
      
      #--------------Special case QKV non fused---------------------
      if len(rhs.shape)==3:
        # Only support something like (8192,2,128)
        hidden, heads, h_per_head = rhs.shape
        rhs = rhs.reshape((hidden, heads*h_per_head))
 
      K_, N = rhs.shape
      assert K == K_, "lhs and rhs must have the same contraction dimension"
      # We know the result shape must be (M * tp_degree, Free dimension of RHS).
      # We do not care if it has heads, batch etc. We reshape it to what HLO wanted later.
      result = result.reshape((M*tp_degree, N))
 
      #---------------------------------- LNC Setup ----------------------------------------
      if NUM_LNC > 1:
        # Shard Sequence over LNC.
        M //= NUM_LNC
        # We do not shard weight as we do not want local colletive.
      #---------------------------------- LNC Setup END ----------------------------------------
 
      TILE_M = min(nl.tile_size.gemm_stationary_fmax, M)  # 128
      TILE_K = nl.tile_size.pmax  # 128
      TILE_N = nl.tile_size.gemm_moving_fmax  # 512
 
      # Meta-parameters
      N_REMAINDER=0
      if N%TILE_N != 0:
        N_REMAINDER=1
      TILES_IN_BLOCK_M=M//TILE_M
      TILES_IN_BLOCK_N=(N//TILE_N)+N_REMAINDER
      TILES_IN_BLOCK_K=K//TILE_K
 
      BLOCK_M = TILE_M * TILES_IN_BLOCK_M
 
      BLOCK_N = (TILE_N * (TILES_IN_BLOCK_N-N_REMAINDER)) + N%TILE_N
      BLOCK_K = TILE_K * TILES_IN_BLOCK_K
 
      assert M == BLOCK_M, "This LHS shape is not supported."
      assert K % BLOCK_K == 0
      # Only whole box can use 4 channels.
      # TP=32 on Trn1 or TP=128 on Trn2.
      TP_DEGREE_TO_CHANNELS = {8:2, 16:2, 32:2 if arch_target == nisa.nc_version.gen3 else 4, 64:2, 128:4}
      TP_DEGREE_TO_REPLICA_GROUP = {
          degree: generate_replica_groups(tp_degree=tp_degree, num_groups=num_groups)
          for degree in [8, 16, 32, 64, 128]
      }
 
      NUM_RANKS = tp_degree
      NUM_BLOCK_M = M // BLOCK_M
      NUM_BLOCK_N = (N // BLOCK_N) + N_REMAINDER
      NUM_BLOCK_K = K // BLOCK_K
      NUM_CHANNELS = TP_DEGREE_TO_CHANNELS[tp_degree]
      # Load the full LHS and RHS into SBUF, tiling to respect the partition dimension limit
      if use_sb_to_sb:
        # in SB to SB the layout is (128, NUM_CHANNELS, hidden/128, seq//NUM_CHANNELS)
        lhsT_sbuf = nl.ndarray((nl.par_dim(TILE_K), NUM_CHANNELS, K//TILE_K, M//NUM_CHANNELS), dtype=lhs.dtype, buffer=nl.sbuf)
      else:
        lhsT_sbuf = nl.ndarray((nl.par_dim(TILE_K), K//TILE_K, M), dtype=lhs.dtype, buffer=nl.sbuf)
 
      rhs_sbuf = nl.ndarray((nl.par_dim(TILE_K), K//TILE_K, N), dtype=rhs.dtype, buffer=nl.sbuf)
      
 
      if NUM_LNC > 1:
        # Result is (NUM_LNCS, NUM_RANKS, NUM_CHANNELS, SEQ_LENGTH//NUM_CHANNELS//NUM_LNCS, HIDDEN)
        result = result.reshape((NUM_LNC, NUM_RANKS, NUM_CHANNELS, M//NUM_CHANNELS, N))
      else:
        # Result is (NUM_RANKS, NUM_CHANNELS, SEQ_LENGTH//NUM_CHANNELS, HIDDEN)
        result = result.reshape((NUM_RANKS, NUM_CHANNELS, M//NUM_CHANNELS, N))
      
      def load_into_rhs_sbuf(rhs, rhs_sbuf):
        if original_fused_qkv_shape is None:
            # Load RHS into SBUF
            # No transpose needed if not embedding table edge case
            # For DMA Transpose - src dim=1 must be 128 and dst dim=0 must be 128
            if transpose_rhs:
              rhs = rhs.reshape(original_rhs_shape)
              free, hidden = rhs.shape
              for h in nl.affine_range(hidden // TILE_K):
                free_tile_src = nl.arange(free)[:, None]
                free_tile_dst = nl.arange(free)[None, :]
                i_rhsT_load_y_rhs = nl.arange(TILE_K)[None, :]
                i_rhsT_load_p_lhs = nl.arange(TILE_K)[:, None]
                i_rhsT_load_y_lhs = nl.arange(min(TILE_K, free))[None, :]
                rhs_sbuf[i_rhsT_load_p_lhs, h, free_tile_dst] = nisa.dma_transpose(rhs[free_tile_src, h*TILE_K + i_rhsT_load_y_rhs])
            else:
              for k in nl.affine_range(K // TILE_K):
                  i_rhs_load_p = nl.arange(TILE_K)[:, None]
                  i_rhs_load_y = nl.arange(N)[None, :]
                  rhs_sbuf[i_rhs_load_p, k, i_rhs_load_y] = nl.load(rhs[k*TILE_K + i_rhs_load_p, i_rhs_load_y])
        else:
            # We must transpose the 3, and hidden
            rhs = rhs.reshape(original_fused_qkv_shape)
            three, hidden, heads, h_per_head = rhs.shape
            original_rhs_sbuf_shape = rhs_sbuf.shape
            # 3, 8192, 2, 128
            rhs = rhs.reshape((three, hidden, heads*h_per_head))
            # Temp buffer on SBUF that we will load into and then copy into the real RHS SBUF
            temp_rhs_sbuf = nl.ndarray((three, nl.par_dim(TILE_K), TILES_IN_BLOCK_K, heads*h_per_head), dtype=rhs.dtype, buffer=nl.sbuf)
            rhs_sbuf = rhs_sbuf.reshape((TILE_K, TILES_IN_BLOCK_K, three, heads*h_per_head))
            for k in nl.affine_range(TILES_IN_BLOCK_K):
              for t in nl.affine_range(three):
                i_rhs_load_p = nl.arange(TILE_K)[:, None]
                load_y = nl.arange(heads*h_per_head)[None, :]
                temp_rhs_sbuf[t, i_rhs_load_p, k, load_y] = nl.load(rhs[t, k*TILE_K+i_rhs_load_p, load_y])
                # The copy itself performs the transpose.
                rhs_sbuf[i_rhs_load_p, k, t, load_y] = nl.copy(temp_rhs_sbuf[t, i_rhs_load_p, k, load_y])
            
            rhs_sbuf = rhs_sbuf.reshape(original_rhs_sbuf_shape)
            
      # Load weight into SBUF
      load_into_rhs_sbuf(rhs=rhs, rhs_sbuf=rhs_sbuf)
 
      if not use_sb_to_sb:
        # We have to do a DMA copy if using HBM cc permute because read and write to LHS input buffer.
        next_lhsT = nl.ndarray(lhs.shape, dtype=lhs.dtype, buffer=nl.hbm)
        lhs_hbm = nl.ndarray(lhs.shape, dtype=lhs.dtype, buffer=nl.hbm)
        dma_copy(dst=lhs_hbm, src=lhs)
        next_lhsT_hbm = next_lhsT.reshape((NUM_CHANNELS, M // NUM_CHANNELS, K))
        lhs_hbm = lhs_hbm.reshape((NUM_CHANNELS, M // NUM_CHANNELS, K))
      else:
        # If using SB to SB our next LHS will be on SBUF.
        next_lhsT_sbuf = nl.ndarray(lhsT_sbuf.shape, dtype=lhs.dtype, buffer=nl.sbuf)
 
      #--------------------------------- Setup END ---------------------------------------
 
      #***********************************************************************************
      #--------------------------------- Helper Functions --------------------------------
      #***********************************************************************************
 
      def load_into_lhs_sbuf_hbm_to_hbm(hbm_buffer):
        for k in nl.affine_range(K // TILE_K):
          i_lhsT_load_p_rhs = nl.arange(TILE_K)[None, :]
          seq_tile_src = nl.arange(M)[:, None]
          seq_tile_dst = nl.arange(M)[None, :]
          i_lhsT_load_y_lhs = nl.arange(TILE_K)[:, None]
          lnc_id = 0
          if NUM_LNC > 1:
            # LNC 1 adds sequence length as an offset
            lnc_id = nl.program_id(0)
          lhsT_sbuf[i_lhsT_load_y_lhs, k, seq_tile_dst] = nisa.dma_transpose(hbm_buffer[M*lnc_id + seq_tile_src, k*TILE_K + i_lhsT_load_p_rhs])
          
      def load_into_lhs_sbuf_use_sb_to_sb(hbm_buffer):
        # SB to SB has a channel axis 
        for channel in nl.affine_range(NUM_CHANNELS):
          for k in nl.affine_range(K // TILE_K):
            seq_tile_src = nl.arange(M//NUM_CHANNELS)[:, None]
            seq_tile_dst = nl.arange(M//NUM_CHANNELS)[None, :]
            i_lhsT_load_p_rhs = nl.arange(TILE_K)[None, :]
            i_lhsT_load_y_lhs = nl.arange(TILE_K)[:, None]
            lnc_id = 0
            if NUM_LNC > 1:
              # LNC 1 adds sequence length as an offset
              lnc_id = nl.program_id(0)
            lhsT_sbuf[i_lhsT_load_y_lhs, channel, k, seq_tile_dst] = nisa.dma_transpose(hbm_buffer[channel*(M//NUM_CHANNELS) + M*lnc_id + seq_tile_src, k*TILE_K + i_lhsT_load_p_rhs])

      def load_into_lhs_sbuf(hbm_buffer):
        if not use_sb_to_sb:
          # This reshape is only applicable when using HBM to HBM permute.
          hbm_buffer = hbm_buffer.reshape((M, K))
          load_into_lhs_sbuf_hbm_to_hbm(hbm_buffer)
        else:
          load_into_lhs_sbuf_use_sb_to_sb(hbm_buffer)
 
      def run_matmul_hbm_to_hbm(lhs_buffer, n, result_tiles):
        # Blocking K dimension (the contraction dimension)
        for k in nl.sequential_range(NUM_BLOCK_K):
          # Loading tiles from rhs_sbuf
          i_rhs_tile = nl.mgrid[0:TILE_K, 0:BLOCK_N]
          rhs_tiles = nl.ndarray((TILES_IN_BLOCK_K, nl.par_dim(TILE_K), BLOCK_N),
                                dtype=rhs.dtype,
                                buffer=nl.sbuf)
          for bk in nl.affine_range(TILES_IN_BLOCK_K):
            rhs_tiles[bk, i_rhs_tile.p, i_rhs_tile.x] = nl.copy(rhs_sbuf[i_rhs_tile.p, k*TILES_IN_BLOCK_K + bk, BLOCK_N*n + i_rhs_tile.x], mask=(BLOCK_N*n + i_rhs_tile.x)<N)
 
          # Blocking M dimension (the LHS free dimension)
          for m in nl.affine_range(NUM_BLOCK_M):
            # Loading tiles from lhs_buffer
            i_lhsT_tile = nl.mgrid[0:TILE_K, 0:BLOCK_M]
            lhsT_tiles = nl.ndarray((TILES_IN_BLOCK_K, nl.par_dim(TILE_K), BLOCK_M),
                                    dtype=lhs.dtype,
                                    buffer=nl.sbuf)
            for bk in nl.affine_range(TILES_IN_BLOCK_K):
              lhsT_tiles[bk, i_lhsT_tile.p, i_lhsT_tile.x] = lhs_buffer[i_lhsT_tile.p, k*TILES_IN_BLOCK_K + bk, BLOCK_M*m + i_lhsT_tile.x]
 
            # Do matmul with all tiles in the blocks
            i_lhsT_mm = nl.mgrid[0:TILE_K, 0:TILE_M]
            i_rhs_mm = nl.mgrid[0:TILE_K, 0:TILE_N]
            i_res_mm = nl.mgrid[0:TILE_M, 0:TILE_N]  
            for bn in nl.affine_range(TILES_IN_BLOCK_N):
              for bm in nl.affine_range(TILES_IN_BLOCK_M):
                res_tile = nl.zeros((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum)
 
                for bk in nl.affine_range(TILES_IN_BLOCK_K):  
                  res_tile[...] += nisa.nc_matmul(
                      lhsT_tiles[bk, i_lhsT_mm.p, bm * TILE_M + i_lhsT_mm.x],
                      rhs_tiles[bk, i_rhs_mm.p, bn * TILE_N + i_rhs_mm.x], mask=(bn * TILE_N + i_rhs_mm.x)<N)
 
                # Accumulate on corresponding SBUF tile  
                result_tiles[m, bm, bn, i_res_mm.p, 
                            i_res_mm.x] += res_tile[i_res_mm.p, i_res_mm.x]
 
      def run_matmul_sb_to_sb(lhs_buffer, n, result_tiles):
        # The token tile per channel.
        TILE_SB_PER_CHANNEL = M//NUM_CHANNELS
        # Blocking K dimension (the contraction dimension)
        for k in nl.sequential_range(NUM_BLOCK_K):
          # Loading tiles from rhs_sbuf
          i_rhs_tile = nl.mgrid[0:TILE_K, 0:BLOCK_N]
          rhs_tiles = nl.ndarray((TILES_IN_BLOCK_K, nl.par_dim(TILE_K), BLOCK_N),
                                dtype=rhs.dtype,
                                buffer=nl.sbuf)
          for bk in nl.affine_range(TILES_IN_BLOCK_K):
            rhs_tiles[bk, i_rhs_tile.p, i_rhs_tile.x] = nl.copy(rhs_sbuf[i_rhs_tile.p, k*TILES_IN_BLOCK_K + bk, BLOCK_N*n + i_rhs_tile.x], mask=(BLOCK_N*n + i_rhs_tile.x)<N)

          # Blocking M dimension (the LHS free dimension)
          for m in nl.affine_range(NUM_BLOCK_M):
            # Loading tiles from lhs_buffer
            i_lhsT_tile = nl.mgrid[0:TILE_K, 0:TILE_SB_PER_CHANNEL]
            lhsT_tiles = nl.ndarray((TILES_IN_BLOCK_K, nl.par_dim(TILE_K), BLOCK_M),
                                    dtype=lhs.dtype,
                                    buffer=nl.sbuf)
            for bk in nl.affine_range(TILES_IN_BLOCK_K):
              for channel in nl.affine_range(NUM_CHANNELS):
                lhsT_tiles[bk, i_lhsT_tile.p, channel*TILE_SB_PER_CHANNEL + i_lhsT_tile.x] = lhs_buffer[i_lhsT_tile.p, channel, k*TILES_IN_BLOCK_K + bk, BLOCK_M*m + i_lhsT_tile.x]

            # Do matmul with all tiles in the blocks
            i_lhsT_mm = nl.mgrid[0:TILE_K, 0:TILE_M]
            i_rhs_mm = nl.mgrid[0:TILE_K, 0:TILE_N]
            i_res_mm = nl.mgrid[0:TILE_M, 0:TILE_N]  
            for bn in nl.affine_range(TILES_IN_BLOCK_N):
              for bm in nl.affine_range(TILES_IN_BLOCK_M):
                res_tile = nl.zeros((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum)

                for bk in nl.affine_range(TILES_IN_BLOCK_K):  
                  res_tile[...] += nisa.nc_matmul(
                      lhsT_tiles[bk, i_lhsT_mm.p, bm * TILE_M + i_lhsT_mm.x],
                      rhs_tiles[bk, i_rhs_mm.p, bn * TILE_N + i_rhs_mm.x], mask=(bn * TILE_N + i_rhs_mm.x)<N)

                # Accumulate on corresponding SBUF tile  
                result_tiles[m, bm, bn, i_res_mm.p, 
                            i_res_mm.x] += res_tile[i_res_mm.p, i_res_mm.x]
 
      def run_matmul(lhs_buffer, n, result_tiles):
        if use_sb_to_sb:
          run_matmul_sb_to_sb(lhs_buffer, n, result_tiles)
        else:
          run_matmul_hbm_to_hbm(lhs_buffer, n, result_tiles)
 
      def launch_collective_permutes(switch_buffers, mask=None):
        for channel in nl.affine_range(NUM_CHANNELS):
          if not use_sb_to_sb:
            buffer_one = next_lhsT_hbm[channel, :, :]
            buffer_two = lhs_hbm[channel, :, :]
          else:
            offset = (K//TILE_K*M)//NUM_CHANNELS
            buffer_one = next_lhsT_sbuf[:, channel*offset:(channel+1)*offset]
            buffer_two = lhsT_sbuf[:, channel*offset:(channel+1)*offset]
          
          # Launch the permute. Switching the buffers each iteration.
          collective_permute_implicit(dst=buffer_one if not switch_buffers else buffer_two, 
                                        src=buffer_two if not switch_buffers else buffer_one, 
                                        replica_groups=TP_DEGREE_TO_REPLICA_GROUP[tp_degree], 
                                        channel_id=channel, num_channels=NUM_CHANNELS, mask=mask)
 
 
      def write_to_output(rank_offset, result_tiles):
        # Copying the result from SBUF to HBM
          if TILES_IN_BLOCK_M//NUM_CHANNELS == 0:
            # EDGE Case: Sequence_Length//NUM_RANKS//NUM_CHANNELS < 128
            # We do not have to tile writes here. Each channel writes one time.
            for m in nl.affine_range(NUM_BLOCK_M):
              for bn in nl.affine_range(TILES_IN_BLOCK_N):
                for channel in nl.static_range(NUM_CHANNELS):
                  OUTPUT_WRITE = result.shape[-2]
                  min_bound = OUTPUT_WRITE*channel
                  max_bound = (OUTPUT_WRITE*channel)+OUTPUT_WRITE
                  bm_p = min_bound//TILE_M
                  min_bound = min_bound%TILE_M
                  max_bound = (max_bound%TILE_M)
                  if max_bound == 0:
                    max_bound = TILE_M
                  assert max_bound != 0 and max_bound <= TILE_M
                  i_res_x = nl.arange(TILE_N)[None, :]
                  i_p = nl.arange(min_bound, max_bound)[:, None]
                  result_seq = nl.arange(M//NUM_CHANNELS)[:, None]
                  i_f = BLOCK_N * n + bn * TILE_N + i_res_x
                  res = result_tiles[m, bm_p, bn, i_p, i_res_x]
                  numerator = (2*rank+rank_offset) * NUM_CHANNELS
                  numerator = numerator+channel
                  if NUM_LNC > 1:
                    nl.store(result[nl.program_id(0), 
                              cc_div(numerator, NUM_CHANNELS, TP_DEGREE_TO_REPLICA_GROUP[tp_degree]), 
                              channel,
                              result_seq, i_f],  
                              value=res, mask=(i_f)<N)
                  else:
                      nl.store(result[cc_div(numerator, NUM_CHANNELS, TP_DEGREE_TO_REPLICA_GROUP[tp_degree]), 
                              channel,
                              result_seq, i_f],  
                              value=res, mask=(i_f)<N)
          else:
            #Case: Sequence_Length//NUM_RANKS//NUM_CHANNELS >= 128 
            # so we have to tile the writes
            for channel in nl.affine_range(NUM_CHANNELS):
              for m in nl.affine_range(NUM_BLOCK_M): 
                for bn in nl.affine_range(TILES_IN_BLOCK_N):
                  for bm in nl.static_range(TILES_IN_BLOCK_M//NUM_CHANNELS):
                    i_res_p = nl.arange(TILE_M)[:, None]
                    i_res_x = nl.arange(TILE_N)[None, :]
                    i_p = nl.arange(TILE_M*bm, (TILE_M*bm)+TILE_M)[:, None]
                    bm_p = (TILES_IN_BLOCK_M//NUM_CHANNELS * channel) + bm
                    i_f = BLOCK_N * n + bn * TILE_N + i_res_x
                    res = result_tiles[m, bm_p, bn, i_res_p, i_res_x]
                    numerator = (2*rank+rank_offset) * NUM_CHANNELS
                    numerator = numerator+channel
                    if NUM_LNC>1:
                      nl.store(result[nl.program_id(0), 
                                cc_div(numerator, NUM_CHANNELS, TP_DEGREE_TO_REPLICA_GROUP[tp_degree]), 
                                channel,
                                i_p, i_f],  
                                value=res, mask=(i_f)<N)
                    else:
                      nl.store(result[cc_div(numerator, NUM_CHANNELS, TP_DEGREE_TO_REPLICA_GROUP[tp_degree]), 
                                channel,
                                i_p, i_f],  
                                value=res, mask=(i_f)<N)
      #***************************************************************************************
      #--------------------------------- Helper Functions END --------------------------------
      #***************************************************************************************
 
      # If using SB to SB we load and transpose from HBM to SBUF one time.
      if use_sb_to_sb:
        load_into_lhs_sbuf(hbm_buffer=lhs)
 
      # Start of double buffered loop.
      for rank in nl.sequential_range(NUM_RANKS//2):
        #*******************************************************************************************
        #--------------------------------- Iteration One -------------------------------------------
        #*******************************************************************************************
        if use_sb_to_sb:
          # We have to reshape our 3D SBUF tensor to 2D for memory access:
          next_lhsT_sbuf = next_lhsT_sbuf.reshape((TILE_K, K//TILE_K*M))
          lhsT_sbuf = lhsT_sbuf.reshape((TILE_K, K//TILE_K*M))
        
        launch_collective_permutes(switch_buffers=False, mask=(rank<NUM_RANKS))
 
 
        if not use_sb_to_sb:
          # If using HBM reshape and load the LHS into SBUF.
          load_into_lhs_sbuf(hbm_buffer=lhs_hbm)
        else:
          # LHS is already on SBUF.
          lhsT_sbuf = lhsT_sbuf.reshape((TILE_K, NUM_CHANNELS, K//TILE_K, M//NUM_CHANNELS))
 
        # Blocking N dimension (the RHS free dimension)
        for n in nl.affine_range(NUM_BLOCK_N):
          result_tiles = nl.zeros((NUM_BLOCK_M, TILES_IN_BLOCK_M, TILES_IN_BLOCK_N,
                                nl.par_dim(TILE_M), TILE_N),
                                dtype=lhs.dtype,
                                buffer=nl.sbuf)
          run_matmul(lhs_buffer=lhsT_sbuf, n=n, result_tiles=result_tiles)
 
          # Copying the result from SBUF to HBM
          write_to_output(rank_offset=0, result_tiles=result_tiles)
 
        #*******************************************************************************************
        #--------------------------------- Iteration Two -------------------------------------------
        #*******************************************************************************************
 
        if not use_sb_to_sb:
          lhs_hbm = lhs_hbm.reshape((NUM_CHANNELS, M // NUM_CHANNELS, K))
        else:
          # We have to reshape our 4D SBUF tensor to 2D for memory access:
          lhsT_sbuf = lhsT_sbuf.reshape((TILE_K, K//TILE_K*M))
          
        launch_collective_permutes(switch_buffers=True, mask=(rank+1<NUM_RANKS))
 
        if not use_sb_to_sb:
          # If using HBM, load the next LHS into SBUF.
          load_into_lhs_sbuf(hbm_buffer=next_lhsT_hbm)
        else:
          # Next LHS is already on SBUF.
          next_lhsT_sbuf = next_lhsT_sbuf.reshape((TILE_K, NUM_CHANNELS, K//TILE_K, M//NUM_CHANNELS))
 
        if not use_sb_to_sb:
          # For iteration 2, if HBM kernel we reuse the lhsT_sbuf from iteration 1.
          lhs_buffer = lhsT_sbuf
        else:
          # We use the already loaded next_lhsT buffer.
          lhs_buffer = next_lhsT_sbuf
 
        # Blocking N dimension (the RHS free dimension)
        for n in nl.affine_range(NUM_BLOCK_N):
          # Allocate result tile.
          result_tiles = nl.zeros((NUM_BLOCK_M, TILES_IN_BLOCK_M, TILES_IN_BLOCK_N,
                                nl.par_dim(TILE_M), TILE_N),
                                dtype=lhs.dtype,
                                buffer=nl.sbuf)
          # Compute the matmul.
          run_matmul(lhs_buffer=lhs_buffer, n=n, result_tiles=result_tiles)
          # Copying the result from SBUF to HBM
          write_to_output(rank_offset=1, result_tiles=result_tiles)
 
      # We can just reshape back to the original shape HLO gave us.
      result = result.reshape(original_result_shape)
 
      # Compile time constant so if statement works.
      if has_token:
        nisa._tiled_offloaded_memcpy(src=input_token, dst=output_token)
