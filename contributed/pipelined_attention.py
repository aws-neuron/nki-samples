"""
Kernel with software pipelining, adapted from official attention nki sample

Author: Hongyi Jin (hongyij@andrew.cmu.edu)

WARNING: These kernels:
   - Are tested only against internal nightly builds
   - May not be compatible with public NeuronSDK releases
   - Have not been extensively tested across all input configurations
   - Carry no compatibility guarantees
   - The behavior of these kernels may be modified without prior notice

"""
import numpy as np

import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
from neuronxcc import nki

from neuronxcc.nki.language import par_dim
from dataclasses import dataclass

from neuronxcc.starfish.support.dtype import bfloat16


sb_mod = nki.compiler.sbuf.mod_alloc
psum_mod = nki.compiler.psum.mod_alloc

def mm1_dot_psum_alloc(idx, pdim_size, fdim_size):
  grp_i, _, tile_i = idx
  grp_i = grp_i % 2
  return (tile_i % 4), 0, 0

def mm2_dot_psum_alloc(idx, pdim_size, fdim_size):
  grp_i, tile_i = idx
  return 4 + (tile_i % 4), 0, 0

def exp_tp_psum_alloc(idx, pdim_size, fdim_size):
  grp_i, tile_i, tp_grp_i = idx
  grp_i = grp_i % 2
  return tp_grp_i , 0, 0


# This kernel can only run on 16k seqlen, 
@nki.compiler.skip_middle_end_transformations
@nki.baremetal(artifacts_dir="debug", additional_compile_opt="--internal-skip-backend-allocation-opt-nki --disable-internal-io-dge")
def flash_fwd(q, k, v,
              softmax_scale=None,
              use_causal_mask=True,
              mixed_precision=True,
              ):
  """
  Flash Attention Forward kernel

  IO tensor layouts:
    - q: shape   (bs, n_heads, d, seq_q)
    - k: shape   (bs, nk_heads, d, seq_k)
    - v: shape   (bs, nv_heads, d, seq_v) if config.should_transpose_v  else (bs, nv_heads, seq_v, d)
    - o: shape (bs, n_heads, seq_q, d)
    - This kernel requires seq_k == seq_v

  IO tensor dtypes:
    - This kernel assumes all IO tensors have the same dtype
    - If mixed_precision is True, then all Tensor Engine operation will be performed in
      bfloat16 and accumulation will be performed in float32. Otherwise the intermediates
      will be in the same type as the inputs.

  Compile-time Constants:
    - softmax_scale: scaling for softmax, is None, default is `1.0/(d**0.5)`
    - mixed_precision: flag to set non-matmul ops in fp32 precision, default is set to `true`, if false, we use same precision as input types
  """
  b, d, seqlen_q = q.shape
  _, _, seqlen_k = k.shape
  
  assert use_causal_mask == False, "causal mask code path disabled"

  assert tuple(v.shape) == (b, seqlen_k, d), f"Expect shape of V to be {(b, seqlen_k, d)} (batch, heads, seqlen_k, d_head) but got {v.shape}"
  assert tuple(k.shape) == (b, d, seqlen_k), f"Expect shape of K to be {(b, d, seqlen_k)} (batch, heads, d_head, seqlen_k) but got {k.shape}"
  assert d <= 128, f" we do not support head_dim > 128, got head dim {d}"
  kernel_dtype = nl.bfloat16 if mixed_precision else q.dtype
  acc_type = np.dtype(np.float32) if mixed_precision else kernel_dtype

  o = nl.ndarray((b, seqlen_q, d), dtype=q.dtype, buffer=nl.shared_hbm)

  batch_id = nl.program_id(0)
  softmax_scale = softmax_scale or (1.0 / (d ** 0.5))

  sb_p = 128
  assert seqlen_k % sb_p == 0
  num_grps = seqlen_k // sb_p
  section_len = 8192
  num_sections = seqlen_q // section_len

  num_2048_tiles_per_section = section_len // 2048
  num_512_tiles_per_section = section_len // 512
  num_128_tiles_per_section = section_len // 128

  sca = 0

  identity = nl.shared_constant(np.identity(128, dtype=np.int8), dtype=nl.bfloat16)
  identity_load = nl.ndarray((par_dim(128), 128), dtype=nl.bfloat16, buffer=sb_mod(base_addr=sca))
  id_p, id_f = nl.mgrid[0:128, 0:128]
  identity_load[id_p, id_f] = nl.load(identity)
  sca += 128 * 2

  zero_bias_tensor = nl.ndarray((128, 1), dtype=nl.float32, buffer=sb_mod(base_addr=sca))
  zero_bias_tensor[...] = 0.0
  sca += 4

  running_max = nl.ndarray((sb_p, num_grps), dtype=nl.float32, buffer=sb_mod(base_addr=sca))
  sca += num_grps * 4
  running_sum = nl.ndarray((sb_p, num_grps), dtype=nl.float32, buffer=sb_mod(base_addr=sca))
  sca += num_grps * 4
  div_25_sbuf = nl.ndarray((128, num_grps), dtype=nl.float32, buffer=sb_mod(base_addr=sca))
  sca += num_grps * 4

  for section_i in nl.sequential_range(num_sections):  
    num_2048_tiles_cur_section = num_2048_tiles_per_section
    num_512_tiles_cur_section = num_512_tiles_per_section
    num_128_tiles_cur_section = num_128_tiles_per_section

    p, n = d, 128*4
    k_loaded = nl.ndarray((num_512_tiles_cur_section, nl.par_dim(p), n), dtype=nl.bfloat16, buffer=sb_mod(base_addr=sca, num_free_tiles=(num_512_tiles_cur_section, )))
    sca += num_512_tiles_cur_section * n * 2
    for i in nl.affine_range(num_512_tiles_cur_section):
      ip_k, if_k = nl.mgrid[0:p, 0:n]
      k_loaded[i, ip_k, if_k] = nl.load(k[batch_id, ip_k, section_len*section_i+512*i+if_k], dtype=nl.bfloat16)

    p, n = sb_p, d
    v_loaded = nl.ndarray((num_128_tiles_cur_section, nl.par_dim(p), n), dtype=nl.bfloat16, buffer=sb_mod(base_addr=sca, num_free_tiles=(num_128_tiles_cur_section, )))
    sca += num_128_tiles_cur_section * n * 2
    for i in nl.affine_range(num_128_tiles_cur_section):
      ip_v, if_v = nl.mgrid[0:p, 0:n]
      v_loaded[i, ip_v, if_v] = nl.load(v[batch_id, section_len*section_i + i * 128 + ip_v, if_v], dtype=nl.bfloat16)

    q_loaded = nl.ndarray((num_grps, nl.par_dim(d), sb_p), dtype=nl.bfloat16, buffer=sb_mod(base_addr=sca, num_free_tiles=(2, )))
    sca += 2 * sb_p * 2
    reduce14_num_parts = 128
    scaling_factor = nl.ndarray((num_grps, nl.par_dim(reduce14_num_parts), 1), dtype=nl.float32, buffer=sb_mod(base_addr=sca, num_free_tiles=(2, )))
    sca += 2 * 1 * 4
    num_blks = num_512_tiles_cur_section
    temp_reduce14_sbuf = nl.ndarray((num_grps, nl.par_dim(reduce14_num_parts), num_blks), dtype=nl.float32, buffer=sb_mod(base_addr=sca, num_free_tiles=(1, )))
    sca += 2 * num_blks * 4

    p, n = d,sb_p
    psum_p, psum_n = 128, sb_p * 4 # (128, 512)
    sbuf_p, sbuf_n = psum_p, psum_n*4 # (128, 2048)

    mhlo_mul_2 = nl.ndarray((num_grps, num_2048_tiles_cur_section, nl.par_dim(sbuf_p), sbuf_n), dtype=nl.float32, buffer=sb_mod(base_addr=sca, num_free_tiles=(2, num_2048_tiles_cur_section)))
    sca += num_2048_tiles_cur_section * sbuf_n * 4 * 2
    mm1_psum_dot = nl.ndarray((num_grps, num_2048_tiles_cur_section, 4, nl.par_dim(psum_p), psum_n), dtype=nl.float32, buffer=nki.compiler.psum.alloc(mm1_dot_psum_alloc))

    final_reduce_max = nl.ndarray((num_grps, nl.par_dim(128), 1), dtype=nl.float32, buffer=sb_mod(base_addr=sca, num_free_tiles=(1, )))
    sca += 2 * 4

    prev_runnning_max = nl.ndarray((num_grps, nl.par_dim(reduce14_num_parts), 1), dtype=nl.float32, buffer=sb_mod(base_addr=sca, num_free_tiles=(1, )))
    sca += 2 * 4

    exp_inst_elems = 2048
    exp_insts = 2048 // exp_inst_elems # num of exp insts per si iter
    ip_final_reduce_sum, _ = nl.mgrid[0:128, 0:1]
    final_reduce_sum_b = nl.ndarray((num_grps, nl.par_dim(128), section_len//exp_inst_elems), dtype=nl.float32, buffer=sb_mod(base_addr=sca, num_free_tiles=(2, )))
    sca += 2 * (section_len // exp_inst_elems) * 4

    final_reduce_sum_b_collect = nl.ndarray((num_grps, nl.par_dim(128), 1), dtype=nl.float32, buffer=sb_mod(base_addr=sca, num_free_tiles=(1, )))
    sca += 2 * 1 * 4

    prev_running_sum = nl.ndarray(shape=(num_grps, nl.par_dim(128), 1), dtype=nl.float32, buffer=sb_mod(base_addr=sca, num_free_tiles=(1, )))
    sca += 2 * 1 * 4

    prev_output =  nl.ndarray((num_grps, nl.par_dim(128), 128), dtype=o.dtype, buffer=sb_mod(base_addr=sca, num_free_tiles=(1, )))
    sca += 2 * 128 * 4
    mm2_sbuf_res = nl.ndarray((num_grps, nl.par_dim(128), 128), dtype=q.dtype, buffer=sb_mod(base_addr=sca, num_free_tiles=(1, )))
    sca += 2 * 128 * 4
    mm2_div_sbuf = nl.ndarray((num_grps, nl.par_dim(128), 128), dtype=q.dtype, buffer=sb_mod(base_addr=sca, num_free_tiles=(1, )))
    sca += 2 * 128 * 4

    num_tps = exp_inst_elems // 128
    num_tp_grps = num_tps // 4
    num_tps_in_grp = 4
    n_per_part = num_tps_in_grp * 128


    # p, n, access_n = 128, 2048, exp_inst_elems
    exp6_sbuf = nl.ndarray((num_grps, num_2048_tiles_cur_section, nl.par_dim(p), 2048), dtype=nl.bfloat16, buffer=sb_mod(base_addr=sca, num_free_tiles=(1, num_2048_tiles_cur_section)))
    sca += 2 * num_2048_tiles_cur_section * 2048 
    
    tp_sbuf = nl.ndarray((num_grps, num_2048_tiles_cur_section, num_tp_grps, nl.par_dim(128), n_per_part), dtype=nl.bfloat16, 
                      buffer=sb_mod(base_addr=sca, num_free_tiles=(1, num_2048_tiles_cur_section, num_tp_grps)))
    sca += num_2048_tiles_cur_section * num_tp_grps * 2 * n_per_part
    
    mm2_p, mm2_n = sb_p, d
    mm2_sbuf = nl.ndarray((num_grps, nl.par_dim(mm2_p), mm2_n), dtype=nl.float32, buffer=sb_mod(base_addr=sca, num_free_tiles=(1, )))
    sca += 2 * mm2_n * 4
    tp_psum = nl.ndarray((num_grps, num_2048_tiles_cur_section, num_tp_grps, nl.par_dim(128), n_per_part), dtype=nl.float32, buffer=nki.compiler.psum.alloc(exp_tp_psum_alloc))

    mm2_psum = nl.ndarray((num_grps, num_2048_tiles_cur_section, nl.par_dim(sb_p), mm2_n), dtype=nl.float32, buffer=nki.compiler.psum.alloc(mm2_dot_psum_alloc))
    iq_p, iq_f = nl.mgrid[0:p, 0:n]
    def load_q(grp_i):
      q_loaded[grp_i, iq_p, iq_f] = nl.load(q[batch_id, iq_p, grp_i*n+iq_f])

    def qk_and_max(grp_i):
      for si in nl.affine_range(num_2048_tiles_cur_section):
        for pi in nl.affine_range(4):
          loc_512_tile_i = si*4+pi
          ip_res, if_res = nl.mgrid[0:128, 0:512]
          ip_reduce_res, _ = nl.mgrid[0:128, 0:1]
          mm1_psum_dot[grp_i, si, pi, ip_res, if_res] = nisa.nc_matmul(q_loaded[grp_i, :, :], k_loaded[loc_512_tile_i, :, :])
          mhlo_mul_2[grp_i, si, ip_res, pi*512+if_res] = nisa.tensor_scalar_reduce(
            data=mm1_psum_dot[grp_i, si, pi, ip_res, if_res], op0=np.multiply, operand0=softmax_scale,
            reduce_op=nl.max, reduce_res=temp_reduce14_sbuf[grp_i, ip_reduce_res, si*4+pi], name="mm1-tsp"
            )
          
    def update_max(grp_i):
      ip_reduce, _= nl.mgrid[0:128, 0:1]
      final_reduce_max[grp_i, ip_reduce, 0] = nisa.tensor_reduce(np.max, temp_reduce14_sbuf[grp_i], 1, negate=True)
      if section_i == 0:
        running_max[ip_reduce, grp_i] = nisa.tensor_copy(final_reduce_max[grp_i])
      if section_i > 0:
        prev_runnning_max[grp_i, ip_reduce, 0] = nisa.activation(np.copy, running_max[ip_reduce, grp_i], scale=-1.0, bias=zero_bias_tensor)
        running_max[ip_reduce, grp_i] = nisa.tensor_tensor(running_max[ip_reduce, grp_i], final_reduce_max[grp_i], op=nl.minimum)
        scaling_factor[grp_i, ip_reduce, 0] = nisa.activation(np.exp, prev_runnning_max[grp_i], bias=running_max[ip_reduce, grp_i], scale=1.0)
      
    assert section_len//exp_inst_elems == num_2048_tiles_cur_section
    def exp(grp_i):
      ip_reduce, _= nl.mgrid[0:128, 0:1]
      for si in nl.affine_range(num_2048_tiles_cur_section):
        p, n, access_n = 128, 2048, exp_inst_elems
        ip_p, ip_n = nl.mgrid[0:p, 0:access_n]
        for pi in nl.affine_range(exp_insts): # This loop doesn't actually exist with current config, exp_insts==1 in current config
          exp6_sbuf[grp_i, si, ip_p, pi*access_n+ip_n] = nisa.activation_reduce(np.exp, mhlo_mul_2[grp_i, si, ip_p, access_n*pi+ip_n], 
                                                                    reduce_op=np.add, reduce_res=final_reduce_sum_b[grp_i, ip_final_reduce_sum, si*exp_insts+pi],
                                                                    bias=running_max[ip_reduce, grp_i], name='exp6',
                                                                    )
  
    def tp(grp_i):
      for si in nl.affine_range(num_2048_tiles_cur_section):  
        for tp_grp in nl.affine_range(num_tp_grps):
          ip_tp, if_tp = nl.mgrid[0:128, 0:128]
          ip_cp, if_cp = nl.mgrid[0:128, 0:n_per_part]
          for ti in nl.affine_range(num_tps_in_grp):
            tp_psum[grp_i, si, tp_grp, ip_tp, ti*128+if_tp] = nisa.nc_matmul(exp6_sbuf[grp_i, si, ip_tp, tp_grp*n_per_part+ti*128+if_tp], identity_load)
          tp_sbuf[grp_i, si, tp_grp, ip_cp, if_cp] = nisa.tensor_copy(tp_psum[grp_i, si, tp_grp], dtype=nl.bfloat16, name='tp-act-cp', 
                                                                    )
    def pv(grp_i):
      mm2_sbuf[grp_i] = 0.0
      for mm2i in nl.affine_range(num_2048_tiles_cur_section):
        num_tp_grps_in_2048_tile = 4
        # mm2_psum = nl.zeros((nl.par_dim(sb_p), mm2_n), dtype=nl.float32, buffer=nl.psum)
        for tp_grp_i in nl.affine_range(num_tp_grps_in_2048_tile):
          mm2_num_grps = 4
          ip_mm2, if_mm2 = nl.mgrid[0:128, 0:128]
          for mm2_si in nl.affine_range(mm2_num_grps):
            mm2_psum[grp_i, mm2i, ip_mm2, if_mm2] += nisa.nc_matmul(tp_sbuf[grp_i, mm2i, tp_grp_i, ip_mm2, mm2_si*128+if_mm2],v_loaded[mm2i*16+tp_grp_i*4+mm2_si, ip_mm2, if_mm2])
        mm2_sbuf[grp_i] = nl.loop_reduce(mm2_psum[grp_i, mm2i], np.add, loop_indices=[mm2i],name='mm2-itt',
                                        )
        
    def fused_qkmax_and_pv(grp_i):
      mm2_sbuf[grp_i] = 0.0
      for si in nl.affine_range(num_2048_tiles_cur_section):
        for pi in nl.affine_range(4):
          loc_512_tile_i = si*4+pi
          ip_res, if_res = nl.mgrid[0:128, 0:512]
          ip_reduce_res, _ = nl.mgrid[0:128, 0:1]
          mm1_psum_dot[grp_i+2, si, pi, ip_res, if_res] = nisa.nc_matmul(q_loaded[grp_i+2, :, :], k_loaded[loc_512_tile_i, :, :])
          mhlo_mul_2[grp_i+2, si, ip_res, pi*512+if_res] = nisa.tensor_scalar_reduce(
            data=mm1_psum_dot[grp_i+2, si, pi, ip_res, if_res], op0=np.multiply, operand0=softmax_scale,
            reduce_op=nl.max, reduce_res=temp_reduce14_sbuf[grp_i+2, ip_reduce_res, si*4+pi], name="mm1-tsp"
            )
        mm2i=si
        num_tp_grps_in_2048_tile = 4
        # mm2_psum = nl.zeros((nl.par_dim(sb_p), mm2_n), dtype=nl.float32, buffer=nl.psum)
        for tp_grp_i in nl.affine_range(num_tp_grps_in_2048_tile):
          mm2_num_grps = 4
          ip_mm2, if_mm2 = nl.mgrid[0:128, 0:128]
          for mm2_si in nl.affine_range(mm2_num_grps):
            mm2_psum[grp_i, mm2i, ip_mm2, if_mm2] += nisa.nc_matmul(tp_sbuf[grp_i, mm2i, tp_grp_i, ip_mm2, mm2_si*128+if_mm2],v_loaded[mm2i*16+tp_grp_i*4+mm2_si, ip_mm2, if_mm2])
        mm2_sbuf[grp_i] = nl.loop_reduce(mm2_psum[grp_i, mm2i], np.add, loop_indices=[mm2i],name='mm2-itt')
             
    def write_back(grp_i):
      ip_o, if_o = nl.mgrid[0:128,0:128]
      
      ip_reduce, _= nl.mgrid[0:128, 0:1]
      final_reduce_sum_b_collect[grp_i] = nisa.tensor_reduce(np.sum, final_reduce_sum_b[grp_i], axis=(1,))
      if section_i == 0:
        running_sum[ip_reduce, grp_i] = nisa.tensor_copy(final_reduce_sum_b_collect[grp_i])
      if section_i > 0: 
        prev_running_sum[grp_i] = nisa.tensor_copy(running_sum[ip_reduce, grp_i])
        running_sum[ip_reduce, grp_i] = nisa.tensor_scalar(prev_running_sum[grp_i, ip_reduce, 0], np.multiply, scaling_factor[grp_i], op1=nl.add, operand1=final_reduce_sum_b_collect[grp_i])
      if section_i == num_sections - 1:
        div_25_sbuf[ip_reduce, grp_i] = nisa.reciprocal(running_sum[ip_reduce, grp_i])
      
      if section_i == 0:
        nl.store(o[batch_id, grp_i*sb_p+ip_o, if_o], value=mm2_sbuf[grp_i])

      if section_i == num_sections - 1:
        prev_output[grp_i] = nl.load(o[batch_id, grp_i*sb_p+ip_o, if_o])
        mm2_sbuf_res[grp_i] = nisa.scalar_tensor_tensor(data=prev_output[grp_i], op0=np.multiply, operand0=scaling_factor[grp_i], op1=np.add, operand1=mm2_sbuf[grp_i])
        
        mm2_div_sbuf[grp_i] = nisa.activation(np.copy, mm2_sbuf_res[grp_i], scale=div_25_sbuf[ip_reduce, grp_i],bias=zero_bias_tensor)
        nl.store(o[batch_id, grp_i*sb_p+ip_o, if_o], value=mm2_div_sbuf[grp_i])
    load_q(0)
    qk_and_max(0)
    update_max(0)
    exp(0)
    tp(0)
    load_q(1)
    qk_and_max(1)
    update_max(1)
    for grp_i in nl.affine_range(num_grps-2,  precise_schedule=True): # for each block of seq_q
      load_q(grp_i+2)
      exp(grp_i+1)
      fused_qkmax_and_pv(grp_i)
      tp(grp_i+1)
      write_back(grp_i)
      update_max(grp_i+2)
    pv(num_grps-2)
    write_back(num_grps-2)
    exp(num_grps-1)
    tp(num_grps-1)
    pv(num_grps-1)
    write_back(num_grps-1)
    
      
  return o

def softmax(x: np.ndarray, dim: int, zero_max_mode=False,
            mixed_precision=False):
    max_value = np.amax(x, axis=dim, keepdims=True)
    max_value = np.maximum(0, max_value) if zero_max_mode else max_value

    exp = np.exp(x - max_value)

    reduce = np.add.reduce(exp.astype(np.float32), axis=dim, keepdims=True).astype(x.dtype)

    result = exp / reduce

    return exp / reduce

def cpu_attention_forward(q, k, v, softmax_scale, use_causal_mask=True, mixed_precision=True):
    def mixed_precision_matmul(a, b):
        input_dtype = a.dtype
        a, b = a.astype(np.float32), b.astype(np.float32)
        c = np.matmul(a, b)
        return c.astype(input_dtype)

    # Compute golden output
    q_scaled = q * softmax_scale

    raw_score = mixed_precision_matmul(q_scaled.transpose(0, 2, 1), k)

    if use_causal_mask:
        # raw_score has K seq in the most inner dim
        # we want to mask all elements where Q idx is smaller than K idx with -inf
        # this maps to the upper triangle of the final two axes
        for i in range(raw_score.shape[0]):
            for j in range(raw_score.shape[1]):
                # -inf triggers invalid input error in softmax implementation, use a small negative instead
                # k=1 to exclude the diagonal, because each token can still attend to itself
                raw_score[i, j][np.triu_indices_from(raw_score[i, j], k=1)] = -9984.0

    norm_score = softmax(raw_score, dim=-1, mixed_precision=mixed_precision)

    # Transpose the result so it has the same layout as ours
    out_golden = mixed_precision_matmul(norm_score, v)

    return out_golden, norm_score

bs=1
seqlen_q=16*1024
seqlen_k=16*1024
d=128
softmax_scale=0.125
dtype=bfloat16

q = np.random.rand(bs, d, seqlen_q).astype(dtype)
k = np.random.rand(bs, d, seqlen_k).astype(dtype)
v = np.random.rand(bs, seqlen_k, d).astype(dtype)
o = flash_fwd[0](q, k, v, mixed_precision=True, softmax_scale=softmax_scale, use_causal_mask=False)
o_std, scores = cpu_attention_forward(q, k, v, softmax_scale, use_causal_mask=False, mixed_precision=True)

np.testing.assert_allclose(o, o_std, atol=1e-2)
