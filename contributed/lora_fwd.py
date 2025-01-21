import sys

from neuronxcc import nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as ni
import numpy as np

def lora_fwd(I_DRAM, PW_DRAM, A_DRAM, B_DRAM, dropout, alpha):
  K, N = I_DRAM.shape
  R, M = B_DRAM.shape

  AI_DRAM = nl.ndarray([R, N], dtype=A_DRAM.dtype, buffer=nl.shared_hbm)
  O_DRAM = nl.ndarray([M, N], dtype=A_DRAM.dtype, buffer=nl.shared_hbm)

  K0 = 128
  M0 = 128
  N0 = 512

  M1 = 4
  N1 = min(4, N // N0)
  K1 = 8

  K2 = K // (K1 * K0)
  M2 = M // (M1 * M0)
  N2 = N // (N1 * N0)

  assert K2 * K1 * K0 == K
  assert M2 * M1 * M0 == M
  assert N2 * N1 * N0 == N
  assert R <= 128

  for n2 in nl.affine_range(N2):
    AI_SBUF = nl.zeros((nl.par_dim(R), N1 * N0), dtype=O_DRAM.dtype, buffer=nl.sbuf)

    for m2 in nl.affine_range(M2):

      PO_SBUF = nl.zeros((M1, nl.par_dim(M0), N1 * N0), dtype=O_DRAM.dtype, buffer=nl.sbuf)
      DO_SBUF = nl.zeros((M1, nl.par_dim(M0), N1 * N0), dtype=O_DRAM.dtype, buffer=nl.sbuf)

      m_start = m2 * M1 * M0
      m_end = m_start + M1 * M0

      A_SBUF = nl.ndarray((K2, nl.par_dim(R), K1 * K0), dtype=A_DRAM.dtype, buffer=nl.sbuf)
      B_SBUF = nl.load(B_DRAM[:, m_start:m_end])

      for k2 in nl.affine_range(K2):
        PW_SBUF = nl.ndarray((K1, nl.par_dim(K0), M1 * M0), dtype=PW_DRAM.dtype, buffer=nl.sbuf)
        I_SBUF = nl.ndarray((K1, nl.par_dim(K0), N1 * N0), dtype=I_DRAM.dtype, buffer=nl.sbuf)

        for k1 in nl.affine_range(K1):
          k_start = k2 * K1 * K0 + k1 * K0
          k_end = k_start + K0

          m_start = m2 * M1 * M0
          m_end = m_start + M1 * M0

          n_start = n2 * N1 * N0
          n_end = n_start + N1 * N0

          PW_SBUF[k1] = nl.load(PW_DRAM[k_start:k_end, m_start:m_end])
          I_SBUF[k1] = nl.load(I_DRAM[k_start:k_end, n_start:n_end])

        for m1 in nl.affine_range(M1):
          for n1 in nl.affine_range(N1):
            PO_PSUM = nl.zeros((M0, N0), dtype=nl.float32, buffer=nl.psum)

            m_start = m1 * M0
            m_end = m_start + M0

            n_start = n1 * N0
            n_end = n_start + N0

            for k1 in nl.affine_range(K1):
              PO_PSUM += ni.nc_matmul(PW_SBUF[k1, :, m_start:m_end], I_SBUF[k1, :, n_start:n_end])

            PO_SBUF[m1, :, n_start:n_end] = nl.loop_reduce(PO_PSUM, op=np.add, loop_indices=[k2], dtype=O_DRAM.dtype)

        # if m2 == 0:
        DI_SBUF = nl.ndarray((K1, nl.par_dim(K0), N1 * N0), dtype=I_DRAM.dtype, buffer=nl.sbuf)
        for k1 in nl.affine_range(K1):
          for n1 in nl.affine_range(N1):
            n_start = n1 * N0
            n_end = n_start + N0

            DI_SBUF[k1, :, n_start:n_end] = nl.dropout(I_SBUF[k1, :, n_start:n_end], dropout, mask=m2==0)

        k_start = k2 * K1 * K0
        k_end = k_start + K1 * K0

        A_SBUF[k2] = nl.load(A_DRAM[:, k_start:k_end], mask=m2==0)
        A_T_SBUF = nl.ndarray((K1, nl.par_dim(K0), R), dtype=A_DRAM.dtype, buffer=nl.sbuf)

        for k1 in nl.affine_range(K1):
          k_start = k1 * K0
          k_end = k_start + K0

          A_T_PSUM = ni.nc_transpose(A_SBUF[k2, :, k_start:k_end])
          A_T_SBUF[k1] = nl.copy(A_T_PSUM)

        for n1 in nl.affine_range(N1):
          AI_PSUM = nl.zeros((nl.par_dim(R), N0), dtype=nl.float32, buffer=nl.psum)

          n_start = n1 * N0
          n_end = n_start + N0

          for k1 in nl.affine_range(K1):
            AI_PSUM += ni.nc_matmul(A_T_SBUF[k1], DI_SBUF[k1, :, n_start:n_end], mask=m2==0)

          AI_SBUF[:, n_start:n_end] = nl.loop_reduce(AI_PSUM, op=np.add, loop_indices=[m2, k2], dtype=O_DRAM.dtype, mask=m2==0)

      n_start = n2 * N1 * N0
      n_end = n_start + N1 * N0
      nl.store(AI_DRAM[:, n_start:n_end], value=AI_SBUF, mask=m2==0)

      # endif

      for m1 in nl.affine_range(M1):
        for n1 in nl.affine_range(N1):
          DO_PSUM = nl.zeros((nl.par_dim(M0), N0), dtype=nl.float32, buffer=nl.psum)

          m_start = m1 * M0
          m_end = m_start + M0

          n_start = n1 * N0
          n_end = n_start + N0

          DO_PSUM[:] = ni.nc_matmul(B_SBUF[:, m_start:m_end], AI_SBUF[:, n_start:n_end])
          DO_SBUF[m1, :, n_start:n_end] = ni.tensor_scalar(DO_PSUM, np.multiply, alpha, dtype=O_DRAM.dtype)

      for m1 in nl.affine_range(M1):
        m_start = m2 * M1 * M0 + m1 * M0
        m_end = m_start + M0

        O_SBUF = nl.ndarray((nl.par_dim(M0), N1 * N0), dtype=O_DRAM.dtype, buffer=nl.sbuf)
        for n1 in nl.affine_range(N1):
          n_start = n1 * N0
          n_end = n_start + N0

          O_SBUF[:, n_start:n_end] = nl.add(PO_SBUF[m1, :, n_start:n_end], DO_SBUF[m1, :, n_start:n_end])

        n_start = n2 * N1 * N0
        n_end = n_start + N1 * N0

        nl.store(O_DRAM[m_start:m_end, n_start:n_end], value=O_SBUF)

  return AI_DRAM, O_DRAM

def benchmark_kernel():
  K, M, N, R = (4096, 4096, 1024, 128)

  # Pad
  K = ((K - 1) // 1024 + 1) * 1024
  M = ((M - 1) // 512 + 1) * 512

  I = np.random.random_sample([K, N]).astype(np.float16)
  PW = np.random.random_sample([K, M]).astype(np.float16)
  A = np.random.random_sample([R, K]).astype(np.float16)
  B = np.random.random_sample([R, M]).astype(np.float16)

  dropout = 0.05
  alpha = 256

  benchmark_func = nki.benchmark(save_neff_name="file.neff")(lora_fwd)
  benchmark_func(I, PW, A, B, dropout, alpha)

def check_correct():
  K, M, N, R = (1024, 512, 2048, 8)

  I = np.random.random_sample([K, N]).astype(np.float16)
  PW = np.random.random_sample([K, M]).astype(np.float16)
  A = np.random.random_sample([R, K]).astype(np.float16)
  B = np.random.random_sample([R, M]).astype(np.float16)

  alpha = 0.5

  PO_corr = PW.T @ I
  AI_corr = A @ I
  DO_corr = B.T @ AI_corr
  O_corr = PO_corr + alpha * DO_corr

  # Dropout is random and so cannot be tested directly
  # With no dropout:

  dropout = 0

  baremetal_func = nki.baremetal(save_neff_name="file.neff")(lora_fwd)
  AI, O = baremetal_func(I, PW, A, B, dropout, alpha)

  print("No dropout")
  print("Is AI close?", np.all(np.isclose(AI, AI_corr, atol=1e-4, rtol=1e-2)))
  print("Is O close?", np.all(np.isclose(O, O_corr, atol=1e-4, rtol=1e-2)))

  # Dropping everything:

  dropout = 1

  baremetal_func = nki.baremetal(save_neff_name="file.neff")(lora_fwd)
  AI, O = baremetal_func(I, PW, A, B, dropout, alpha)

  print("Drop everything")
  print("Is AI close?", np.all(np.isclose(AI, np.zeros_like(AI_corr), atol=1e-4, rtol=1e-2)))
  print("Is O close?", np.all(np.isclose(O, PO_corr, atol=1e-4, rtol=1e-2)))

def main():
  check_correct()
  benchmark_kernel()

if __name__ == "__main__":
  main()
