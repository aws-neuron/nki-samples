import sys

from neuronxcc import nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as ni
import numpy as np

def lora_bwd(I_DRAM, PW_DRAM, A_DRAM, B_DRAM, AI_DRAM, dO_DRAM, alpha):
  K, M = PW_DRAM.shape
  R, N = AI_DRAM.shape
  dtype = PW_DRAM.dtype

  dB_DRAM = nl.ndarray([R, M], dtype=dtype, buffer=nl.shared_hbm)
  dA_DRAM = nl.ndarray([R, K], dtype=dtype, buffer=nl.shared_hbm)
  dI_DRAM = nl.ndarray([K, N], dtype=dtype, buffer=nl.shared_hbm)

  K0 = 128
  M0 = 128
  N0 = 128

  M1 = 8
  N1 = 8
  K1 = 4

  K2 = K // (K1 * K0)
  M2 = M // (M1 * M0)
  N2 = N // (N1 * N0)

  assert K2 * K1 * K0 == K
  assert M2 * M1 * M0 == M
  assert N2 * N1 * N0 == N
  assert R <= 128

  AI_SBUF = nl.ndarray((N2, nl.par_dim(R), N1 * N0), dtype=dtype, buffer=nl.sbuf)
  AI_T_SBUF = nl.ndarray((N2, N1, nl.par_dim(N0), R), dtype=dtype, buffer=nl.sbuf)

  dSB_SBUF = nl.zeros((M2, nl.par_dim(R), M1 * M0), dtype=dtype, buffer=nl.sbuf)
  dA_SBUF = nl.zeros((K2, nl.par_dim(R), K1 * K0), dtype=dtype, buffer=nl.sbuf)

  B_SBUF = nl.ndarray((M2, nl.par_dim(R), M1 * M0), dtype=dtype, buffer=nl.sbuf)
  B_T_SBUF = nl.ndarray((M2, M1, nl.par_dim(M0), R), dtype=dtype, buffer=nl.sbuf)
  for m2 in nl.affine_range(M2):
    m_start = m2 * M1 * M0
    m_end = m_start + M1 * M0

    B_SBUF[m2] = nl.load(B_DRAM[:, m_start:m_end])

    for m1 in nl.affine_range(M1):
      m_start = m1 * M0
      m_end = m_start + M0

      B_T_PSUM = ni.nc_transpose(B_SBUF[m2, :, m_start:m_end])
      B_T_SBUF[m2, m1] = ni.tensor_scalar(B_T_PSUM, op0=np.add, operand0=1, op1=np.add, operand1=-1)

  A_SBUF = nl.ndarray((K2, nl.par_dim(R), K1 * K0), dtype=dtype, buffer=nl.sbuf)
  for k2 in nl.affine_range(K2):
    k_start = k2 * K1 * K0
    k_end = k_start + K1 * K0

    A_SBUF[k2] = nl.load(A_DRAM[:, k_start:k_end])

  for n2 in nl.affine_range(N2):
    n_start = n2 * N1 * N0
    n_end = n_start + N1 * N0

    AI_SBUF[n2] = nl.load(AI_DRAM[:, n_start:n_end])
    dSAI_SBUF = nl.zeros((N1, nl.par_dim(R), N0), dtype=dtype, buffer=nl.sbuf)
    dAI_SBUF = nl.zeros((N1, nl.par_dim(R), N0), dtype=dtype, buffer=nl.sbuf)

    # Transpose AI
    for n1 in nl.affine_range(N1):
      n_start = n1 * N0
      n_end = n_start + N0

      AI_T_PSUM = ni.nc_transpose(AI_SBUF[n2, :, n_start:n_end])
      AI_T_SBUF[n2, n1] = ni.tensor_scalar(AI_T_PSUM, op0=np.add, operand0=1, op1=np.add, operand1=-1)

    for k2 in nl.affine_range(K2):
      dI_PW_SBUF = nl.ndarray((K1, nl.par_dim(K0), N1 * N0), dtype=dtype, buffer=nl.sbuf)

      for m2 in nl.affine_range(M2):

        dO_SBUF = nl.ndarray((M1, nl.par_dim(M0), N1 * N0), dtype=dtype, buffer=nl.sbuf)
        dO_T_SBUF = nl.ndarray((M1, N1, nl.par_dim(N0), M0), dtype=dtype, buffer=nl.sbuf)

        dO_T_PSUM = nl.zeros((M1, N1, nl.par_dim(N0), M0), dtype=nl.float32, buffer=nl.psum)

        for m1 in nl.affine_range(M1):
          m_start = m2 * M1 * M0 + m1 * M0
          m_end = m_start + M0

          n_start = n2 * N1 * N0
          n_end = n_start + N1 * N0

          dO_SBUF[m1] = nl.load(dO_DRAM[m_start:m_end, n_start:n_end])

        # if k2 == 0:
        for m1 in nl.affine_range(M1):
          dSB_PSUM = nl.zeros((nl.par_dim(R), M0), dtype=nl.float32, buffer=nl.psum)

          for n1 in nl.affine_range(N1):
            n_start = n1 * N0
            n_end = n_start + N0

            dO_T_PSUM[m1, n1] = ni.nc_transpose(dO_SBUF[m1, :, n_start:n_end], mask=k2==0)
            dO_T_SBUF[m1, n1] = ni.tensor_scalar(dO_T_PSUM[m1, n1], op0=np.add, operand0=1, op1=np.add, operand1=-1, mask=k2==0)

            m_start = m2 * M1 * M0 + m1 * M0
            m_end = m_start + M0

            dSB_PSUM += ni.nc_matmul(AI_T_SBUF[n2, n1], dO_T_SBUF[m1, n1], mask=k2==0)

          m_start = m1 * M0
          m_end = m_start + M0
          dSB_SBUF[m2, :, m_start:m_end] = nl.loop_reduce(dSB_PSUM, op=np.add, loop_indices=[n2, k2], mask=k2==0)

        for n1 in nl.affine_range(N1):
          dSAI_PSUM = nl.zeros((nl.par_dim(R), N0), dtype=nl.float32, buffer=nl.psum)

          for m1 in nl.affine_range(M1):
            n_start = n1 * N0
            n_end = n_start + N0

            dSAI_PSUM += ni.nc_matmul(B_T_SBUF[m2, m1], dO_SBUF[m1, :, n_start:n_end], mask=k2==0)

          dSAI_SBUF[n1] = nl.loop_reduce(dSAI_PSUM, op=np.add, loop_indices=[m2, k2], mask=k2==0)

        # endif

        PW_SBUF = nl.ndarray((K1, nl.par_dim(K0), M1 * M0), dtype=dtype, buffer=nl.sbuf)
        PW_T_SBUF = nl.ndarray((K1, M1, nl.par_dim(M0), K0), dtype=dtype, buffer=nl.sbuf)
        for k1 in nl.affine_range(K1):
          k_start = k2 * K1 * K0 + k1 * K0
          k_end = k_start + K0

          m_start = m2 * M1 * M0
          m_end = m_start + M1 * M0

          PW_SBUF[k1] = nl.load(PW_DRAM[k_start:k_end, m_start:m_end])

          for m1 in nl.affine_range(M1):
            m_start = m1 * M0
            m_end = m_start + M0

            PW_T_PSUM = ni.nc_transpose(PW_SBUF[k1, :, m_start:m_end])
            PW_T_SBUF[k1, m1] = ni.tensor_scalar(PW_T_PSUM, op0=np.add, operand0=1, op1=np.add, operand1=-1)

          for n1 in nl.affine_range(N1):
            dI_PW_PSUM = nl.zeros((nl.par_dim(K0), N0), dtype=nl.float32, buffer=nl.psum)

            n_start = n1 * N0
            n_end = n_start + N0

            for m1 in nl.affine_range(M1):
              dI_PW_PSUM += ni.nc_matmul(PW_T_SBUF[k1, m1], dO_SBUF[m1, :, n_start:n_end])

            dI_PW_SBUF[k1, :, n_start:n_end] = nl.loop_reduce(dI_PW_PSUM, op=np.add, loop_indices=[m2])


      dI_A_SBUF = nl.ndarray((K1, nl.par_dim(K0), N1 * N0), dtype=dtype, buffer=nl.sbuf)
      dAI_T_SBUF = nl.ndarray((N1, nl.par_dim(N0), R), dtype=dtype, buffer=nl.sbuf)
      for n1 in nl.affine_range(N1):
        dAI_SBUF[n1] = ni.tensor_scalar(dSAI_SBUF[n1], np.multiply, alpha, mask=k2==0)

        dAI_T_PSUM = ni.nc_transpose(dAI_SBUF[n1])
        dAI_T_SBUF[n1] = ni.tensor_scalar(dAI_T_PSUM, op0=np.add, operand0=1, op1=np.add, operand1=-1)

        for k1 in nl.affine_range(K1):
          k_start = k1 * K0
          k_end = k_start + K0

          dI_A_PSUM = ni.nc_matmul(A_SBUF[k2, :, k_start:k_end], dAI_SBUF[n1])

          n_start = n1 * N0
          n_end = n_start + N0

          dI_A_SBUF[k1, :, n_start:n_end] = ni.tensor_scalar(dI_A_PSUM, op0=np.add, operand0=1, op1=np.add, operand1=-1)

      dI_SBUF = nl.ndarray((K1, nl.par_dim(K0), N1 * N0), dtype=dtype, buffer=nl.sbuf)
      I_SBUF = nl.ndarray((K1, nl.par_dim(K0), N1 * N0), dtype=dtype, buffer=nl.sbuf)
      I_T_SBUF = nl.ndarray((K1, N1, nl.par_dim(N0), K0), dtype=dtype, buffer=nl.sbuf)
      for k1 in nl.affine_range(K1):
        k_start = k2 * K1 * K0 + k1 * K0
        k_end = k_start + K0

        n_start = n2 * N1 * N0
        n_end = n_start + N1 * N0

        dI_SBUF[k1] = ni.tensor_tensor(dI_A_SBUF[k1], dI_PW_SBUF[k1], np.add)
        nl.store(dI_DRAM[k_start:k_end, n_start:n_end], value=dI_SBUF[k1])

        I_SBUF[k1] = nl.load(I_DRAM[k_start:k_end, n_start:n_end])

        dA_PSUM = nl.zeros((nl.par_dim(R), K0), dtype=nl.float32, buffer=nl.psum)
        for n1 in nl.affine_range(N1):
          n_start = n1 * N0
          n_end = n_start + N0

          I_T_PSUM = ni.nc_transpose(I_SBUF[k1, :, n_start:n_end])
          I_T_SBUF[k1, n1] = nl.copy(I_T_PSUM)

          dA_PSUM += ni.nc_matmul(dAI_T_SBUF[n1], I_T_SBUF[k1, n1])

        k_start = k1 * K0
        k_end = k_start + K0

        dA_SBUF[k2, :, k_start:k_end] = nl.loop_reduce(dA_PSUM, op=np.add, loop_indices=[n2])


  dB_SBUF = nl.ndarray((M2, nl.par_dim(R), M1 * M0), dtype=dtype, buffer=nl.sbuf)
  for m2 in nl.affine_range(M2):
    dB_SBUF[m2] = ni.tensor_scalar(dSB_SBUF[m2], np.multiply, alpha)

    m_start = m2 * M1 * M0
    m_end = m_start + M1 * M0

    nl.store(dB_DRAM[:, m_start:m_end], value=dB_SBUF[m2])

  for k2 in nl.affine_range(K2):
    k_start = k2 * K1 * K0
    k_end = k_start + K1 * K0

    nl.store(dA_DRAM[:, k_start:k_end], value=dA_SBUF[k2])

  return dB_DRAM, dA_DRAM, dI_DRAM

def benchmark_kernel():
  K, M, N, R = (11008, 4096, 1024, 128)

  # Pad
  K = ((K - 1) // 512 + 1) * 512
  M = ((M - 1) // 1024 + 1) * 1024

  # Rank orders are carried over from the forward pass
  I = np.random.random_sample([K, N]).astype(np.float16)
  PW = np.random.random_sample([K, M]).astype(np.float16)
  A = np.random.random_sample([R, K]).astype(np.float16)
  B = np.random.random_sample([R, M]).astype(np.float16)
  AI = np.random.random_sample([R, N]).astype(np.float16)
  dO = np.random.random_sample([M, N]).astype(np.float16)

  alpha = 256

  benchmark_func = nki.benchmark(save_neff_name="file.neff")(lora_bwd)
  benchmark_func(I, PW, A, B, AI, dO, alpha)

def check_correct():
  K, M, N, R = (1024, 1024, 2048, 8)

  # Rank orders are carried over from the forward pass
  I = np.random.random_sample([K, N]).astype(np.float16)
  PW = np.random.random_sample([K, M]).astype(np.float16)
  A = np.random.random_sample([R, K]).astype(np.float16)
  B = np.random.random_sample([R, M]).astype(np.float16)
  AI = np.random.random_sample([R, N]).astype(np.float16)
  dO = np.random.random_sample([M, N]).astype(np.float16)

  alpha = 1 / 16

  baremetal_func = nki.baremetal(save_neff_name="file.neff")(lora_bwd)
  dB, dA, dI = baremetal_func(I, PW, A, B, AI, dO, alpha)

  dSB_corr = AI @ dO.T
  dB_corr = alpha * dSB_corr
  dSAI_corr = B @ dO
  dAI_corr = alpha * dSAI_corr
  dI_A_corr = A.T @ dAI_corr
  dI_PW_corr = PW @ dO
  dI_corr = dI_A_corr + dI_PW_corr
  dA_corr = dAI_corr @ I.T

  print("Is dB close?", np.all(np.isclose(dB, dB_corr, atol=1e-4, rtol=1e-2)))
  print("Is dA close?", np.all(np.isclose(dA, dA_corr, atol=1e-4, rtol=1e-2)))
  print("Is dI close?", np.all(np.isclose(dI, dI_corr, atol=1e-4, rtol=1e-2)))

def main():
  check_correct()
  benchmark_kernel()

if __name__ == "__main__":
  main()
