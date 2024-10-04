from neuronxcc.nki import baremetal, benchmark
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as ni
import numpy as np

@benchmark(save_neff_name='file.neff', save_trace_name='profile.ntff', additional_compile_opt=' --disable-internal-io-dge ')
def matmul(A_DRAM, B_DRAM, O_DRAM, K2, K1, K0, M2, M1, M0, N2, N1, N0):
  for n2 in nl.affine_range(N2):
    for m2 in nl.affine_range(M2):

      O_SBUF = nl.zeros((M1, nl.par_dim(M0), N1 * N0), dtype=O_DRAM.dtype, buffer=nl.sbuf)

      for k2 in nl.affine_range(K2):
        A_SBUF = nl.ndarray((K1, nl.par_dim(K0), M1 * M0), dtype=A_DRAM.dtype, buffer=nl.sbuf)
        B_SBUF = nl.ndarray((K1, nl.par_dim(K0), N1 * N0), dtype=B_DRAM.dtype, buffer=nl.sbuf)

        for k1 in nl.affine_range(K1):
          k_start = k2 * K1 * K0 + k1 * K0
          k_end = k_start + K0

          m_start = m2 * M1 * M0
          m_end = m_start + M1 * M0

          n_start = n2 * N1 * N0
          n_end = n_start + N1 * N0

          A_SBUF[k1] = nl.load(A_DRAM[k_start:k_end, m_start:m_end])
          B_SBUF[k1] = nl.load(B_DRAM[k_start:k_end, n_start:n_end])

        for m1 in nl.affine_range(M1):
          for n1 in nl.affine_range(N1):
            PO_PSUM = nl.zeros((M0, N0), dtype=nl.float32, buffer=nl.psum)

            m_start = m1 * M0
            m_end = m_start + M0

            n_start = n1 * N0
            n_end = n_start + N0

            for k1 in nl.affine_range(K1):
              PO_PSUM += ni.nc_matmul(A_SBUF[k1, :, m_start:m_end], B_SBUF[k1, :, n_start:n_end])

            O_SBUF[m1, :, n_start:n_end] = nl.loop_reduce(PO_PSUM, op=np.add, loop_indices=[k2], dtype=O_DRAM.dtype)

      for m1 in nl.affine_range(M1):
        m_start = m2 * M1 * M0 + m1 * M0
        m_end = m_start + M0

        n_start = n2 * N1 * N0
        n_end = n_start + N1 * N0

        nl.store(O_DRAM[m_start:m_end, n_start:n_end], value=O_SBUF[m1])

def launch():
  K, M, N = (8192, 4096, 8192)

  K0 = 128
  M0 = 128
  N0 = 512

  M1 = 4
  N1 = 4
  K1 = 8

  K2 = K // (K1 * K0)
  M2 = M // (M1 * M0)
  N2 = N // (N1 * N0)

  assert K2 * K1 * K0 == K
  assert M2 * M1 * M0 == M
  assert N2 * N1 * N0 == N

  A = np.random.random_sample([K, M]).astype(np.float16)
  B = np.random.random_sample([K, N]).astype(np.float16)
  O = np.ndarray(shape=[M, N], dtype=np.float16)

  matmul(A, B, O, K2, K1, K0, M2, M1, M0, N2, N1, N0)

  return A, B, O

def main():
  A, B, O = launch()
  print(O[0, 0])

if __name__ == "__main__":
  main()
