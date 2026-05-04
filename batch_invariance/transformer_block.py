"""
NKI Transformer Block — minimal pre-norm decoder block using the three
batch-invariant kernels (matmul, rmsnorm, attention).

Shape constraints imposed by the kernels:
  d_head == 128          (attention kernel: D_TILE hardcoded to 128)
  seq    % 128 == 0      (Q_TILE=128, M_TILE=128)
  seq    %  64 == 0      (KV_TILE=64 for nondet attention)
  d_model % 128 == 0     (K=d_model in Q/K/V projections, K_TILE=128)
  d_ffn   % 128 == 0     (K=d_ffn  in FFN-down projection, K_TILE=128)
  d_model <= 512         (matmul b_tile free-dim limit: N cols in SBUF)
  d_ffn   <= 512         (same limit for FFN-up b_tile)

Sensible demo values: seq=512, d_model=256, d_head=128, d_ffn=512
"""

import torch
from kernels.attention_batch_invariant import nki_attention_kernel_isa
from kernels.matmul_batch_invariant    import nki_matmul_kernel_isa
from kernels.rmsnorm_batch_invariant   import nki_rmsnorm_kernel_isa


def make_block_weights(d_model, d_head, d_ffn, dtype=torch.bfloat16):
    """
    Returns a dict of CPU tensors.  Move to device before passing to the block:
        weights = make_block_weights(...)
        weights = {k: v.to(device) for k, v in weights.items()}

    Weight shapes (designed for matmul(a, b) = a @ b via the NKI kernel):
      The NKI matmul kernel computes  result = a^T @ b  where a=[K,M], b=[K,N].
      The wrapper `matmul(a, b)` calls kernel(a.T, b) so it computes a @ b normally.
      Constraint: b.shape[0] must equal a.shape[1] — same as standard matmul.
    """
    scale = 0.02
    return {
        # Projections: [in_features, out_features] — same layout as nn.Linear.weight.T
        'wq': torch.randn(d_model, d_head,  dtype=dtype) * scale,
        'wk': torch.randn(d_model, d_head,  dtype=dtype) * scale,
        'wv': torch.randn(d_model, d_head,  dtype=dtype) * scale,
        'wo': torch.randn(d_head,  d_model, dtype=dtype) * scale,
        'w1': torch.randn(d_model, d_ffn,   dtype=dtype) * scale,
        'w2': torch.randn(d_ffn,   d_model, dtype=dtype) * scale,
        # RMSNorm gains
        'g_attn': torch.ones(d_model, dtype=dtype),
        'g_ffn':  torch.ones(d_model, dtype=dtype),
    }


def nki_transformer_block(x, weights, deterministic=True):
    """
    Pre-norm transformer block:
      x -> RMSNorm -> QKV proj -> Attention -> out proj -> residual
        -> RMSNorm -> FFN up -> ReLU -> FFN down -> residual

    Args:
        x:             [seq, d_model]  on XLA device
        weights:       dict from make_block_weights, on XLA device
        deterministic: passed to all three NKI kernels

    Returns:
        [seq, d_model] on XLA device
    """
    device = x.device

    # Move weights to device if needed (idempotent if already there)
    w = {k: v.to(device) for k, v in weights.items()}

    def mm(a, b):
        """a @ b via NKI matmul kernel.  a=[r,c], b=[c,n] -> [r,n]."""
        return nki_matmul_kernel_isa(a.T.contiguous(), b, deterministic=deterministic)

    def rms(a, g):
        return nki_rmsnorm_kernel_isa(a, g, deterministic=deterministic)

    def attn(q, k, v):
        return nki_attention_kernel_isa(q, k, v, deterministic=deterministic)

    # 1. Pre-attention RMSNorm
    x_norm = rms(x, w['g_attn'])                  # [seq, d_model]

    # 2. QKV projections  [seq, d_model] @ [d_model, d_head] -> [seq, d_head]
    q = mm(x_norm, w['wq'])
    k = mm(x_norm, w['wk'])
    v = mm(x_norm, w['wv'])

    # 3. Attention  [seq, d_head] -> [seq, d_head]
    attn_out = attn(q, k, v)

    # 4. Output projection + residual  [seq, d_head] @ [d_head, d_model] -> [seq, d_model]
    x = x + mm(attn_out, w['wo'])

    # 5. Pre-FFN RMSNorm
    x_norm = rms(x, w['g_ffn'])                   # [seq, d_model]

    # 6. FFN  [seq, d_model] @ [d_model, d_ffn] -> [seq, d_ffn] -> [seq, d_model]
    h = mm(x_norm, w['w1'])                        # [seq, d_ffn]
    h = torch.relu(h)                              # element-wise, stays on device
    x = x + mm(h, w['w2'])                         # [seq, d_model]

    return x
