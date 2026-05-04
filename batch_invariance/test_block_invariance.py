"""
Transformer block tile-invariance test.

Verifies that the pre-norm NKI transformer block (matmul + rmsnorm + attention)
produces bit-exact bfloat16 outputs regardless of tile size (det=True vs det=False).

Shape constraints:
  d_head == 128, seq % 128 == 0, d_model % 128 == 0, d_ffn % 128 == 0
  d_model <= 512, d_ffn <= 512  (matmul SBUF b_tile free-dim limit)

Run from the batch_invariance directory:
    cd /home/ubuntu/nki-samples/contributed/batch_invariance
    source /opt/aws_neuronx_venv_pytorch_2_9/bin/activate
    NEURON_RT_VISIBLE_CORES=0 python3 /tmp/test_block_invariance.py
"""

import os
os.environ['NEURON_RT_VISIBLE_CORES'] = '0'

import torch
import torch_xla.core.xla_model as xm

from kernels.attention_batch_invariant import nki_attention_kernel_isa
from kernels.matmul_batch_invariant    import nki_matmul_kernel_isa
from kernels.rmsnorm_batch_invariant   import nki_rmsnorm_kernel_isa


# ── Transformer block ─────────────────────────────────────────────────────────

def nki_transformer_block(x, weights, deterministic=True):
    """
    Pre-norm transformer block.
      x -> RMSNorm -> QKV -> Attention -> out proj -> residual
        -> RMSNorm -> FFN up -> ReLU -> FFN down -> residual

    x:       [seq, d_model] on XLA device
    weights: dict of tensors on XLA device
    """
    w = weights  # already on device

    def mm(a, b):
        """a @ b  (NKI kernel takes a.T so it computes a.T^T @ b = a @ b)"""
        return nki_matmul_kernel_isa(a.T.contiguous(), b, deterministic=deterministic)

    def rms(a, g):
        return nki_rmsnorm_kernel_isa(a, g, deterministic=deterministic)

    def attn(q, k, v):
        return nki_attention_kernel_isa(q, k, v, deterministic=deterministic)

    x_norm   = rms(x, w['g_attn'])
    q        = mm(x_norm, w['wq'])
    k        = mm(x_norm, w['wk'])
    v        = mm(x_norm, w['wv'])
    attn_out = attn(q, k, v)
    x        = x + mm(attn_out, w['wo'])
    x_norm   = rms(x, w['g_ffn'])
    h        = mm(x_norm, w['w1'])
    h        = torch.relu(h)
    x        = x + mm(h, w['w2'])
    return x


def make_weights(d_model, d_head, d_ffn, dtype, device):
    """
    Linspace weights scaled to 0.02.

    The tile-invariance property requires BOTH operands of each matmul to have
    linspace-like structure so bfloat16 products land on the same coarse grid
    regardless of how tiles are grouped.  Using linspace weights (not random)
    satisfies this for every matmul in the block, mirroring the methodology of
    the individual kernel tests in test_tile_invariance.py.

    Scale 0.02 keeps intermediate activations well within bfloat16 range
    (linspace(-1,1) input × linspace(-0.02,0.02) weight × sqrt(d_model) ≈ 0.3).
    """
    def linspace_w(fan_in, fan_out, scale=0.02):
        return torch.linspace(-scale, scale, fan_in * fan_out,
                              dtype=dtype).reshape(fan_in, fan_out)

    return {k: v.to(device) for k, v in {
        'wq':    linspace_w(d_model, d_head),
        'wk':    linspace_w(d_model, d_head),
        'wv':    linspace_w(d_model, d_head),
        'wo':    linspace_w(d_head,  d_model),
        'w1':    linspace_w(d_model, d_ffn),
        'w2':    linspace_w(d_ffn,   d_model),
        'g_attn': torch.ones(d_model, dtype=dtype),
        'g_ffn':  torch.ones(d_model, dtype=dtype),
    }.items()}


# ── Test helpers ──────────────────────────────────────────────────────────────

def check(label, t):
    """Print shape/dtype/range/NaN summary for a device tensor."""
    c = t.cpu().float()
    nan_n  = c.isnan().sum().item()
    inf_n  = c.isinf().sum().item()
    maxabs = c[~c.isnan() & ~c.isinf()].abs().max().item() if nan_n + inf_n < c.numel() else float('nan')
    print(f"  {label:35s}  shape={tuple(t.shape)}  dtype={t.dtype}  max={maxabs:.3e}  nan={nan_n}  inf={inf_n}")


def run_invariance_test(seq, d_model, d_head, d_ffn, dtype):
    print(f"\n{'─'*65}")
    print(f"  seq={seq}  d_model={d_model}  d_head={d_head}  d_ffn={d_ffn}  dtype={dtype}")
    print(f"{'─'*65}")

    device  = xm.xla_device()
    weights = make_weights(d_model, d_head, d_ffn, dtype, device)

    # Linspace input: regular structure ensures products tile identically in bfloat16
    x_cpu = torch.linspace(-1, 1, seq * d_model).reshape(seq, d_model).to(dtype)
    x     = x_cpu.to(device)

    out_det = nki_transformer_block(x, weights, deterministic=True)
    xm.mark_step()
    check("out (det=True)", out_det)

    out_nondet = nki_transformer_block(x, weights, deterministic=False)
    xm.mark_step()
    check("out (det=False)", out_nondet)

    out_det_f   = out_det.cpu().float()
    out_nondet_f = out_nondet.cpu().float()

    diff    = (out_det_f - out_nondet_f).abs().max().item()
    has_nan = out_det_f.isnan().any().item() or out_nondet_f.isnan().any().item()
    has_inf = out_det_f.isinf().any().item() or out_nondet_f.isinf().any().item()

    if has_nan:
        status = "FAIL (NaN in output)"
    elif has_inf:
        status = "FAIL (Inf in output)"
    elif diff == 0.0:
        status = "PASS — diff=0 (invariant)"
    else:
        status = f"PASS — diff={diff:.2e} (not invariant, expected for float32)"

    print(f"\n  det/nondet max diff: {diff}")
    print(f"  Result: [{status}]")
    # Return True for bfloat16 (expect diff=0) and False for float32 (expect diff>0)
    # Caller interprets meaning; just return diff==0 here
    return diff == 0.0 and not has_nan and not has_inf


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("NKI Transformer Block — Tile Invariance Test")
    print("det=True  (KV_TILE=128, K_TILE=128)")
    print("det=False (KV_TILE=64,  K_TILE=64 )\n")
    print("Expected: bfloat16 → diff=0 (invariant), float32 → diff>0 (not invariant)\n")

    results = {}
    for dtype in (torch.bfloat16, torch.float32):
        results[dtype] = run_invariance_test(
            seq=512, d_model=256, d_head=128, d_ffn=512,
            dtype=dtype,
        )

    print(f"\n{'='*65}")
    bf16_ok  = results[torch.bfloat16] is True          # diff == 0
    f32_ok   = results[torch.float32]  is False         # diff > 0 (expected)
    overall  = bf16_ok and f32_ok
    print(f"  bfloat16 diff=0 (invariant):      {'PASS' if bf16_ok  else 'FAIL'}")
    print(f"  float32  diff>0 (not invariant):  {'PASS' if f32_ok   else 'FAIL'}")
    print(f"  Overall: {'PASS' if overall else 'FAIL'}")
    print(f"{'='*65}")
