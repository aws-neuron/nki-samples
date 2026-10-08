# SPDX-License-Identifier: MIT-0
"""
Chunked gated delta rule (prefill of Gated DeltaNet and Kimi Delta Attention layers) as one NKI kernel.

Contributed from the Kiln project (https://github.com/foxl-ai/kiln), where it is the default prefill
path for Gated DeltaNet (GDN) and Kimi Delta Attention (KDA) layers on trn1.

WARNING: This kernel:
   - Was validated on trn1 (NeuronCore-v2) with Neuron SDK 2.32 / NKI 0.6.0 only
   - Has not been tested across all input configurations
   - Carries no compatibility guarantees
   - May change without prior notice

What it computes, for every v head h, from an initial state S [Dk, Dv]:

    S_t = diag(exp(g_t)) S_{t-1} + k_t (beta_t (v_t - (diag(exp(g_t)) S_{t-1})^T k_t))^T
    o_t = S_t^T q_t

with per-channel log decays g [T, Hv, Dk] (KDA, kind=1) or one log decay per token and head
g [T, Hv] (GDN, kind=0). It returns every o_t and the final state. Dk = Dv = 128, Hv a multiple of Hk
(GQA: v head h reads k head h // (Hv // Hk)).

The algorithm is the chunked form of flash-linear-attention (fla, https://github.com/fla-org/flash-linear-attention,
MIT): the intra-chunk products A = beta k k^T and Aqk = q k^T under the decay (fla/ops/kda/chunk_intra.py), the
UT transform T = (I + A)^-1 with w = T (beta k exp(G)) and u = T (beta v) (fla/ops/kda/wy_fast.py,
fla/ops/utils/solve_tril.py), the state recurrence (fla/ops/common/chunk_delta_h.py) and the output
(fla/ops/gla/chunk.py). No fla code is used; what is specific here is how each piece is laid onto the
128 x 128 tensor engine and the vector and scalar engines:

- Chunks of 128 tokens: every chunk matrix is one [128, 128] tile. All matmuls are float32 (an fp32 x fp32
  nc_matmul is exact to ~1.6e-7 relative on trn1), a transpose is a matmul with the identity, exp runs on the
  scalar engine.
- KDA's decayed products A_ij = sum_d (beta k_i)_d k_jd exp(G_id - G_jd) are matmuls over d after factoring
  exp(G_i - G_j) = exp(G_i - G_r) exp(G_r - G_j) through a reference row r. As in fla's safe-gate path, rows
  are cut into sub-chunks of 16 and sub-chunk I uses its first row r = 16 I as the reference for all columns
  j <= 16 I + 15. Inside the diagonal sub-chunk exp(G_r - G_j) <= exp(15 |lower_bound|), so the KDA path
  REQUIRES a lower bound on every log decay (a "safe gate": GLM-5.3-Flash and Kimi K3 use -5, so <= e^75 in
  fp32). KDA without such a bound (for example Kimi-Linear-48B) must not use this kernel.
- GDN's decay is a scalar per (token, head): the products are one matmul each and the decay is applied
  after, as exp(min(G_i - G_j, mask)).
- The UT transform is computed transposed, Y = T^T = (I + A^T)^-1: 8 x 8 diagonal blocks inverted by
  repeated squaring ((I + V)(I + V^2)(I + V^4), exact for nilpotent blocks of 8), then four merge levels
  Y += Y (V o Lo_s) Y (the [[A, 0], [C, D]] block-inverse merge), with block masks as constants:
  18 matmuls per chunk.
- The chunk is applied to the state as one affine map, S' = P S + Q with P = diag(exp(G_L)) - (k exp(G_L - G))^T w
  and Q = (k exp(G_L - G))^T u, and o = (q exp(G) - Aqk w) S + Aqk u. Every term that does not involve S is
  computed before S is known, so the sequential part per chunk is two matmul pairs.
- `units` (chunk, head) units are issued together, every step for each unit in turn, so the in-order engine
  queues hold independent work between dependent steps (2 was the best setting measured on trn1).

IO (all float32, in HBM):
  q, k  [T, Hk, 128]   q already scaled (for example by Dk ** -0.5); l2-normalised q and k are the usual inputs
  v     [T, Hv, 128]
  g     [T, Hv, 128] (KDA) or [T, Hv] (GDN): log decays (<= 0)
  beta  [T, Hv]
  s0    [Hv, 128, 128] initial state
  consts [128, 11, 128] from delta_rule_constants()
  returns o [T, Hv, 128] and the final state [Hv, 128, 128]
T must be a multiple of 128; pad with beta = g = 0 (padding leaves the state unchanged).

Device time on one trn1 NeuronCore (Neuron SDK 2.32, NKI 0.6.0, neuron-bench nc_latency p50 of the kernel's NEFF),
Hk = Hv = heads, Dk = Dv = 128:

    heads   T      KDA       GDN
    2       512    0.14 ms   0.12 ms
    2       8192   2.00 ms   1.62 ms
    8       2048   2.06 ms   1.63 ms
    8       8192   8.19 ms   6.45 ms
"""
import argparse
import os

import numpy as np

import nki
import nki.isa as nisa
import nki.language as nl

L = 128  # tokens per chunk (one tile)
R = 16  # rows per reference sub-chunk (KDA)
B8 = 8  # diagonal blocks of the inverse inverted by squaring
LEVELS = (8, 16, 32, 64)  # merge levels of the inverse
KIND_GDN, KIND_KDA = 0, 1
# Constant tiles (delta_rule_constants()): index of each in the [128, NCST, 128] block.
C_I, C_U, C_SU, C_BL, C_BU, C_LO, C_ONES, C_MZ = 0, 1, 2, 3, 4, 5, 9, 10
NCST = 11
NEG = -1.0e30
F32 = nl.float32


def delta_rule_constants() -> np.ndarray:
    """[128, NCST, 128] fp32, tile c at [:, c, :] (partition p, column q): I; U [p <= q]; SU [p > q];
    BL [same 8-block, p > q]; BU = BL^T; LO_s for s in LEVELS [same 2s-block, p in its lower half, q in its
    upper half]; ONES; MZ: 0 where p <= q, -1e30 elsewhere."""
    r = np.arange(L)
    p, q = r[:, None], r[None, :]
    tiles = [p == q, p <= q, p > q, (p // B8 == q // B8) & (p > q), (p // B8 == q // B8) & (p < q)]
    for s in LEVELS:
        tiles.append((p // (2 * s) == q // (2 * s)) & (p // s % 2 == 1) & (q // s % 2 == 0))
    tiles.append(np.ones((L, L), dtype=bool))
    out = np.stack([t.astype(np.float32) for t in tiles] + [np.where(p <= q, 0.0, NEG).astype(np.float32)], axis=1)
    assert out.shape == (L, NCST, L)
    return np.ascontiguousarray(out)


def _sb(shape):
    return nl.ndarray(shape, dtype=F32, buffer=nl.sbuf)


def _ps(shape):
    return nl.ndarray(shape, dtype=F32, buffer=nl.psum)


def _mm(dst, st, mv, acc=False):
    nisa.nc_matmul(dst=dst, stationary=st, moving=mv, accumulate=acc)


def _dve(dst, a, b, op):
    nisa.tensor_tensor(dst=dst, data1=a, data2=b, op=op, engine=nisa.vector_engine)


def _act_copy(dst, src, scale=1.0):
    nisa.activation(dst=dst, op=nl.copy, data=src, scale=scale)


def _inverse(NTs, C):
    """Y = T^T = (I - NT)^-1 for each unit's strictly upper NT, each step issued for every unit in turn."""
    n = len(NTs)
    Ns, Nb, NbT, NL = [], [], [], []
    for h in range(n):
        pn = _ps((128, 128))
        _mm(pn, NTs[h], C[:, C_I, :])  # N = NT^T
        ns = _sb((128, 128))
        _act_copy(ns, pn)
        Ns.append(ns)
    for h in range(n):
        nb = _sb((128, 128))
        _dve(nb, Ns[h], C[:, C_BL, :], nl.multiply)
        Nb.append(nb)
        nbt = _sb((128, 128))
        _dve(nbt, NTs[h], C[:, C_BU, :], nl.multiply)
        NbT.append(nbt)
        row = []
        for i in range(len(LEVELS)):
            nl_ = _sb((128, 128))
            _dve(nl_, Ns[h], C[:, C_LO + i, :], nl.multiply)
            row.append(nl_)
        NL.append(row)
    P2, P2T, P4, Y = [], [], [], []
    for h in range(n):
        pp = _ps((128, 2, 128))
        _mm(pp[:, 0, :], NbT[h], Nb[h])  # Nb @ Nb
        _mm(pp[:, 1, :], Nb[h], NbT[h])  # NbT @ NbT
        p2 = _sb((128, 128))
        nisa.tensor_copy(dst=p2, src=pp[:, 0, :], engine=nisa.vector_engine)
        p2t = _sb((128, 128))
        _act_copy(p2t, pp[:, 1, :])
        P2.append(p2)
        P2T.append(p2t)
        y1 = _sb((128, 128))
        _dve(y1, NbT[h], C[:, C_I, :], nl.add)
        Y.append(y1)
    for h in range(n):
        p4p = _ps((128, 128))
        _mm(p4p, P2T[h], P2[h])  # P2 @ P2
        p4 = _sb((128, 128))
        _act_copy(p4, p4p)
        P4.append(p4)
        py = _ps((128, 128))
        _mm(py, P2[h], Y[h])  # P2^T Y
        y2 = _sb((128, 128))
        _dve(y2, Y[h], py, nl.add)
        Y[h] = y2
    for h in range(n):
        py = _ps((128, 128))
        _mm(py, P4[h], Y[h])  # P4^T Y
        y3 = _sb((128, 128))
        _dve(y3, Y[h], py, nl.add)
        Y[h] = y3
    for i in range(len(LEVELS)):
        Xs, Rs = [], []
        for h in range(n):
            px = _ps((128, 128))
            _mm(px, Y[h], C[:, C_I, :])  # X = Y^T
            pr = _ps((128, 128))
            _mm(pr, NL[h][i], Y[h])  # (N o LO)^T Y
            xs = _sb((128, 128))
            _act_copy(xs, px)
            rs = _sb((128, 128))
            nisa.tensor_copy(dst=rs, src=pr, engine=nisa.vector_engine)
            Xs.append(xs)
            Rs.append(rs)
        for h in range(n):
            pw = _ps((128, 128))
            _mm(pw, Xs[h], Rs[h])  # Y R
            yn = _sb((128, 128))
            _dve(yn, Y[h], pw, nl.add)
            Y[h] = yn
    return Y


@nki.jit
def chunked_delta_rule_fwd(q, k, v, g, beta, s0, consts, kind: int, units: int = 2):
    """Chunked gated delta rule (see the module docstring for the math and the IO layout).

    kind: KIND_KDA (1, g [T, Hv, 128]) or KIND_GDN (0, g [T, Hv]).
    units: (chunk, v head) units whose instructions are interleaved.
    Returns o fp32 [T, Hv, 128] and the final state fp32 [Hv, 128, 128]."""
    T, Hk, D = q.shape
    Hv, Dv = v.shape[1], v.shape[2]
    rep = Hv // Hk
    NCH = T // 128
    o = nl.ndarray((T, Hv, Dv), dtype=F32, buffer=nl.shared_hbm)
    S_out = nl.ndarray((Hv, D, Dv), dtype=F32, buffer=nl.shared_hbm)
    C = _sb((128, NCST, 128))
    nisa.dma_copy(dst=C, src=consts)
    ident = C[:, C_I, :]
    U = C[:, C_U, :]
    # With an SPMD grid of 2 (LNC=2 on trn2) each program runs its own v heads h0 .. h0 + HP - 1 and writes
    # only their o columns and states. Grid 1: every head.
    npg, pid = (nl.num_programs(axes=0), nl.program_id(axis=0)) if nl.program_ndim() != 0 else (1, 0)
    HP = Hv // npg if Hv % npg == 0 else Hv
    h0 = pid * HP if Hv % npg == 0 else 0
    mine = Hv % npg == 0 or pid == 0  # heads that do not split evenly all run on program 0
    St = [None] * Hv
    for h in range(h0, h0 + HP):
        if mine:
            s_init = _sb((128, Dv))
            nisa.dma_copy(dst=s_init, src=s0.ap(pattern=[[Dv, 128], [1, Dv]], offset=h * D * Dv))
            St[h] = s_init
    # Units (chunk c, v head h) in chunk-major order, `units` at a time; the (o, S) chain of a head runs in
    # chunk order.
    NU = NCH * HP if mine else 0
    for u0 in range(0, NU, units):
        heads = []
        t0s = []
        for u in range(u0, min(u0 + units, NU)):
            heads.append(h0 + u % HP)
            t0s.append((u // HP) * 128)
        n = len(heads)
        # Loads (token-major tiles: rows are tokens).
        qt, kt, vt, bt, gt = [], [], [], [], []
        for j in range(n):
            h = heads[j]
            hk = h // rep
            a = _sb((128, D))
            nisa.dma_copy(dst=a, src=q.ap(pattern=[[Hk * D, 128], [1, D]], offset=t0s[j] * Hk * D + hk * D))
            qt.append(a)
            b = _sb((128, D))
            nisa.dma_copy(dst=b, src=k.ap(pattern=[[Hk * D, 128], [1, D]], offset=t0s[j] * Hk * D + hk * D))
            kt.append(b)
            vv = _sb((128, Dv))
            nisa.dma_copy(dst=vv, src=v.ap(pattern=[[Hv * Dv, 128], [1, Dv]], offset=t0s[j] * Hv * Dv + h * Dv))
            vt.append(vv)
            bb = _sb((128, 1))
            nisa.dma_copy(dst=bb, src=beta.ap(pattern=[[Hv, 128], [1, 1]], offset=t0s[j] * Hv + h))
            bt.append(bb)
            if kind == 1:
                gg = _sb((128, D))
                nisa.dma_copy(dst=gg, src=g.ap(pattern=[[Hv * D, 128], [1, D]], offset=t0s[j] * Hv * D + h * D))
            else:
                gg = _sb((128, 1))
                nisa.dma_copy(dst=gg, src=g.ap(pattern=[[Hv, 128], [1, 1]], offset=t0s[j] * Hv + h))
            gt.append(gg)
        # Token-major scaled operands; RHS = [beta v | beta k exp(G)].
        kb, RHS = [], []
        for j in range(n):
            a = _sb((128, D))
            nisa.tensor_scalar(dst=a, data=kt[j], op0=nl.multiply, operand0=bt[j], engine=nisa.vector_engine)
            kb.append(a)
            rh = _sb((128, 2, 128))
            _act_copy(rh[:, 0, :], vt[j], scale=bt[j])
            RHS.append(rh)
        # d-major transposes (and KDA's cumulative decay G^T = g^T U) on the tensor engine.
        qT, kT, kbT, GT = [], [], [], []
        for j in range(n):
            p1 = _ps((128, 4, 128))
            _mm(p1[:, 0, :], qt[j], ident)
            _mm(p1[:, 1, :], kt[j], ident)
            _mm(p1[:, 2, :], kb[j], ident)
            if kind == 1:
                _mm(p1[:, 3, :], gt[j], U)
            a = _sb((128, 128))
            _act_copy(a, p1[:, 0, :])
            qT.append(a)
            b = _sb((128, 128))
            nisa.tensor_copy(dst=b, src=p1[:, 1, :], engine=nisa.vector_engine)
            kT.append(b)
            cc = _sb((128, 128))
            _act_copy(cc, p1[:, 2, :])
            kbT.append(cc)
            if kind == 1:
                gg = _sb((128, 128))
                nisa.tensor_copy(dst=gg, src=p1[:, 3, :], engine=nisa.vector_engine)
                GT.append(gg)
        NT, QkT, qGT, kd, GL = [], [], [], [], []
        if kind == 1:
            for j in range(n):
                # Reference rows: EQ[:, i] = exp(G_i - G_r(i)); kg_I = k^T exp(G_r - G_j), j <= r + 15.
                ngr = _sb((128, 8))
                nisa.tensor_scalar(dst=ngr, data=GT[j].ap(pattern=[[128, 128], [16, 8]], offset=0),
                                   op0=nl.multiply, operand0=-1.0, engine=nisa.vector_engine)
                eq = _sb((128, 128))
                for b_ in range(8):
                    nisa.activation(dst=eq[:, b_ * 16:(b_ + 1) * 16], op=nl.exp,
                                    data=GT[j][:, b_ * 16:(b_ + 1) * 16], bias=ngr[:, b_:b_ + 1], scale=1.0)
                kq = _sb((128, 2, 128))  # [kb^T EQ | q^T EQ]
                _dve(kq[:, 0, :], kbT[j], eq, nl.multiply)
                _dve(kq[:, 1, :], qT[j], eq, nl.multiply)
                z = _ps((128, 2, 128))
                for b_ in range(8):
                    w_ = (b_ + 1) * 16
                    ek = _sb((128, 128))
                    nisa.activation(dst=ek[:, 0:w_], op=nl.exp, data=GT[j][:, 0:w_],
                                    bias=GT[j][:, b_ * 16:b_ * 16 + 1], scale=-1.0)
                    kg = _sb((128, 128))
                    if b_ < 7:
                        nisa.memset(dst=kg[:, w_:128], value=0.0, engine=nisa.gpsimd_engine)
                    _dve(kg[:, 0:w_], kT[j][:, 0:w_], ek[:, 0:w_], nl.multiply)
                    _mm(z[:, :, b_ * 16:(b_ + 1) * 16], kg, kq[:, :, b_ * 16:(b_ + 1) * 16])
                qk = _sb((128, 128))
                _dve(qk, z[:, 1, :], U, nl.multiply)
                QkT.append(qk)
                nt = _sb((128, 128))
                _act_copy(nt, z[:, 0, :], scale=-1.0)
                NT.append(nt)
            for j in range(n):
                gam = _sb((128, 128))
                nisa.activation(dst=gam, op=nl.exp, data=GT[j])
                qg = _sb((128, 128))
                _dve(qg, qT[j], gam, nl.multiply)
                qGT.append(qg)
                gl = _sb((128, 1))
                nisa.activation(dst=gl, op=nl.exp, data=GT[j][:, 127:128])
                GL.append(gl)
                p2 = _ps((128, 2, 128))
                _mm(p2[:, 0, :], U, gt[j])  # G token-major
                _mm(p2[:, 1, :], C[:, C_SU, :], gt[j])  # G_L - G
                eg = _sb((128, 2, 128))
                nisa.activation(dst=eg, op=nl.exp, data=p2)
                _dve(RHS[j][:, 1, :], kb[j], eg[:, 0, :], nl.multiply)
                kdd = _sb((128, 128))
                _dve(kdd, kt[j], eg[:, 1, :], nl.multiply)
                kd.append(kdd)
        else:
            for j in range(n):
                gu = _sb((128, 128))
                nisa.tensor_scalar(dst=gu, data=U, op0=nl.multiply, operand0=gt[j], engine=nisa.vector_engine)
                pe = _ps((128, 128))
                _mm(pe, C[:, C_ONES, :], gu)  # [j, i] = G_i
                pg = _ps((128, 1))
                _mm(pg, U, gt[j])  # G_t
                gc = _sb((128, 1))
                nisa.tensor_copy(dst=gc, src=pg, engine=nisa.vector_engine)
                em = _sb((128, 128))
                nisa.scalar_tensor_tensor(dst=em, data=pe, op0=nl.subtract, operand0=gc, op1=nl.minimum,
                                          operand1=C[:, C_MZ, :])
                dec = _sb((128, 128))
                nisa.activation(dst=dec, op=nl.exp, data=em)
                gb = _sb((128, 128))
                nisa.activation(dst=gb, op=nl.exp, data=pe)
                GL.append(gb[:, 127:128])
                gm = _sb((128, 1))
                nisa.activation(dst=gm, op=nl.exp, data=gc)
                nisa.tensor_scalar(dst=RHS[j][:, 1, :], data=kb[j], op0=nl.multiply, operand0=gm,
                                   engine=nisa.vector_engine)
                kdd = _sb((128, 128))
                nisa.tensor_scalar(dst=kdd, data=kt[j], op0=nl.multiply, operand0=dec[:, 127:128],
                                   engine=nisa.vector_engine)
                kd.append(kdd)
                kq = _sb((128, 2, 128))
                _act_copy(kq[:, 0, :], kbT[j])
                _act_copy(kq[:, 1, :], qT[j])
                z = _ps((128, 2, 128))
                _mm(z, kT[j], kq)  # [k_j . beta k_i | k_j . q_i]
                nt = _sb((128, 128))
                nisa.scalar_tensor_tensor(dst=nt, data=z[:, 0, :], op0=nl.multiply, operand0=-1.0,
                                          op1=nl.multiply, operand1=dec)
                NT.append(nt)
                qk = _sb((128, 128))
                _dve(qk, z[:, 1, :], dec, nl.multiply)
                QkT.append(qk)
                qg = _sb((128, 128))
                _dve(qg, qT[j], gb, nl.multiply)
                qGT.append(qg)
        Y = _inverse(NT, C)
        # u | w, then every term of the chunk's affine map that does not involve S.
        UW, NPT, MT = [], [], []
        for j in range(n):
            pu = _ps((128, 2, 128))
            _mm(pu, Y[j], RHS[j])  # T [beta v | beta k exp(G)]
            uw = _sb((128, 2, 128))
            _act_copy(uw, pu)
            UW.append(uw)
        for j in range(n):
            pn = _ps((128, 128))
            _mm(pn, UW[j][:, 1, :], kd[j])  # w^T kd [d', d]
            npt = _sb((128, 128))
            nisa.scalar_tensor_tensor(dst=npt, data=ident, op0=nl.multiply, operand0=GL[j], op1=nl.subtract,
                                      operand1=pn)
            NPT.append(npt)
            pm = _ps((128, 128))
            _mm(pm, UW[j][:, 1, :], QkT[j])  # w^T Aqk^T [d, i]
            mt = _sb((128, 128))
            _dve(mt, qGT[j], pm, nl.subtract)
            MT.append(mt)
        # The sequential part: o = M S + Aqk u, S' = P S + Q.
        for j in range(n):
            h = heads[j]
            po = _ps((128, Dv))
            _mm(po, QkT[j], UW[j][:, 0, :])
            _mm(po, MT[j], St[h], acc=True)
            ps_ = _ps((128, Dv))
            _mm(ps_, kd[j], UW[j][:, 0, :])
            _mm(ps_, NPT[j], St[h], acc=True)
            ob = _sb((128, Dv))
            _act_copy(ob, po)
            nisa.dma_copy(dst=o.ap(pattern=[[Hv * Dv, 128], [1, Dv]], offset=t0s[j] * Hv * Dv + h * Dv), src=ob)
            sn = _sb((128, Dv))
            nisa.tensor_copy(dst=sn, src=ps_, engine=nisa.vector_engine)
            St[h] = sn
    for h in range(h0, h0 + HP):
        if mine:
            nisa.dma_copy(dst=S_out.ap(pattern=[[Dv, 128], [1, Dv]], offset=h * D * Dv), src=St[h])
    if npg > 1:  # both programs' heads written before either ends
        nisa.core_barrier(data=o, cores=(0, 1))
        nisa.core_barrier(data=S_out, cores=(0, 1))
    return o, S_out


# ---------------------------------------------------------------------------------------------------
# NumPy reference and correctness check


def delta_rule_reference(q, k, v, g, beta, s0):
    """Token-by-token recurrence in float64. q, k [T, Hk, D]; v [T, Hv, Dv]; g [T, Hv, D] or [T, Hv];
    beta [T, Hv]; s0 [Hv, D, Dv]. Returns o [T, Hv, Dv] and the final state."""
    T, Hk, D = k.shape
    Hv = v.shape[1]
    rep = Hv // Hk
    q, k, v, g, beta = (x.astype(np.float64) for x in (q, k, v, g, beta))
    S = s0.astype(np.float64).copy()
    o = np.zeros((T, Hv, v.shape[2]))
    for t in range(T):
        for h in range(Hv):
            qt, kt = q[t, h // rep], k[t, h // rep]
            e = np.exp(g[t, h]) if g.ndim == 3 else np.full(D, np.exp(g[t, h]))
            S1 = e[:, None] * S[h]
            delta = beta[t, h] * (v[t, h] - S1.T @ kt)
            S[h] = S1 + np.outer(kt, delta)
            o[t, h] = S[h].T @ qt
    return o, S


def make_inputs(T, Hk, Hv, kind, seed=0, lower_bound=-5.0, D=128):
    """Inputs as a GDN / KDA mixer builds them: l2-normalised q (scaled by D ** -0.5) and k, beta in [0, 1),
    KDA gates lower_bound * sigmoid(.) (the safe gate), GDN gates -exp(A) softplus(.), a random state."""
    rng = np.random.default_rng(seed)

    def l2(x):
        return x / np.linalg.norm(x, axis=-1, keepdims=True)

    k = l2(rng.standard_normal((T, Hk, D))).astype(np.float32)
    beta = rng.random((T, Hv)).astype(np.float32)
    q = (l2(rng.standard_normal((T, Hk, D))) * D ** -0.5).astype(np.float32)
    v = rng.standard_normal((T, Hv, D)).astype(np.float32)
    if kind == KIND_KDA:
        g = lower_bound / (1.0 + np.exp(-(rng.standard_normal((T, Hv, D)) * 2)))
    else:
        g = -np.exp(rng.standard_normal(Hv))[None, :] * np.log1p(np.exp(rng.standard_normal((T, Hv))))
    s0 = (rng.standard_normal((Hv, D, D)) * 0.1).astype(np.float32)
    return q, k, v, g.astype(np.float32), beta, s0


def run(kernel, q, k, v, g, beta, s0, kind, units=2):
    return kernel(q, k, v, g, beta, s0, delta_rule_constants(), kind=kind, units=units)


def check_correct(backend="baremetal", T=512, Hk=2, Hv=4, kinds=(KIND_KDA, KIND_GDN), tol=1e-4):
    """Kernel output against the float64 recurrence, for KDA and GDN (GQA when Hv > Hk for GDN), from a non-zero
    state; reports max |error| relative to the max magnitude of o and of the final state."""
    kernel = nki.simulate(chunked_delta_rule_fwd) if backend == "simulate" else chunked_delta_rule_fwd
    ok = True
    for kind in kinds:
        name = "kda" if kind == KIND_KDA else "gdn"
        hk = Hv if kind == KIND_KDA else Hk
        q, k, v, g, beta, s0 = make_inputs(T, hk, Hv, kind, seed=T + kind)
        want_o, want_s = delta_rule_reference(q, k, v, g, beta, s0)
        got_o, got_s = run(kernel, q, k, v, g, beta, s0, kind)
        err_o = np.abs(np.asarray(got_o, dtype=np.float64) - want_o).max() / np.abs(want_o).max()
        err_s = np.abs(np.asarray(got_s, dtype=np.float64) - want_s).max() / np.abs(want_s).max()
        passed = err_o < tol and err_s < tol
        ok &= passed
        print(f"[check_correct] {backend:9s} {name} T={T} Hk={hk} Hv={Hv}  rel max|err| o={err_o:.2e} "
              f"S={err_s:.2e}  {'PASS' if passed else 'FAIL'}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["simulate", "baremetal"],
                    default="baremetal" if os.path.exists("/dev/neuron0") else "simulate")
    ap.add_argument("--seq", type=int, default=512, help="T, a multiple of 128")
    ap.add_argument("--kind", choices=["both", "kda", "gdn"], default="both")
    ap.add_argument("--heads", type=int, default=None,
                    help="Hv; with it Hk = Hv as well (one shape per kind, e.g. to time its NEFF). Default: "
                         "KDA Hk = Hv = 4, GDN Hk = 2, Hv = 4")
    args = ap.parse_args()
    kinds = {"both": (KIND_KDA, KIND_GDN), "kda": (KIND_KDA,), "gdn": (KIND_GDN,)}[args.kind]
    Hk, Hv = (args.heads, args.heads) if args.heads else (2, 4)
    assert check_correct(args.backend, T=args.seq, Hk=Hk, Hv=Hv, kinds=kinds)


if __name__ == "__main__":
    main()
