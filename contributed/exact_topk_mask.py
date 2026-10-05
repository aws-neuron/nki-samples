# SPDX-License-Identifier: MIT-0
"""
Exact top-k selection mask (radix select) as one NKI kernel, for NeuronCore-v2 (trn1/inf2) and later.

Contributed from the Kiln project (https://github.com/foxl-ai/kiln), where it selects the keys of DeepSeek-style
sparse attention (the indexer's top-k) on trn1.

WARNING: This kernel:
   - Was validated on trn1 (NeuronCore-v2) with Neuron SDK 2.32 / NKI 0.6.0 only
   - Has not been tested across all input configurations
   - Carries no compatibility guarantees
   - May change without prior notice

What it selects, per row of P float32 scores: the `keep` largest, with a deterministic tie rule. Every selected
score is >= every unselected one, and among the scores equal to the keep-th largest value t, the ones with the
lowest indices are selected (the same set as a stable descending sort). The output is an additive mask, 0.0 where
selected and -1e30 elsewhere, ready to add to attention logits. With vis_only=1, scores <= -5e29 (already masked
candidates) are never selected, so a row with fewer than `keep` visible scores selects exactly its visible ones.
Comparisons are IEEE float32 (-0.0 == +0.0); scores must not be NaN; the vector engine may flush subnormals to zero.

How (no sort, no data-dependent control flow; 31 + ceil(log2(P + 1)) rounds of compare-and-count):
1. Sign: if count(s >= 0) < keep then t < 0: select on s2 = -s with k2 = P + 1 - keep (the keep-th largest of s is
   minus the k2-th largest of -s), else s2 = s, k2 = keep. Either way the k2-th largest of s2 is |t| >= 0.
2. Radix over the bit pattern of |t|: for a non-negative float32 the int32 bit pattern orders like the value, so the
   largest pattern x with count(s2 >= float(x)) >= k2 is |t|'s. It is built from bit 30 down in 31 rounds (candidate =
   prefix | 2^b as an integer OR, its count by one compare-and-sum), so the threshold is never rounded: t = +-float(x).
3. Ties: room = keep - count(s > t) >= 1, and of the tied scores (s == t) the first `room` by index: the largest
   lim with count(tied and j < lim) < room, by binary lifting over the position, then select tied and j <= lim.
4. Out: 0.0 for (s > t) or (tied and j <= lim), -1e30 elsewhere.

Layout: each row is cut into g pieces of W = P / g scores, one piece per partition (g a power of two, chosen so that
rows x g fills up to 128 partitions: 8 rows of 2112 scores use 16 pieces of 132), so a round's compare-and-sum reads
W values per partition. A row's count is the sum of its g partial counts, one fp32 matmul with a block-diagonal 0/1
matrix (exact, counts < 2^24) that leaves every piece its row's total. Rows beyond 128 / g are processed in tiles of
128 partitions, `tiles` tiles at a time with their instructions interleaved.

IO: sc float32 [R * g, W] = scores [R, P] viewed as pieces (see topk_mask_inputs()); returns float32 [R * g, W],
the mask in the same layout (reshape to [R, P]). keep < P; W <= 4096.

Device time on one trn1 NeuronCore (Neuron SDK 2.32, NKI 0.6.0, neuron-bench nc_latency p50 of the kernel's NEFF):
8 x 2112 keep 512: 0.079 ms; 256 x 2112 keep 512: 0.279 ms; 8 x 8448 keep 2048: 0.103 ms; 64 x 8192 keep 2048: 0.297 ms.
"""
import argparse
import os

import numpy as np

import nki
import nki.isa as nisa
import nki.language as nl

NEG_INF = -1e30
VISIBLE = -5e29  # with vis_only, scores at or below this are never selected
BIG = float(2 ** 20)  # position of an untied score in the tie search (above any limit, exact in fp32)
MAX_W = 4096  # scores per partition
F32, I32 = nl.float32, nl.int32
VE = nisa.vector_engine


def _sb(shape, dtype=None):
    return nl.ndarray(shape, dtype=dtype or F32, buffer=nl.sbuf)


def _total(cp, n, g, G):
    """A row's count on every one of its g pieces: [n, 1] partial counts -> [n, 1] totals (PSUM)."""
    if g == 1:
        return cp
    ps = nl.ndarray((n, 1), dtype=F32, buffer=nl.psum)
    nisa.nc_matmul(dst=ps, stationary=G[0:n, 0:n], moving=cp, accumulate=False)
    return ps


def _consts(W: int, g: int, lg: int):
    """Position within the piece (every partition); with g > 1 the block-diagonal 0/1 matrix G[k, m] = [k // g ==
    m // g] and each partition's piece offset (p % g) W; 2^b as int32 bit patterns, bits[30 - b]."""
    wio_i = _sb((128, W), I32)
    nisa.iota(dst=wio_i, pattern=[[1, W]], offset=0, channel_multiplier=0)
    wio = _sb((128, W))
    nisa.tensor_copy(dst=wio, src=wio_i, engine=VE)
    po = _sb((128, 1))
    G = None
    if g > 1:
        ipi = _sb((128, 1), I32)
        nisa.iota(dst=ipi, pattern=[[0, 1]], offset=0, channel_multiplier=1)
        kq_i = _sb((128, 1), I32)
        nisa.tensor_scalar(dst=kq_i, data=ipi, op0=nl.right_shift, operand0=lg, engine=VE)
        kq = _sb((128, 1))
        nisa.tensor_copy(dst=kq, src=kq_i, engine=VE)
        mq_i = _sb((128, 128), I32)
        nisa.iota(dst=mq_i, pattern=[[1, 128]], offset=0, channel_multiplier=0)
        nisa.tensor_scalar(dst=mq_i, data=mq_i, op0=nl.right_shift, operand0=lg, engine=VE)
        mq = _sb((128, 128))
        nisa.tensor_copy(dst=mq, src=mq_i, engine=VE)
        G = _sb((128, 128))
        nisa.tensor_scalar(dst=G, data=mq, op0=nl.equal, operand0=kq, engine=VE)
        po_i = _sb((128, 1), I32)
        nisa.tensor_scalar(dst=po_i, data=ipi, op0=nl.bitwise_and, operand0=g - 1, engine=VE)
        nisa.tensor_scalar(dst=po, data=po_i, op0=nl.multiply, operand0=float(W), engine=VE)
    else:
        nisa.memset(dst=po, value=0.0)
    top = _sb((128, 1), I32)
    nisa.iota(dst=top, pattern=[[0, 1]], offset=1 << 30, channel_multiplier=0)
    bits = [top]
    for b in range(29, -1, -1):
        bt = _sb((128, 1), I32)
        nisa.tensor_scalar(dst=bt, data=top, op0=nl.right_shift, operand0=30 - b, engine=VE)
        bits.append(bt)
    return wio, po, G, bits


def _select(S, R0, NS, W: int, g: int, G, po, wio, bits, keep: int, nbits: int, vis_only: int, out):
    """Steps 1-4 of the module docstring for a group of row tiles whose scores S[i] [NS[i], W] are in SBUF
    (partition rows R0[i]..): their mask written to out."""
    P = g * W
    nt = len(NS)
    S2, SG, K2, PRE, SCR = [], [], [], [], []
    for i in range(nt):
        SCR.append(_sb((NS[i], W)))
    # 1. sign
    for i in range(nt):
        n = NS[i]
        cp = _sb((n, 1))
        nisa.tensor_scalar_reduce(dst=SCR[i], data=S[i], op0=nl.greater_equal, operand0=0.0,
                                  reduce_op=nl.add, reduce_res=cp)
        tot = _total(cp, n, g, G)
        f = _sb((n, 1))
        nisa.tensor_scalar(dst=f, data=tot, op0=nl.less, operand0=float(keep), engine=VE)
        sg = _sb((n, 1))
        nisa.tensor_scalar(dst=sg, data=f, op0=nl.multiply, operand0=-2.0, op1=nl.add, operand1=1.0, engine=VE)
        SG.append(sg)
        k2 = _sb((n, 1))
        nisa.tensor_scalar(dst=k2, data=f, op0=nl.multiply, operand0=float(P + 1 - 2 * keep), op1=nl.add,
                           operand1=float(keep), engine=VE)
        K2.append(k2)
        s2 = _sb((n, W))
        nisa.tensor_scalar(dst=s2, data=S[i], op0=nl.multiply, operand0=sg, engine=VE)
        S2.append(s2)
        pre = _sb((n, 1), I32)
        nisa.memset(dst=pre, value=0)
        PRE.append(pre)
    # 2. radix over the bit pattern of |t|
    for b in range(30, -1, -1):
        for i in range(nt):
            n = NS[i]
            cand = _sb((n, 1), I32)
            nisa.tensor_tensor(dst=cand, data1=PRE[i], data2=bits[30 - b][0:n], op=nl.bitwise_or, engine=VE)
            cp = _sb((n, 1))
            nisa.tensor_scalar_reduce(dst=SCR[i], data=S2[i], op0=nl.greater_equal, operand0=cand.view(F32),
                                      reduce_op=nl.add, reduce_res=cp)
            tot = _total(cp, n, g, G)
            inc = _sb((n, 1), I32)
            nisa.tensor_scalar(dst=inc, data=tot, op0=nl.greater_equal, operand0=K2[i], op1=nl.multiply,
                               operand1=float(1 << b), engine=VE)
            nisa.tensor_tensor(dst=PRE[i], data1=PRE[i], data2=inc, op=nl.bitwise_or, engine=VE)
    # 3. above, room, ties
    AB, ROOM, TJ, LIM = [], [], [], []
    for i in range(nt):
        n = NS[i]
        th = _sb((n, 1))
        nisa.tensor_scalar(dst=th, data=PRE[i].view(F32), op0=nl.multiply, operand0=SG[i], engine=VE)
        ab = _sb((n, W))
        abp = _sb((n, 1))
        nisa.tensor_scalar_reduce(dst=ab, data=S[i], op0=nl.greater, operand0=th, reduce_op=nl.add, reduce_res=abp)
        AB.append(ab)
        abt = _total(abp, n, g, G)
        room = _sb((n, 1))
        nisa.tensor_scalar(dst=room, data=abt, op0=nl.multiply, operand0=-1.0, op1=nl.add, operand1=float(keep),
                           engine=VE)
        ROOM.append(room)
        tied = _sb((n, W))
        nisa.tensor_scalar(dst=tied, data=S[i], op0=nl.equal, operand0=th, engine=VE)
        tj = _sb((n, W))  # w where tied, w + BIG elsewhere
        nisa.scalar_tensor_tensor(dst=tj, data=tied, op0=nl.multiply, operand0=-BIG, op1=nl.add, operand1=wio[0:n])
        nisa.tensor_scalar(dst=tj, data=tj, op0=nl.add, operand0=BIG, engine=VE)
        TJ.append(tj)
        lim = _sb((n, 1))
        nisa.memset(dst=lim, value=0.0)
        LIM.append(lim)
    for b in range(nbits - 1, -1, -1):
        for i in range(nt):
            n = NS[i]
            thr = _sb((n, 1))  # lim + 2^b in this piece's positions
            nisa.tensor_scalar(dst=thr, data=LIM[i], op0=nl.add, operand0=float(1 << b), op1=nl.subtract,
                               operand1=po[0:n], engine=VE)
            cp = _sb((n, 1))
            nisa.tensor_scalar_reduce(dst=SCR[i], data=TJ[i], op0=nl.less, operand0=thr, reduce_op=nl.add,
                                      reduce_res=cp)
            tot = _total(cp, n, g, G)
            inc = _sb((n, 1))
            nisa.tensor_scalar(dst=inc, data=tot, op0=nl.less, operand0=ROOM[i], op1=nl.multiply,
                               operand1=float(1 << b), engine=VE)
            nisa.tensor_tensor(dst=LIM[i], data1=LIM[i], data2=inc, op=nl.add, engine=VE)
    # 4. out
    for i in range(nt):
        r0 = R0[i]
        n = NS[i]
        lp = _sb((n, 1))
        nisa.tensor_tensor(dst=lp, data1=LIM[i], data2=po[0:n], op=nl.subtract, engine=VE)
        sel = _sb((n, W))
        nisa.scalar_tensor_tensor(dst=sel, data=TJ[i], op0=nl.less_equal, operand0=lp, op1=nl.add, operand1=AB[i])
        if vis_only:
            vm = _sb((n, W))
            nisa.tensor_scalar(dst=vm, data=S[i], op0=nl.greater, operand0=VISIBLE, engine=VE)
            nisa.tensor_tensor(dst=sel, data1=sel, data2=vm, op=nl.multiply, engine=VE)
        o = _sb((n, W))
        nisa.tensor_scalar(dst=o, data=sel, op0=nl.subtract, operand0=1.0, op1=nl.multiply, operand1=-NEG_INF,
                           engine=VE)
        nisa.dma_copy(dst=out.ap(pattern=[[W, n], [1, W]], offset=r0 * W), src=o)


@nki.jit
def topk_mask_fwd(sc, keep: int, g: int, lg: int, nbits: int, vis_only: int = 0, tiles: int = 2):
    """Exact top-`keep` mask of each row (module docstring). sc float32 [N, W] = scores [N / g, g W] as pieces,
    keep < g W, g = 2 ** lg, 2 ** nbits > g W. Returns float32 [N, W]: 0.0 selected, -1e30 not."""
    N, W = sc.shape
    out = nl.ndarray((N, W), dtype=F32, buffer=nl.shared_hbm)
    wio, po, G, bits = _consts(W, g, lg)
    NT = (N + 127) // 128
    # With an SPMD grid of 2 (LNC=2 on trn2), groups of row tiles alternate between the two programs.
    npg, pid = (nl.num_programs(axes=0), nl.program_id(axis=0)) if nl.program_ndim() != 0 else (1, 0)
    for t0 in range(0, NT, tiles):
        if (t0 // tiles) % npg != pid:
            continue
        R0, NS, S = [], [], []
        for t in range(t0, min(t0 + tiles, NT)):
            R0.append(t * 128)
            NS.append(min(128, N - t * 128))
        for i in range(len(NS)):
            s = _sb((NS[i], W))
            nisa.dma_copy(dst=s, src=sc.ap(pattern=[[W, NS[i]], [1, W]], offset=R0[i] * W))
            S.append(s)
        _select(S, R0, NS, W, g, G, po, wio, bits, keep, nbits, vis_only, out)
    if npg > 1:
        nisa.core_barrier(data=out, cores=(0, 1))
    return out


# ---------------------------------------------------------------------------------------------------
# Host helpers, NumPy reference and correctness check


def pieces(rows: int, n: int) -> int:
    """g: pieces per row (a power of two dividing n) so that rows x g fills up to 128 partitions."""
    g = 1
    while rows * g * 2 <= 128 and n % (g * 2) == 0:
        g *= 2
    return g


def topk_mask_inputs(scores: np.ndarray, keep: int, vis_only: bool = False) -> dict:
    """The kernel's arguments for scores [R, P] float32 (keep < P)."""
    R, P = scores.shape
    g = pieces(R, P)
    assert P // g <= MAX_W, f"{P // g} scores per partition (more than {MAX_W})"
    assert keep < P
    return dict(sc=np.ascontiguousarray(scores.astype(np.float32).reshape(R * g, P // g)), keep=int(keep), g=g,
                lg=g.bit_length() - 1, nbits=P.bit_length(), vis_only=int(vis_only))


def topk_mask(kernel, scores: np.ndarray, keep: int, vis_only: bool = False) -> np.ndarray:
    R, P = scores.shape
    return np.asarray(kernel(**topk_mask_inputs(scores, keep, vis_only))).reshape(R, P)


def reference_mask(scores: np.ndarray, keep: int, vis_only: bool = False) -> np.ndarray:
    """The same selection by a stable descending sort (ties: lowest index first)."""
    s = scores.astype(np.float32)
    R, P = s.shape
    order = np.argsort(-s, axis=1, kind="stable")
    sel = np.zeros((R, P), dtype=bool)
    np.put_along_axis(sel, order[:, :keep], True, axis=1)
    if vis_only:
        sel &= s > VISIBLE
    return np.where(sel, 0.0, NEG_INF).astype(np.float32)


def make_scores(kind: str, R: int, P: int, keep: int, rng) -> np.ndarray:
    if kind == "randn":
        return rng.standard_normal((R, P)).astype(np.float32)
    if kind == "ties":  # 16 distinct values: thousands of exact ties at the threshold
        return rng.integers(0, 16, size=(R, P)).astype(np.float32)
    if kind == "negative":  # the keep-th largest is negative (step 1's sign flip)
        return -np.abs(rng.standard_normal((R, P))).astype(np.float32) - 1.0
    if kind == "wide":  # values spread over many orders of magnitude, both signs
        return (rng.standard_normal((R, P)) * 10.0 ** rng.uniform(-30, 30, size=(R, P))).astype(np.float32)
    if kind == "zeros":  # mostly +-0.0: ties across the sign of zero
        s = np.where(rng.random((R, P)) < 0.5, 0.0, -0.0).astype(np.float32)
        s[:, ::7] = rng.standard_normal((R, P))[:, ::7]
        return s
    if kind == "masked":  # some rows have fewer than keep visible scores (vis_only)
        s = rng.standard_normal((R, P)).astype(np.float32)
        nvis = rng.integers(keep // 2, P, size=R)
        s[np.arange(P)[None, :] >= nvis[:, None]] = NEG_INF
        return s
    raise ValueError(kind)


SHAPES = [(8, 2112, 512), (256, 2112, 512), (8, 8448, 2048), (64, 8192, 2048)]
KINDS = ["randn", "ties", "negative", "wide", "zeros", "masked"]


def check_correct(backend="baremetal", shapes=SHAPES, kinds=KINDS):
    kernel = nki.simulate(topk_mask_fwd) if backend == "simulate" else topk_mask_fwd
    rng = np.random.default_rng(0)
    ok = True
    for R, P, keep in shapes:
        for kind in kinds:
            vis = kind == "masked"
            sc = make_scores(kind, R, P, keep, rng)
            got = topk_mask(kernel, sc, keep, vis_only=vis)
            want = reference_mask(sc, keep, vis_only=vis)
            bad_rows = int((got != want).any(axis=1).sum())
            ok &= bad_rows == 0
            print(f"[check_correct] {backend:9s} rows={R} P={P} keep={keep} {kind:9s} rows differing from the "
                  f"stable-sort selection: {bad_rows}/{R}  {'PASS' if bad_rows == 0 else 'FAIL'}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["simulate", "baremetal"],
                    default="baremetal" if os.path.exists("/dev/neuron0") else "simulate")
    ap.add_argument("--small", action="store_true", help="the first shape only (a quick simulator run)")
    ap.add_argument("--shape", type=int, nargs=3, metavar=("ROWS", "P", "KEEP"), default=None,
                    help="one shape with random scores only (one compiled kernel, e.g. to time its NEFF)")
    args = ap.parse_args()
    if args.shape:
        assert check_correct(args.backend, shapes=[tuple(args.shape)], kinds=["randn"])
        return
    shapes = SHAPES[:1] if args.small else SHAPES
    assert check_correct(args.backend, shapes=shapes)


if __name__ == "__main__":
    main()
