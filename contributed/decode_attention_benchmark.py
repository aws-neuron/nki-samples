"""
Device latency for `decode_attention_gqa_fwd` on a NeuronDevice.

Two measurements, both on the assumption that decode attention is memory-bound:

  1. GQA isolation. Query heads fixed, KV heads halved. Each halving halves the
     K/V cache traffic, so if the kernel is bandwidth-limited the latency should
     halve with it.
  2. Length scaling. seqlen_kv swept with the head counts fixed. Latency should
     be linear in seqlen_kv plus a fixed launch cost.

Why this compile route:

    with nki_ir_context() as context:
        result   = ParserFrontend().compile(context, kernel, inputs=...)
        compiled = CompiledKernel.from_frontend(result, compile_opts)
        result   = compiled.benchmark(warmup=..., iterations=..., **tensors)

This mirrors `nki.framework.compiled.compile_to_bir`, which is what `@nki.jit`
itself runs. It matters that it is the PARSER frontend: the kernel is written
against nl/nisa, and `nki.compiler.kernel_builder.compile_kernel` is a separate
authoring API whose inputs are TileViews, so a kernel written for one frontend
cannot be compiled by the other. The compile happens once and `benchmark()`
replays the NEFF, which is the whole point: timing a plain `kernel(*args)` call
instead measures the compiler, because the standalone path re-runs the frontend
on every invocation (~1.5 s per call on inf2, against ~67 us of kernel time at
seqlen_kv=512).

Requires a NeuronDevice. The numeric checks in decode_attention.py do not, and
still run anywhere via nki.simulate; that is why this lives in its own file.

Run `python decode_attention_benchmark.py`. Add --csv for machine-readable rows.
"""
import argparse
import inspect
import os
import sys
import tempfile

import numpy as np

from decode_attention import decode_attention_gqa_fwd, _make_gqa_inputs

# Fallback only. _peak_hbm_bw() prefers the SDK's own constant so the "% peak"
# column cites the toolchain rather than a number copied into this file.
_PEAK_HBM_FALLBACK = 410e9          # bytes/s, one NeuronCore-v2

# Defaults for the sweeps. d=128 fills the partition axis; fp32 because
# NeuronCore-v2 needs an fp32 matmul destination (see BF16_SUPPORTED).
_HEAD_DIM = 128
_SWEEP_SEQLEN = 2048                # length held fixed while KV heads vary
_SWEEP_Q_HEADS = 8
_SWEEP_KV_HEADS = (8, 4, 2, 1)
_LENGTH_KV_HEADS = 2                # head counts held fixed while length varies
_LENGTHS = (128, 512, 1024, 2048, 4096, 8192)


def _ir_context():
    """The context manager the compiler opens around a trace.

    Taken from compile_to_bir's globals rather than imported: compile_to_bir is
    the function actually defined against this name, so the two cannot drift
    apart in a later SDK. It is also re-exported from the kernel_builder module,
    but that is incidental to the builder API and not a contract.
    """
    from nki.framework.compiled import compile_to_bir

    factory = compile_to_bir.__globals__.get("nki_ir_context")
    if factory is None:
        raise RuntimeError(
            "nki_ir_context is not in compile_to_bir's globals; the compiler "
            "pipeline has changed shape and this harness needs updating")
    return factory


def _peak_hbm_bw(target="trn1"):
    """Peak HBM bandwidth per core, preferring the SDK's own constant.

    _PEAK_HBM_BW is keyed by target rather than being a single number, so pick
    the entry for the target we compile for. Returns the source alongside the
    value: the "% peak" column means something different if this number is
    assumed rather than read, and the caller prints which it was.
    """
    from nki.compiler.ncc_driver import CompiledKernel

    peak = getattr(CompiledKernel, "_PEAK_HBM_BW", None)
    source = "SDK"
    if isinstance(peak, dict):
        match = next((v for k, v in peak.items()
                      if str(k).lower() == target.lower()), None)
        if match is None:
            return _PEAK_HBM_FALLBACK, f"assumed, no SDK entry for {target!r}"
        peak, source = match, f"SDK[{target}]"
    try:
        peak = float(peak)
    except (TypeError, ValueError):
        return _PEAK_HBM_FALLBACK, "assumed"
    # Guard the units: the column is bytes/s. Anything this small is GB/s.
    if 0 < peak < 1e6:
        peak *= 1e9
    return (peak, source) if peak > 0 else (_PEAK_HBM_FALLBACK, "assumed")


def hbm_bytes(meta):
    """Bytes crossing HBM for one decode step.

    The K and V caches are read in full (2 * n_kv * d * seqlen_kv elements),
    and Q comes in and the output goes out (2 * n_q * d). Nothing else is
    resident, so this is the traffic a bandwidth-bound kernel would be paying.
    """
    itemsize = np.dtype(meta["dtype"]).itemsize
    d, n = meta["d"], meta["seqlen_kv"]
    return itemsize * d * (2 * meta["n_kv_heads"] * n + 2 * meta["n_q_heads"])


def measure(seqlen_kv, n_q_heads, n_kv_heads, warmup, iters, peak_bw):
    """Compile once, benchmark on device, and check the result against NumPy.

    Everything stays inside the IR context: the MLIR module belongs to it, and
    from_frontend is the step that runs neuronx-cc to produce the NEFF.
    """
    from nki.compiler.frontend import ParserFrontend
    from nki.compiler.ncc_driver import CompiledKernel, CompileOptions

    args, ref, meta = _make_gqa_inputs(d=_HEAD_DIM, seqlen_kv=seqlen_kv,
                                       n_q_heads=n_q_heads,
                                       n_kv_heads=n_kv_heads)
    q, k, v = args[0], args[1], args[2]

    # compile() takes the kernel's arguments as a dict keyed by parameter name,
    # so that option names like `target` cannot collide with kernel arguments.
    # Deriving the names from the kernel keeps this correct if its signature moves.
    names = list(inspect.signature(decode_attention_gqa_fwd.func).parameters)
    inputs = dict(zip(names, args))

    # The compiler writes the NEFF and its intermediates here. A temp dir keeps
    # them off the working tree and is cleaned up once benchmark() has replayed.
    with tempfile.TemporaryDirectory(prefix="nki_decode_bench_") as workdir:
        opts = CompileOptions(target="trn1", artifacts_dir=workdir,
                              output_path=os.path.join(workdir, "kernel.neff"))
        with _ir_context()() as context:
            compiled = CompiledKernel.from_frontend(
                ParserFrontend().compile(
                    context,
                    decode_attention_gqa_fwd,
                    inputs=inputs,
                    target=opts.target,
                    lnc=opts.lnc,
                    artifacts_dir=opts.artifacts_dir,
                    output_names=None,
                    enable_device_dump=opts.enable_device_dump,
                    debug=opts.debug,
                    lower_dma_transpose=opts.lower_dma_transpose,
                ),
                opts,
            )
            # Scalars were specialized into the NEFF at compile time, so replay
            # passes tensors only.
            res = compiled.benchmark(warmup=warmup, iterations=iters,
                                     q=q, k=k, v=v)

    nbytes = hbm_bytes(meta)
    latency = getattr(res, "latency", None) or 0.0
    row = dict(meta,
               nbytes=nbytes,
               latency_us=latency * 1e6,
               std_us=(getattr(res, "latency_std", None) or 0.0) * 1e6,
               # The compiler's static cost model, computed without running
               # anything. Agreement with the measurement is the check that
               # these timings are the kernel and not harness overhead.
               est_us=(getattr(compiled, "total_time_ns", None) or 0) / 1e3,
               utilization=getattr(compiled, "estimated_utilization", None),
               gbs=(nbytes / latency / 1e9) if latency else 0.0,
               pct_peak=(100.0 * nbytes / latency / peak_bw) if latency else 0.0,
               max_diff=None)

    # benchmark() returns real outputs, so the timed path is also the checked
    # path. A fast wrong kernel is not worth reporting.
    for arr in (getattr(res, "outputs", None) or {}).values():
        candidate = np.asarray(arr)
        if candidate.shape == ref.shape:
            row["max_diff"] = float(
                np.abs(candidate.astype(np.float32) - ref).max())
    return row


_COLUMNS = "  {:>5}  {:>5}  {:>8}  {:>9}  {:>8}  {:>8}  {:>7}  {:>6}  {:>10}"
_HEADER = _COLUMNS.format("Hkv", "group", "MiB", "lat us", "std us",
                          "est us", "GB/s", "%peak", "max|diff|")


def print_table(title, rows, first_col):
    """One sweep as a fixed-width table. first_col names the swept variable."""
    print(f"\n{title}")
    print(_HEADER.replace("Hkv", f"{first_col:>3}", 1))
    print("  " + "-" * (len(_HEADER) - 2))
    for r in rows:
        key = r["n_kv_heads"] if first_col == "Hkv" else r["seqlen_kv"]
        diff = "-" if r["max_diff"] is None else f"{r['max_diff']:.2e}"
        print(_COLUMNS.format(key, r["group"], f"{r['nbytes'] / 1024**2:.2f}",
                              f"{r['latency_us']:.2f}", f"{r['std_us']:.2f}",
                              f"{r['est_us']:.1f}", f"{r['gbs']:.1f}",
                              f"{r['pct_peak']:.1f}", diff))


def print_utilization(row):
    """Compiler cost-model utilization, highest first: where the time goes."""
    util = row.get("utilization") or {}
    if not util:
        return
    print(f"\nEngine utilization, compiler static estimate "
          f"(seqlen_kv={row['seqlen_kv']}, Hkv={row['n_kv_heads']})")
    print("  {:<8}  {:>8}".format("engine", "% busy"))
    print("  " + "-" * 18)
    for name, pct in sorted(util.items(), key=lambda kv: -kv[1]):
        print("  {:<8}  {:>8}".format(name, f"{pct:.1f}"))


def print_csv(rows):
    print("\nseqlen_kv,n_q_heads,n_kv_heads,group,bytes,latency_us,std_us,"
          "est_us,gbs,pct_peak,max_diff")
    for r in rows:
        print(f"{r['seqlen_kv']},{r['n_q_heads']},{r['n_kv_heads']},"
              f"{r['group']},{r['nbytes']},{r['latency_us']:.3f},"
              f"{r['std_us']:.3f},{r['est_us']:.1f},{r['gbs']:.3f},"
              f"{r['pct_peak']:.3f},{r['max_diff']}")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Device latency for decode_attention_gqa_fwd.")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--csv", action="store_true",
                        help="also print machine-readable rows")
    parser.add_argument("--quick", action="store_true",
                        help="one config only, to check the device path")
    args = parser.parse_args(argv)

    peak_bw, peak_src = _peak_hbm_bw()

    def run(seqlen_kv, n_kv_heads):
        return measure(seqlen_kv, _SWEEP_Q_HEADS, n_kv_heads,
                       args.warmup, args.iters, peak_bw)

    print(f"decode_attention_gqa_fwd on NeuronCore-v2, fp32, d={_HEAD_DIM}, "
          f"seqlen_q=1")
    print(f"{args.warmup} warmup + {args.iters} timed iterations per config, "
          f"compiled once per config")

    # Smoke test first: if the device path or the numerics are broken, say so
    # before spending compile time on the rest. This config sits in both sweeps,
    # so it is measured once and reused rather than compiled three times.
    first = run(_SWEEP_SEQLEN, _LENGTH_KV_HEADS)
    if first["max_diff"] is None:
        print("\nWARNING: benchmark() returned no comparable output; numerics "
              "are unverified on this path and timings are provisional.")
    elif first["max_diff"] > 1e-2:
        print(f"\nSTOP: output disagrees with the NumPy reference "
              f"(max|diff|={first['max_diff']:.2e}). Timings are moot.")
        return 1
    if first["latency_us"] <= 0:
        print("\nSTOP: benchmark() reported no latency.")
        return 1

    if args.quick:
        print_table(f"Single config: Hq={_SWEEP_Q_HEADS}, "
                    f"seqlen_kv={_SWEEP_SEQLEN}", [first], "Hkv")
        print_utilization(first)
        return 0

    gqa = [first if h == _LENGTH_KV_HEADS else run(_SWEEP_SEQLEN, h)
           for h in _SWEEP_KV_HEADS]
    print_table(f"GQA isolation: Hq={_SWEEP_Q_HEADS}, seqlen_kv={_SWEEP_SEQLEN}, "
                f"KV heads halved", gqa, "Hkv")

    lengths = [first if n == _SWEEP_SEQLEN else run(n, _LENGTH_KV_HEADS)
               for n in _LENGTHS]
    print_table(f"Length scaling: Hq={_SWEEP_Q_HEADS}, "
                f"Hkv={_LENGTH_KV_HEADS}", lengths, "N")

    print_utilization(first)

    print(f"\n%peak is against {peak_bw / 1e9:.0f} GB/s per NeuronCore-v2 "
          f"({peak_src}), one core of the two on an inf2 chip.")
    print("est us is the compiler's static cost model, not a measurement.")

    if args.csv:
        print_csv(gqa + lengths)
    return 0


if __name__ == "__main__":
    sys.exit(main())
