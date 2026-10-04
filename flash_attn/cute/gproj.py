"""Host dispatch for the opt-in, fixed-geometry Hopper formed-operand path.

No diagnostic tensors or events are allocated unless explicitly requested with
the private acceptance context below. Production uses five CuTe kernels and
the unchanged two native dK/dV zero initializers.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

import torch
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32
from quack.compile_utils import make_fake_tensor as fake_tensor

from .cache_utils import get_jit_cache
from .cute_dsl_utils import (get_broadcast_dims, maybe_contiguous,
                             to_cute_tensor, validate_output_layout)
from .testing import is_fake_mode
from .gproj_bwd import FlashAttentionBackwardGProj
from .gproj_preprocess import GProjPreprocess
from .gproj_ops import FusedQZ, GProjTensors, OperandProbes, KeyCorrection

ABI = 1
_CACHES = {name: get_jit_cache("gproj_" + name)
           for name in ("pre", "main", "qz", "key")}


@dataclass(frozen=True)
class _Diagnostic:
    callback: object
    rows: tuple = ()
    keys: tuple = ()
    time_passes: bool = False


_DIAGNOSTIC = ContextVar("fa4_gproj_diagnostic", default=None)


@contextmanager
def _diagnostic(callback, *, rows=(), keys=(), time_passes=False):
    """Acceptance-only instrumentation; absent from ordinary training calls."""
    if bool(rows) != bool(keys) or len(rows) > 129 or len(keys) > 129:
        raise ValueError("Supply both bounded row/key selections, or neither")
    if len(set(rows)) != len(rows) or len(set(keys)) != len(keys):
        raise ValueError("Probe selections must be unique")
    token = _DIAGNOSTIC.set(_Diagnostic(callback, tuple(rows), tuple(keys), time_passes))
    try:
        yield
    finally:
        _DIAGNOSTIC.reset(token)


def backward(*args, diagnostic=None, **kwargs):
    # CUDA autograd may execute Python backward on a different thread. Forward
    # explicitly saves its acceptance observer; do not depend on thread-local
    # ContextVar inheritance at that boundary. Direct replays use their context.
    if diagnostic is None:
        return _backward(*args, **kwargs)
    token = _DIAGNOSTIC.set(diagnostic)
    try:
        return _backward(*args, **kwargs)
    finally:
        _DIAGNOSTIC.reset(token)


def specialization(q, k, v, dout):
    """Only ABI/broadcast and singleton block facts; no exact length/stride keys."""
    n = q.shape[1]
    return (ABI, *(get_broadcast_dims(t) for t in (q, k, v, dout)),
            n <= 64, n <= 48)


def _convert(x):
    if isinstance(x, torch.Tensor):
        return to_cute_tensor(x, assumed_align=16)
    if isinstance(x, (GProjTensors, OperandProbes)):
        return type(x)(*(_convert(v) for v in x))
    if isinstance(x, float):
        return Float32(x)
    return x


def _run(name, obj, args, key, *, descriptors=None, state=None):
    cache = _CACHES[name]
    if key not in cache:
        descriptors = descriptors if descriptors is not None else tuple(_convert(x) for x in args)
        cache[key] = cute.compile(obj, *descriptors,
            stream=cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi")
    if not is_fake_mode():
        debug = _DIAGNOSTIC.get()
        if debug is not None and debug.time_passes:
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            cache[key](*args)
            end.record()
            state.setdefault("events", {})[name] = (start, end)
        else:
            cache[key](*args)


def _aux_shapes():
    # Length, rounded rows and raw accumulator extent are runtime dimensions.
    n, qr, rawq, rawk = (cute.sym_int() for _ in range(4))
    bf = lambda heads: fake_tensor(cutlass.BFloat16, (1, n, heads, 256), divisibility=8)
    fpraw = lambda heads, extent: fake_tensor(Float32, (1, heads, extent), divisibility=4)
    return dict(q=bf(16), k=bf(4), dq=bf(16), dk=bf(4), z=bf(16),
                a=fpraw(16, rawq), b=fpraw(16, rawq), dka=fpraw(4, rawk),
                mass=fake_tensor(Float32, (16, qr), divisibility=4),
                rho=fake_tensor(Float32, (16, qr), divisibility=4),
                delta=fake_tensor(Float32, (1, 16, qr), divisibility=4),
                lse=fake_tensor(Float32, (1, 16, n), divisibility=1),
                log2lse=fake_tensor(Float32, (1, 16, qr), divisibility=4))


def _probe_shapes(n, present):
    if not present:
        return dict(rows=None, keys=None, sampleq=None, samplek=None, columns=None, lam=None)
    nr, nk = cute.sym_int(), cute.sym_int()
    return dict(rows=fake_tensor(Int32, (nr,), divisibility=1),
                keys=fake_tensor(Int32, (nk,), divisibility=1),
                sampleq=fake_tensor(Float32, (3, nr, 16, 256), divisibility=4),
                samplek=fake_tensor(Float32, (3, nk, 4, 256), divisibility=4),
                columns=fake_tensor(cutlass.BFloat16, (nk, 16, n), divisibility=1),
                lam=fake_tensor(Float32, (16, cute.sym_int()), divisibility=4))


def _key_only(state, key_pipeline, *, output=None, columns=None, sample=None):
    """Acceptance alternate key pass on identical retained pre-correction state."""
    if key_pipeline not in ("single_stage", "two_stage"):
        raise ValueError("Unknown key pipeline")
    q, k, z, log2lse, dka = (state[x] for x in ("q", "k", "z", "log2lse", "dka"))
    output = torch.empty_like(k) if output is None else output
    ids = state.get("ids_k")
    probed = columns is not None or sample is not None
    if probed and (columns is None or sample is None or ids is None):
        raise ValueError("Key probes require ids, columns and a sample buffer")
    args = (q, k, z, log2lse, dka, output, ids if probed else None, columns, sample)
    key = (*state["specialization"], key_pipeline, probed)
    desc = None
    if key not in _CACHES["key"]:
        f = _aux_shapes()
        p = _probe_shapes(f["q"].shape[1], probed)
        desc = (f["q"], f["k"], f["z"], f["log2lse"], f["dka"], f["dk"],
                p["keys"], p["columns"], p["samplek"])
    _run("key", KeyCorrection(2 if key_pipeline == "two_stage" else 1), args, key,
         descriptors=desc, state=state)
    return output


def _backward(q, k, v, out, dout, lse, *, key_pipeline, dq=None, dk=None, dv=None):
    from .interface import _bwd_postprocess_convert
    if not is_fake_mode() and not q.is_cuda:
        raise ValueError("GProj inputs must be on CUDA")
    q, k, v, out, dout, lse = [maybe_contiguous(x) for x in (q, k, v, out, dout, lse)]
    n, device = q.shape[1], q.device
    if out.shape != q.shape or dout.shape != q.shape or lse.shape != (1, 16, n):
        raise ValueError("Invalid saved GProj forward/backward shapes")
    if any(x.dtype != torch.bfloat16 or x.device != device for x in (out, dout)):
        raise ValueError("GProj requires BF16 O/dO on the input device")
    if lse.dtype != torch.float32 or lse.device != device:
        raise ValueError("GProj requires FP32 LSE on the input device")
    outputs = []
    for provided, source, name in ((dq, q, "dq"), (dk, k, "dk"), (dv, v, "dv")):
        if provided is None:
            provided = torch.empty_like(source)
        if provided.shape != source.shape or provided.dtype != source.dtype or provided.device != device:
            raise ValueError(f"Invalid GProj {name} output")
        validate_output_layout(provided, name, 16)
        outputs.append(provided)
    dq, dk, dv = outputs
    qr, kr = (n + 63) // 64 * 64, (n + 47) // 48 * 48
    fp = lambda *shape: torch.empty(shape, dtype=torch.float32, device=device)
    a, b = fp(1, 16, qr * 256), fp(1, 16, qr * 256)
    mass, rho = fp(16, qr), fp(16, qr)
    delta, log2lse = fp(1, 16, qr), fp(1, 16, qr)
    # Keep the native GQA dK/dV accumulation initializers exactly as before.
    dka = torch.zeros((1, 4, kr * 256), dtype=torch.float32, device=device)
    dva = torch.zeros_like(dka)
    z = torch.empty_like(q)
    debug = _DIAGNOSTIC.get()
    spec = specialization(q, k, v, dout)
    state = dict(q=q, k=k, z=z, log2lse=log2lse, dka=dka, a=a, b=b,
                 mass=mass, rho=rho, delta=delta, specialization=spec)
    probes = ids_r = ids_k = sampleq = samplek = cols2 = lam = None
    if debug is not None and debug.rows:
        if any(not 0 <= x < n for x in (*debug.rows, *debug.keys)):
            raise ValueError("Probe index outside the logical sequence")
        ids_r, ids_k = (torch.tensor(x, dtype=torch.int32, device=device)
                        for x in (debug.rows, debug.keys))
        nr, nk = len(debug.rows), len(debug.keys)
        p, g = (torch.zeros((nr, 16, n), dtype=torch.float32, device=device) for _ in range(2))
        r, t = (torch.zeros((nr, 16, n), dtype=q.dtype, device=device) for _ in range(2))
        cols1, cols2 = (torch.zeros((nk, 16, n), dtype=q.dtype, device=device) for _ in range(2))
        sampleq, samplek, lam = fp(3, nr, 16, 256), fp(3, nk, 4, 256), fp(16, qr)
        probes = OperandProbes(ids_r, ids_k, p, g, r, t, cols1)
        state.update(probes=probes, columns2=cols2, sampleq=sampleq,
                     samplek=samplek, lam=lam, ids_k=ids_k)
    prekey, mainkey = (ABI,), (*spec, probes is not None)
    needs_shapes = (prekey not in _CACHES["pre"] or mainkey not in _CACHES["main"]
                    or mainkey not in _CACHES["qz"])
    f = _aux_shapes() if needs_shapes else None
    pf = _probe_shapes(f["q"].shape[1], probes is not None) if f is not None else None
    preargs = (out, dout, delta, lse, log2lse, a, None, None, None, None, None, 1.0,
               None, None, b, mass, rho)
    predesc = None if f is None else (f["q"], f["q"], f["delta"], f["lse"], f["log2lse"], f["a"],
               None, None, None, None, None, Float32(1.0), None, None,
               f["b"], f["mass"], f["rho"])
    _run("pre", GProjPreprocess(cutlass.BFloat16, 256, 256, 64, hdim_multiple_of=64),
         preargs, prekey, descriptors=predesc, state=state)
    main = FlashAttentionBackwardGProj(cutlass.BFloat16, 256, 256, 4, True,
        tile_m=64, tile_n=48, Q_stage=2, dO_stage=1, PdS_stage=1,
        SdP_swapAB=False, dKV_swapAB=True, dQ_swapAB=False,
        AtomLayoutMSdP=1, AtomLayoutNdKV=1, AtomLayoutMdQ=1, num_threads=384)
    mainargs = (q, k, v, dout, log2lse, delta, a, dka, dva, 1 / 16,
                GProjTensors(b, mass, rho, probes))
    # Unlike static DLPack probe strides at N=1, these remain reusable when
    # rows/keys/N change. Probes use scalar accesses and unaligned outer strides.
    probe_desc = None
    if probes is not None and f is not None:
        shape = (pf["rows"].shape[0], 16, f["q"].shape[1])
        probe_desc = OperandProbes(pf["rows"], pf["keys"],
            fake_tensor(Float32, shape), fake_tensor(Float32, shape),
            fake_tensor(cutlass.BFloat16, shape), fake_tensor(cutlass.BFloat16, shape),
            pf["columns"])
    maindesc = None
    if mainkey not in _CACHES["main"]:
        maindesc = (*(_convert(x) for x in mainargs[:-1]),
                    GProjTensors(f["b"], f["mass"], f["rho"], probe_desc))
    _run("main", main, mainargs, mainkey, descriptors=maindesc, state=state)
    qargs = (a, b, mass, rho, q, dq, z, ids_r, sampleq, lam)
    qdesc = None if f is None else (f["a"], f["b"], f["mass"], f["rho"], f["q"], f["dq"], f["z"],
             pf["rows"], pf["sampleq"], pf["lam"])
    _run("qz", FusedQZ(), qargs, (*spec, probes is not None), descriptors=qdesc, state=state)
    _key_only(state, key_pipeline, output=dk, columns=cols2, sample=samplek)
    if debug is not None and debug.time_passes and not is_fake_mode():
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
    _bwd_postprocess_convert(dva, dv, 1.0, None, None, 90, cutlass.BFloat16,
                            256, 48, 256, 1, True, fake_mode=is_fake_mode(),
                            hdim_multiple_of=64)
    if debug is not None and debug.time_passes and not is_fake_mode():
        end.record()
        state.setdefault("events", {})["dv"] = (start, end)
    if debug is not None:
        debug.callback(state)
    return dq, dk, dv
