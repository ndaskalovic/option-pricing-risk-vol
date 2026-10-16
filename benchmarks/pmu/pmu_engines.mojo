"""Hardware counters (Apple Silicon PMU via kperf) for the engines. Needs root.

Build (from repo root), then run with sudo from a real terminal:
    clang -O2 -shared -install_name @rpath/libkpc_helper.dylib -o benchmarks/pmu/libkpc_helper.dylib benchmarks/pmu/kpc_helper.c
    .venv/bin/mojo build -I . benchmarks/pmu/pmu_engines.mojo -o benchmarks/pmu/pmu_engines \\
        -Xlinker -L$PWD/benchmarks/pmu -Xlinker -lkpc_helper -Xlinker -rpath -Xlinker $PWD/benchmarks/pmu
    sudo benchmarks/pmu/pmu_engines
Counts are per thread (other processes do not pollute instruction/uop counts).
"""
from std.memory.alloc import unsafe_alloc
from std.math import log
from std.ffi import external_call
from autodiff import Number, Tape
from engines import mc_compiled, fd_compiled, fd_price
from models.gbm import GbmCall
from models.bs_fd import BsAmericanPut
from vanilla_euro_call import EuropeanCallOption
import std.random

comptime NEV = 6
comptime NWORK = 4
comptime NS = 52
comptime W = 32


@no_inline
def work(w: Int, nP: Int, z: Pointer[Float32, _]) -> Float64:
    var p = Array[Float64, 5](fill=0)
    p[0] = 100; p[1] = 0.01; p[2] = 0.2; p[3] = 1.0; p[4] = 100
    if w == 0:
        var r = mc_compiled[GbmCall, W, DType.float32](p, NS, nP, z)
        return r.price + r.grad[0] + r.grad[1] + r.grad[2] + r.grad[3] + r.grad[4]
    if w == 1:
        var t = Tape[W, DType.float32](NS * 8 + 64)
        var s0 = Number[W, DType.float32](100, t); var r = Number[W, DType.float32](0.01, t)
        var sg = Number[W, DType.float32](0.2, t); var T = Number[W, DType.float32](1.0, t)
        var K = Number[W, DType.float32](100.0, t)
        t.set_mark()
        var v = EuropeanCallOption[W, DType.float32](s0, r, sg, T, K, NS, nP, z)
        var d = Float64(v.mean()) + Float64(s0.mean_grad()) + Float64(sg.mean_grad()) + Float64(r.mean_grad())
        t.free()
        return d
    var lo = log(100.0) - 1.2; var hi = log(100.0) + 1.2
    if w == 2:
        var f = fd_compiled[BsAmericanPut, 1, DType.float64](p, lo, hi, 401, 400)
        return f.price + f.grad[0] + f.grad[1] + f.grad[2] + f.grad[3] + f.grad[4]
    var pv = Array[SIMD[DType.float64, 1], 5](fill=0)
    for j in range(5):
        pv[j] = p[j]
    return Float64(fd_price[BsAmericanPut, 1, DType.float64](pv, lo, hi, 401, 400)[0])


def fmt(x: Float64) -> String:
    var v = Int(x * 100 + (0.5 if x >= 0 else -0.5))
    var frac = String(abs(v) % 100)
    if abs(v) % 100 < 10: frac = "0" + frac
    return (" " + ("-" if v < 0 else "") + String(abs(v) // 100) + "." + frac).ascii_rjust(16)


def main() raises:
    var nP = 200_000 // W
    var z = unsafe_alloc[Float32](NS * nP * W)
    std.random.seed(7); std.random.randn[DType.float32](z, NS * nP * W)
    var names: List[String] = ["MC compiled", "MC tape", "FD Amer 401x400", "FD price only"]
    var a = unsafe_alloc[UInt64](NEV); var b = unsafe_alloc[UInt64](NEV)
    var res = unsafe_alloc[UInt64](NWORK * NEV)
    print("per-run counts in millions (best of 7 by cycles); MC: GBM call 200k x 52 f32 W=32; FD: American put f64")
    for g in range(Int(external_call["kc_num_groups", Int32]())):
        var n = Int(external_call["kc_init", Int32](Int32(g)))
        if n == -3:
            print("PMU: permission denied - run with sudo"); return
        if n < 0:
            continue
        for w in range(NWORK):
            var best = UInt64.MAX
            for _ in range(7):
                external_call["kc_read", NoneType](a)
                var v = work(w, nP, z)
                external_call["kc_read", NoneType](b)
                if v == 12345.678: print(v)
                if b[unsafe_offset=0] - a[unsafe_offset=0] < best:
                    best = b[unsafe_offset=0] - a[unsafe_offset=0]
                    for e in range(NEV):
                        res[unsafe_offset=w * NEV + e] = b[unsafe_offset=e] - a[unsafe_offset=e]
        var hdr = String("event").ascii_ljust(40)
        for w in range(NWORK): hdr += names[w].ascii_rjust(16)
        print(hdr)
        for e in range(NEV):
            var nm = String(unsafe_from_utf8_ptr=external_call["kc_event_name", Pointer[UInt8, ImmUntrackedOrigin]](Int32(g), Int32(e)))
            var line = nm.ascii_ljust(40)
            if external_call["kc_event_ok", Int32](Int32(e)) == 0:
                print(line + "n/a"); continue
            for w in range(NWORK):
                line += fmt(Float64(res[unsafe_offset=w * NEV + e]) * 1e-6)
            print(line)
    a.unsafe_free(); b.unsafe_free(); res.unsafe_free(); z.unsafe_free()
