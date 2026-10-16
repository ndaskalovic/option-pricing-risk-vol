"""Compiled MC engine vs the tape pricer, price + 5 Greeks, f32, 200k paths x 52 steps.

Run from repo root: mojo run -I . benchmarks/bench_engine.mojo
"""
from std.memory.alloc import unsafe_alloc
from std.math import exp, sqrt, log
from autodiff import Number, Tape
from engines import mc_compiled
from models.gbm import GbmCall, GbmAsianCall
from vanilla_euro_call import EuropeanCallOption
import std.random, std.time

comptime NS = 52
comptime DT = DType.float32


@no_inline
def primal_log[W: Int](nP: Int, z: Pointer[Float32, _]) -> Float64:
    var dt = SIMD[DT, W](1.0) / NS; var drift = (SIMD[DT, W](0.01) - 0.02) * dt; var vs = SIMD[DT, W](0.2) * sqrt(dt)
    var acc = SIMD[DT, W](0); var x0 = log(SIMD[DT, W](100))
    for i in range(nP):
        var x = x0
        for j in range(NS):
            x = x + drift + vs * z.unsafe_load[width=W]((i * NS + j) * W)
        acc += max(exp(x) - 100, 0)
    return Float64(acc.reduce_add())


@no_inline
def tape_logspace[W: Int](nP: Int, z: Pointer[Float32, _]) -> Float64:
    var t = Tape[W, DT](NS * 8 + 64)
    var s0 = Number[W, DT](100, t); var r = Number[W, DT](0.01, t); var sg = Number[W, DT](0.2, t)
    var T = Number[W, DT](1.0, t); var K = Number[W, DT](100.0, t)
    t.set_mark()
    var p = EuropeanCallOption[W, DT](s0, r, sg, T, K, NS, nP, z)
    var d = Float64(p.mean()) + Float64(s0.mean_grad()) + Float64(r.mean_grad()) + Float64(sg.mean_grad()) + Float64(T.mean_grad()) + Float64(K.mean_grad())
    t.free(); return d


@no_inline
def engine[W: Int, asian: Bool](nP: Int, z: Pointer[Float32, _]) -> Float64:
    var p = Array[Float64, 5](fill=0)
    p[0] = 100; p[1] = 0.01; p[2] = 0.2; p[3] = 1.0; p[4] = 100
    # consume price and every gradient, so no part of the reverse sweep is dead code
    comptime if asian:
        var r = mc_compiled[GbmAsianCall, W, DT](p, NS, nP, z)
        return r.price + r.grad[0] + r.grad[1] + r.grad[2] + r.grad[3] + r.grad[4]
    else:
        var r = mc_compiled[GbmCall, W, DT](p, NS, nP, z)
        return r.price + r.grad[0] + r.grad[1] + r.grad[2] + r.grad[3] + r.grad[4]


def ms(a: Int, b: Int) -> Float64:
    return Float64(b - a) * 1e-6


def bench[W: Int]():
    var nP = 200_000 // W
    var z = unsafe_alloc[Float32](NS * nP * W)
    std.random.seed(7); std.random.randn[DType.float32](z, NS * nP * W)
    var b = Array[Float64, 4](fill=1e9); var v = 0.0
    for _ in range(21):
        var t0 = Int(std.time.perf_counter_ns()); v += primal_log[W](nP, z)
        var t1 = Int(std.time.perf_counter_ns()); v += tape_logspace[W](nP, z)
        var t2 = Int(std.time.perf_counter_ns()); v += engine[W, False](nP, z)
        var t3 = Int(std.time.perf_counter_ns()); v += engine[W, True](nP, z)
        var t4 = Int(std.time.perf_counter_ns())
        b[0] = min(b[0], ms(t0, t1)); b[1] = min(b[1], ms(t1, t2)); b[2] = min(b[2], ms(t2, t3)); b[3] = min(b[3], ms(t3, t4))
    print("W", W, "| primal(log)", b[0], "| tape pricer", b[1], "(", b[1] / b[0], "x) | compiled engine", b[2], "(", b[2] / b[0], "x) | engine Asian", b[3], "| tape/engine", b[1] / b[2], "| chk", v)
    z.unsafe_free()


def main() raises:
    bench[16]()
    bench[32]()
