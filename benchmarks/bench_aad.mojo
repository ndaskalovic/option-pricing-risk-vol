"""AAD overhead benchmark: EuropeanCallOption (price + 5 Greeks) vs a plain-SIMD primal.

Run from repo root: mojo run -I . benchmarks/bench_aad.mojo
"""
from std.memory.alloc import unsafe_alloc
from std.math import exp, sqrt
from autodiff import Number, Tape
from vanilla_euro_call import EuropeanCallOption
import std.random
import std.time

comptime DT = DType.float32
comptime REPS = 15


def primal[W: Int](nSteps: Int, nP: Int, z: Pointer[Float32, _]) -> Float64:
    var dt = Float32(1.0) / Float32(nSteps)
    var drift = (Float32(0.01) - 0.02) * dt
    var vs = Float32(0.2) * sqrt(dt)
    var acc = SIMD[DT, W](0)
    for i in range(nP):
        var s = SIMD[DT, W](100)
        for j in range(nSteps):
            s = s * exp(drift + vs * z.unsafe_load[width=W]((i * nSteps + j) * W))
        acc += max(s - 100, 0)
    return Float64(acc.reduce_add()) * Float64(exp(Float32(-0.01))) / Float64(nP * W)


def aad[W: Int](nSteps: Int, nP: Int, z: Pointer[Float32, _]) -> Float64:
    var tape = Tape[W, DT](nSteps * 8 + 64)
    var s0 = Number[W, DT](100, tape)
    var r = Number[W, DT](0.01, tape)
    var sigma = Number[W, DT](0.2, tape)
    var T = Number[W, DT](1.0, tape)
    var K = Number[W, DT](100.0, tape)
    tape.set_mark()
    var p = EuropeanCallOption[W, DT](s0, r, sigma, T, K, nSteps, nP, z)
    var d = Float64(s0.mean_grad()) + 0 * Float64(p.mean())
    tape.free()
    return d


def bench[W: Int](nSteps: Int, nPathsScalar: Int = 200_000):
    var nP = nPathsScalar // W
    var z = unsafe_alloc[Float32](nSteps * nP * W)
    std.random.seed(7)
    std.random.randn[DType.float32](z, nSteps * nP * W)
    var tp = 1e9
    var ta = 1e9
    var v = 0.0
    for _ in range(REPS):
        var t0 = std.time.perf_counter_ns()
        v += primal[W](nSteps, nP, z)
        var t1 = std.time.perf_counter_ns()
        v += aad[W](nSteps, nP, z)
        var t2 = std.time.perf_counter_ns()
        tp = min(tp, Float64(Int(t1 - t0)) * 1e-6)
        ta = min(ta, Float64(Int(t2 - t1)) * 1e-6)
    print("W", W, "| paths", nPathsScalar, "| steps", nSteps, "| primal ms", tp, "| aad ms", ta, "| aad/primal", ta / tp, "| chk", v)
    z.unsafe_free()


def main() raises:
    comptime for w in [8, 16, 32, 64]:
        bench[w](52)
    bench[32](252)

