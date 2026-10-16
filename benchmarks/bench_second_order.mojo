"""Second-order cost: H.v (forward-over-reverse) and full Hessians vs first order, MC and FD.

Run from repo root: mojo run -I . benchmarks/bench_second_order.mojo
Every output (price, gradient, H.v / Hessian) is consumed so no sweep is dead code.
"""
from std.math import log
from std.memory.alloc import unsafe_alloc
from engines import mc_compiled, mc_hvp, mc_hessian, mc_hessian_bumped
from engines import FdModel, fd_compiled, fd_price, fd_hvp, fd_hessian, fd_hessian_bumped
from models.gbm import GbmCallSmooth
from models.bs_fd import BsAmericanPut, BsAmericanPutSmooth
import std.random, std.time

comptime NS = 52
comptime REPS = 11


def _p() -> Array[Float64, 5]:
    var p = Array[Float64, 5](fill=0)
    p[0] = 100; p[1] = 0.01; p[2] = 0.2; p[3] = 1.0; p[4] = 100
    return p^


def _v() -> Array[Float64, 5]:
    var v = Array[Float64, 5](fill=0)
    v[0] = 1; v[1] = 0.3; v[2] = -0.5; v[3] = 0.2; v[4] = 0.7
    return v^


def _now() -> Int:
    return Int(std.time.perf_counter_ns())


def _ms(a: Int, b: Int) -> Float64:
    return Float64(b - a) * 1e-6


def mc[W: Int, DT: DType]():
    var nP = 200_000 // W
    var z = unsafe_alloc[Float32](NS * nP * W)
    std.random.seed(7); std.random.randn[DType.float32](z, NS * nP * W)
    var p = _p(); var v = _v()
    var b = Array[Float64, 4](fill=1e9); var chk = 0.0
    for _ in range(REPS):
        var t0 = _now()
        var g = mc_compiled[GbmCallSmooth, W, DT](p, NS, nP, z)
        var t1 = _now()
        var hv = mc_hvp[GbmCallSmooth, W, DT](p, v, NS, nP, z)
        var t2 = _now()
        var h = mc_hessian[GbmCallSmooth, W, DT](p, NS, nP, z)
        var t3 = _now()
        var hb = mc_hessian_bumped[GbmCallSmooth, W, DT](p, NS, nP, z)
        var t4 = _now()
        chk += g.price + hv.price + hv.dprice + h.price + hb.price
        for i in range(5):
            chk += g.grad[i] + hv.grad[i] + hv.hv[i]
        for i in range(25):
            chk += h.h[i] + hb.h[i]
        b[0] = min(b[0], _ms(t0, t1)); b[1] = min(b[1], _ms(t1, t2))
        b[2] = min(b[2], _ms(t2, t3)); b[3] = min(b[3], _ms(t3, t4))
    print("MC smooth call 200k x", NS, DT, "W", W, "| grad", b[0], "ms | hvp", b[1], "(", b[1] / b[0],
          "x) | hessian AD", b[2], "(", b[2] / b[0], "x) | hessian bumped", b[3], "(", b[3] / b[0], "x) | chk", chk)
    z.unsafe_free()


def fd[W: Int, DT: DType, M: FdModel](nx: Int, nt: Int):
    var p5 = _p(); var v5 = _v()
    var p = Array[Float64, M.NP](fill=0); var v = Array[Float64, M.NP](fill=0)
    var pv = Array[SIMD[DT, W], M.NP](fill=0)
    for j in range(M.NP):
        p[j] = p5[j]; v[j] = v5[j]; pv[j] = SIMD[DT, W](p5[j])
    var lo = log(100.0) - 1.2; var hi = log(100.0) + 1.2
    var b = Array[Float64, 5](fill=1e9); var chk = 0.0
    for _ in range(REPS):
        var t0 = _now()
        var pr = fd_price[M, W, DT](pv, lo, hi, nx, nt)
        var t1 = _now()
        var g = fd_compiled[M, W, DT](p, lo, hi, nx, nt)
        var t2 = _now()
        var hv = fd_hvp[M, W, DT](p, v, lo, hi, nx, nt)
        var t3 = _now()
        var h = fd_hessian[M, W, DT](p, lo, hi, nx, nt)
        var t4 = _now()
        var hb = fd_hessian_bumped[M, W, DT](p, lo, hi, nx, nt)
        var t5 = _now()
        chk += Float64(pr[0].cast[DType.float64]()) + g.price + hv.price + hv.dprice + h.price + hb.price
        for i in range(M.NP):
            chk += g.grad[i] + hv.grad[i] + hv.hv[i]
        for i in range(M.NP * M.NP):
            chk += h.h[i] + hb.h[i]
        b[0] = min(b[0], _ms(t0, t1)); b[1] = min(b[1], _ms(t1, t2)); b[2] = min(b[2], _ms(t2, t3))
        b[3] = min(b[3], _ms(t3, t4)); b[4] = min(b[4], _ms(t4, t5))
    print("FD American put eps", M.EXERCISE_EPS, nx, "x", nt, DT, "W", W, "| price", b[0], "ms | grad", b[1], "(", b[1] / b[0],
          "x price) | hvp", b[2], "(", b[2] / b[1], "x grad) | hessian AD", b[3], "(", b[3] / b[1],
          "x grad) | hessian bumped", b[4], "(", b[4] / b[1], "x grad) | chk", chk)


def main() raises:
    mc[16, DType.float32]()
    mc[32, DType.float32]()
    mc[8, DType.float64]()
    fd[1, DType.float64, BsAmericanPut](201, 200)
    fd[1, DType.float64, BsAmericanPut](401, 400)
    fd[4, DType.float64, BsAmericanPut](401, 400)
    fd[1, DType.float64, BsAmericanPutSmooth[0.002]](201, 200)
    fd[1, DType.float64, BsAmericanPutSmooth[0.002]](401, 400)
    fd[4, DType.float64, BsAmericanPutSmooth[0.002]](401, 400)
