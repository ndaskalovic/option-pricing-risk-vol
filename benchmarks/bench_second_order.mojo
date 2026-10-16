"""Second-order cost vs the gradient, MC and FD, 5 parameters.

Run from repo root: mojo run -I . benchmarks/bench_second_order.mojo
  H.v            one forward-over-reverse pass (Dual parameters seeded with v)
  AD Hessian     5 such passes (v = e_j)
  bumped Hessian 11 gradient passes (central differences of the AAD gradient)
Every output is consumed.
"""
from std.math import log
from std.memory.alloc import unsafe_alloc
from autodiff import Vec, Dual
from engines import FdModel, McModel, mc_greeks, fd_greeks, fd_price
from models.gbm import GbmCallSmooth
from models.bs_fd import BsAmericanPut, BsAmericanPutSmooth
import std.random, std.time

comptime NS = 52
comptime REPS = 11
comptime NP = 5


def base() -> Array[Float64, NP]:
    var p = Array[Float64, NP](fill=0)
    p[0] = 100; p[1] = 0.01; p[2] = 0.2; p[3] = 1.0; p[4] = 100
    return p^


def vec[DT: DType, W: Int](p: Array[Float64, NP]) -> Array[Vec[DT, W], NP]:
    var r = Array[Vec[DT, W], NP](fill=Vec[DT, W].splat(0))
    for j in range(NP):
        r[j] = Vec[DT, W].splat(p[j])
    return r^


def dual[DT: DType, W: Int](p: Array[Float64, NP], v: Array[Float64, NP]) -> Array[Dual[DT, W], NP]:
    var r = Array[Dual[DT, W], NP](fill=Dual[DT, W].splat(0))
    for j in range(NP):
        r[j] = Dual[DT, W](SIMD[DT, W](p[j]), SIMD[DT, W](v[j]))
    return r^


def unit(j: Int) -> Array[Float64, NP]:
    var e = Array[Float64, NP](fill=0)
    if j >= 0:
        e[j] = 1
    return e^


def bumped(p: Array[Float64, NP], j: Int, s: Float64) -> Array[Float64, NP]:
    var q = p.copy()
    q[j] += s * 1e-2 * max(abs(p[j]), 1e-2)
    return q^


def _now() -> Int:
    return Int(std.time.perf_counter_ns())


def _ms(a: Int, b: Int) -> Float64:
    return Float64(b - a) * 1e-6


def mc[W: Int, DT: DType]():
    comptime M = GbmCallSmooth
    var nP = 200_000 // W
    var z = unsafe_alloc[Float32](NS * nP * W)
    std.random.seed(7); std.random.randn[DType.float32](z, NS * nP * W)
    var p = base()
    var v = unit(-1)
    v[0] = 1; v[1] = 0.3; v[2] = -0.5; v[3] = 0.2; v[4] = 0.7
    var b = Array[Float64, 4](fill=1e9); var chk = 0.0
    for _ in range(REPS):
        var t0 = _now()
        var g = mc_greeks[M, Vec[DT, W]](vec[DT, W](p), NS, nP, z)
        chk += g.price + g.grad[0] + g.grad[1] + g.grad[2] + g.grad[3] + g.grad[4]
        var t1 = _now()
        var h = mc_greeks[M, Dual[DT, W]](dual[DT, W](p, v), NS, nP, z)
        chk += h.price_tangent
        for i in range(NP):
            chk += h.grad_tangent[i]
        var t2 = _now()
        for j in range(NP):
            var c = mc_greeks[M, Dual[DT, W]](dual[DT, W](p, unit(j)), NS, nP, z)
            for i in range(NP):
                chk += c.grad_tangent[i]
        var t3 = _now()
        chk += mc_greeks[M, Vec[DT, W]](vec[DT, W](p), NS, nP, z).price
        for j in range(NP):
            var gp = mc_greeks[M, Vec[DT, W]](vec[DT, W](bumped(p, j, 1)), NS, nP, z)
            var gm = mc_greeks[M, Vec[DT, W]](vec[DT, W](bumped(p, j, -1)), NS, nP, z)
            for i in range(NP):
                chk += gp.grad[i] - gm.grad[i]
        var t4 = _now()
        b[0] = min(b[0], _ms(t0, t1)); b[1] = min(b[1], _ms(t1, t2))
        b[2] = min(b[2], _ms(t2, t3)); b[3] = min(b[3], _ms(t3, t4))
    print("MC smooth call 200k x", NS, DT, "W", W, "| grad", b[0], "ms | H.v", b[1], "(", b[1] / b[0],
          "x) | AD Hessian", b[2], "(", b[2] / b[0], "x) | bumped Hessian", b[3], "(", b[3] / b[0], "x) | chk", chk)
    z.unsafe_free()


def fd[W: Int, DT: DType, M: FdModel](nx: Int, nt: Int):
    var p5 = base()
    var p = Array[Float64, M.NP](fill=0)
    var v = Array[Float64, M.NP](fill=0)
    for j in range(M.NP):
        p[j] = p5[j]; v[j] = 0.3 + 0.1 * Float64(j)
    var lo = log(100.0) - 1.2; var hi = log(100.0) + 1.2
    comptime V = Vec[DT, W]
    comptime D = Dual[DT, W]

    def vp(q: Array[Float64, M.NP]) capturing -> Array[V, M.NP]:
        var r = Array[V, M.NP](fill=V.splat(0))
        for j in range(M.NP):
            r[j] = V.splat(q[j])
        return r^

    def dp(q: Array[Float64, M.NP], t: Array[Float64, M.NP]) capturing -> Array[D, M.NP]:
        var r = Array[D, M.NP](fill=D.splat(0))
        for j in range(M.NP):
            r[j] = D(SIMD[DT, W](q[j]), SIMD[DT, W](t[j]))
        return r^

    var b = Array[Float64, 5](fill=1e9); var chk = 0.0
    for _ in range(REPS):
        var t0 = _now()
        chk += Float64(fd_price[M, V](vp(p), lo, hi, nx, nt).value()[0])
        var t1 = _now()
        var g = fd_greeks[M, V](vp(p), lo, hi, nx, nt)
        chk += Float64(g.price.value()[0])
        for i in range(M.NP):
            chk += Float64(g.grad[i].value()[0])
        var t2 = _now()
        var h = fd_greeks[M, D](dp(p, v), lo, hi, nx, nt)
        chk += Float64(h.price.tangent()[0])
        for i in range(M.NP):
            chk += Float64(h.grad[i].tangent()[0])
        var t3 = _now()
        for j in range(M.NP):
            var e = Array[Float64, M.NP](fill=0)
            e[j] = 1
            var c = fd_greeks[M, D](dp(p, e), lo, hi, nx, nt)
            for i in range(M.NP):
                chk += Float64(c.grad[i].tangent()[0])
        var t4 = _now()
        chk += Float64(fd_greeks[M, V](vp(p), lo, hi, nx, nt).price.value()[0])
        for j in range(M.NP):
            var qp = p.copy(); qp[j] += 1e-2 * max(abs(p[j]), 1e-2)
            var qm = p.copy(); qm[j] -= 1e-2 * max(abs(p[j]), 1e-2)
            var gp = fd_greeks[M, V](vp(qp), lo, hi, nx, nt)
            var gm = fd_greeks[M, V](vp(qm), lo, hi, nx, nt)
            for i in range(M.NP):
                chk += Float64(gp.grad[i].value()[0] - gm.grad[i].value()[0])
        var t5 = _now()
        b[0] = min(b[0], _ms(t0, t1)); b[1] = min(b[1], _ms(t1, t2)); b[2] = min(b[2], _ms(t2, t3))
        b[3] = min(b[3], _ms(t3, t4)); b[4] = min(b[4], _ms(t4, t5))
    print("FD American put eps", M.EXERCISE_EPS, nx, "x", nt, DT, "W", W, "| price", b[0], "ms | grad", b[1],
          "(", b[1] / b[0], "x price) | H.v", b[2], "(", b[2] / b[1], "x grad) | AD Hessian", b[3], "(", b[3] / b[1],
          "x grad) | bumped Hessian", b[4], "(", b[4] / b[1], "x grad) | chk", chk)


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
