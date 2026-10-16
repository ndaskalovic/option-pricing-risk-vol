"""FD surface engine: one solve per underlier for every strike and maturity.

Run from repo root: mojo run -I . tests/test_fd_surface.mojo
1. European surface (price, delta, gamma, theta, dV/dK; Dual vega, rho) vs Black-Scholes closed form
2. Dual tangents (vega, rho) == central bumps of the surface (European exact, American)
3. American book: reverse-sweep gradient == sum of position-weighted forward tangents (exact)
4. American surface prices vs the single-option engine (fd_compiled, BsAmericanPut)
"""
from std.math import exp, log, sqrt, erf
from std.os import abort
from autodiff.ct import Vec, Dual, FdSurface, SurfacePoint, fd_surface, fd_surface_book, fd_compiled
from models.bs_fd import BsPutMoneyness, BsAmericanPut
import std.random

comptime F = DType.float64
comptime S0 = 100.0
comptime R_ = 0.03
comptime Q_ = 0.02
comptime SIG = 0.25


def mats() -> List[Float64]:
    var t = List[Float64]()
    for v in [1.0 / 52, 1.0 / 12, 0.25, 0.5, 1.0, 2.0]:
        t.append(v)
    return t^


def strikes() -> List[Float64]:
    var k = List[Float64]()
    for i in range(13):
        k.append(70.0 + 5.0 * Float64(i))
    return k^


def grid(sig: Float64, tmax: Float64) -> Tuple[Float64, Float64]:
    var z = 5.0 * sig * sqrt(tmax)
    return (log(S0 / 130.0) - z, log(S0 / 70.0) + z)


def nx_for(lo: Float64, hi: Float64, h: Float64) -> Int:
    return Int((hi - lo) / h) + 1


def ncdf(x: Float64) -> Float64:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def npdf(x: Float64) -> Float64:
    return exp(-0.5 * x * x) / sqrt(2.0 * 3.141592653589793)


def bs_put(k: Float64, t: Float64) -> List[Float64]:
    """Price, delta, gamma, theta (calendar), dV/dK, vega, rho."""
    var st = SIG * sqrt(t)
    var d1 = (log(S0 / k) + (R_ - Q_ + 0.5 * SIG * SIG) * t) / st; var d2 = d1 - st
    var dq = exp(-Q_ * t); var dr = exp(-R_ * t)
    var o = List[Float64]()
    o.append(k * dr * ncdf(-d2) - S0 * dq * ncdf(-d1))
    o.append(-dq * ncdf(-d1))
    o.append(dq * npdf(d1) / (S0 * st))
    o.append(-S0 * dq * npdf(d1) * SIG / (2 * sqrt(t)) + R_ * k * dr * ncdf(-d2) - Q_ * S0 * dq * ncdf(-d1))
    o.append(dr * ncdf(-d2))
    o.append(S0 * dq * npdf(d1) * sqrt(t))
    o.append(-k * t * dr * ncdf(-d2))
    return o^


def check(mut fails: Int, label: String, err: Float64, tol: Float64):
    print("   ", label, "max err", err, "(tol", tol, ")", "OK" if err <= tol else "FAIL")
    if not (err <= tol):
        fails += 1


def tangent_vs_bump[american: Bool](mut fails: Int, pe: Array[Dual[F, 2], 3], lo: Float64, hi: Float64, nx: Int,
                                    T: List[Float64], K: List[Float64], rel: Float64):
    comptime D2 = Dual[F, 2]
    comptime V1 = Vec[F, 1]
    var sa = fd_surface[BsPutMoneyness[american], D2](pe, lo, hi, nx, T, 1.0 / 500)
    var eps = 1e-6
    var bumped = List[FdSurface[V1]]()
    for bi in range(4):
        var pb = Array[V1, 3](fill=V1.splat(0))
        pb[0] = V1.splat(R_ + (eps if bi == 2 else (-eps if bi == 3 else 0.0)))
        pb[1] = V1.splat(Q_)
        pb[2] = V1.splat(SIG + (eps if bi == 0 else (-eps if bi == 1 else 0.0)))
        bumped.append(fd_surface[BsPutMoneyness[american], V1](pb, lo, hi, nx, T, 1.0 / 500))
    var ev = 0.0; var er = 0.0; var vscale = 0.0; var rscale = 0.0
    for m in range(len(T)):
        for k in K:
            var o = sa.option(m, SIMD[F, 2](S0), SIMD[F, 2](k))
            var pr = List[Float64]()
            for bi in range(4):
                pr.append(Float64(bumped[bi].option(m, SIMD[F, 1](S0), SIMD[F, 1](k))[0].value()[0]))
            var vb = (pr[0] - pr[1]) / (2 * eps); var rb = (pr[2] - pr[3]) / (2 * eps)
            ev = max(ev, abs(Float64(o[0].tangent()[0]) - vb)); er = max(er, abs(Float64(o[0].tangent()[1]) - rb))
            vscale = max(vscale, abs(vb)); rscale = max(rscale, abs(rb))
    sa.free()
    for bi in range(4):
        bumped[bi].free()
    var lab = "American" if american else "European"
    check(fails, String(lab) + " vega Dual vs bump (rel to max |vega|)", ev / vscale, rel)
    check(fails, String(lab) + " rho Dual vs bump (rel to max |rho|)", er / rscale, rel)


def main() raises:
    var fails = 0
    var T = mats(); var K = strikes()
    var lo_hi = grid(SIG, 2.0); var lo = lo_hi[0]; var hi = lo_hi[1]
    var nx = nx_for(lo, hi, 0.0025)

    # ── 1. European vs closed form; lane 0 tangent = d/dsigma, lane 1 = d/dr ──
    print("1. European surface vs Black-Scholes (h 0.0025, dt 1/1000, nx", nx, ")")
    comptime D2 = Dual[F, 2]
    var pe = Array[D2, 3](fill=D2.splat(0))
    pe[0] = D2(SIMD[F, 2](R_), SIMD[F, 2](0, 1))
    pe[1] = D2(SIMD[F, 2](Q_), SIMD[F, 2](0, 0))
    pe[2] = D2(SIMD[F, 2](SIG), SIMD[F, 2](1, 0))
    var se = fd_surface[BsPutMoneyness[False], D2](pe, lo, hi, nx, T, 1.0 / 1000)
    var emax = List[Float64](length=7, fill=0.0)
    for m in range(len(T)):
        for k in K:
            var o = se.option(m, SIMD[F, 2](S0), SIMD[F, 2](k))
            var b = bs_put(k, T[m])
            var got = List[Float64]()
            for g in range(5):
                got.append(Float64(o[g].value()[0]))
            got.append(Float64(o[0].tangent()[0]))      # vega
            got.append(Float64(o[0].tangent()[1]))      # rho
            for g in range(7):
                emax[g] = max(emax[g], abs(got[g] - b[g]))
    se.free()
    var names = ["price", "delta", "gamma", "theta", "dV/dK", "vega (Dual)", "rho (Dual)"]
    var tols = [2e-3, 2e-4, 2e-4, 5e-2, 2e-4, 2e-2, 2e-3]
    for g in range(7):
        check(fails, String(names[g]), emax[g], tols[g])

    # ── 2. Dual tangents == central bumps of the same scheme (European: exact; American: the
    #       tangent holds the exercise set fixed, a bump of 1e-6 can flip a node) ──
    print("2. Dual vega/rho == central bumps (h 0.005, dt 1/500)")
    var nx2 = nx_for(lo, hi, 0.005)
    tangent_vs_bump[False](fails, pe, lo, hi, nx2, T, K, 5e-7)
    tangent_vs_bump[True](fails, pe, lo, hi, nx2, T, K, 1e-5)

    # ── 3. American book: one reverse sweep == position-weighted forward tangents ──
    print("3. American book (", len(T) * len(K), "options, random positions): reverse sweep vs forward tangents")
    std.random.seed(3)
    var pos = List[Float64]()
    for _ in range(len(T) * len(K)):
        pos.append(std.random.random_float64(-10.0, 10.0))
    var pts = List[SurfacePoint[F, 1]]()
    for m in range(len(T)):
        for ki in range(len(K)):
            pts.append(SurfacePoint[F, 1](m, SIMD[F, 1](log(S0 / K[ki])), SIMD[F, 1](pos[m * len(K) + ki] * K[ki])))
    comptime V1 = Vec[F, 1]
    var pv = Array[V1, 3](fill=V1.splat(0))
    pv[0] = V1.splat(R_); pv[1] = V1.splat(Q_); pv[2] = V1.splat(SIG)
    var book = fd_surface_book[BsPutMoneyness[True], V1](pv, lo, hi, nx2, T, 1.0 / 500, pts)
    # forward: lanes 0, 1, 2 carry d/dr, d/dq, d/dsigma
    comptime D4 = Dual[F, 4]
    var pf = Array[D4, 3](fill=D4.splat(0))
    pf[0] = D4(SIMD[F, 4](R_), SIMD[F, 4](1, 0, 0, 0))
    pf[1] = D4(SIMD[F, 4](Q_), SIMD[F, 4](0, 1, 0, 0))
    pf[2] = D4(SIMD[F, 4](SIG), SIMD[F, 4](0, 0, 1, 0))
    var sf = fd_surface[BsPutMoneyness[True], D4](pf, lo, hi, nx2, T, 1.0 / 500)
    var fwd = SIMD[F, 4](0); var val = 0.0
    for m in range(len(T)):
        for ki in range(len(K)):
            var w = pos[m * len(K) + ki] * K[ki]
            var u = sf.read(m, SIMD[F, 4](log(S0 / K[ki])))[0]
            fwd += w * u.tangent(); val += w * Float64(u.value()[0])
    sf.free()
    var gs = ["dbook/dr", "dbook/dq", "dbook/dsigma"]
    var bv = Float64(book.value.value()[0])
    check(fails, "book value vs forward", abs(bv - val) / abs(val), 1e-12)
    for j in range(3):
        var g = Float64(book.grad[j].value()[0]); var f = Float64(fwd[j])
        print("     ", gs[j], "reverse", g, "forward", f)
        check(fails, String(gs[j]) + " rel", abs(g - f) / abs(f), 1e-10)
    book.surface.free()

    # ── 4. American surface vs the single-option engine (q = 0) ──
    print("4. American surface (q=0) vs fd_compiled[BsAmericanPut] 801x800 per option")
    var p0 = Array[V1, 3](fill=V1.splat(0))
    p0[0] = V1.splat(R_); p0[1] = V1.splat(0.0); p0[2] = V1.splat(SIG)
    var s0 = fd_surface[BsPutMoneyness[True], V1](p0, lo, hi, nx, T, 1.0 / 1000)
    var e4 = 0.0
    for m in range(len(T)):
        for k in [80.0, 100.0, 120.0]:
            var q = Array[Float64, 5](fill=0)
            q[0] = S0; q[1] = R_; q[2] = SIG; q[3] = T[m]; q[4] = k
            var z = 5.0 * SIG * sqrt(T[m])
            var single = fd_compiled[BsAmericanPut, 1, F](q, log(S0) - 0.5 - z, log(S0) + 0.5 + z, 801, 800)
            var surf = Float64(s0.option(m, SIMD[F, 1](S0), SIMD[F, 1](k))[0].value()[0])
            e4 = max(e4, abs(surf - single.price))
    s0.free()
    check(fails, "price vs single-option engine (abs, S=100)", e4, 2e-3)

    if fails > 0:
        abort("test_fd_surface FAILED")
    print("test_fd_surface PASSED")
