"""FD engine, f64: forward == reverse, gradient == finite differences, European call vs
Black-Scholes, American put vs a CRR binomial tree, lanes == single solves.

Run from repo root: mojo run -I . tests/test_fd_engine.mojo
"""
from std.os import abort
from std.math import exp, sqrt, log
from autodiff import Vec, Dual
from engines import FdModel, fd_greeks, fd_price
from models.bs_fd import BsEuroCall, BsAmericanPut

comptime DT = DType.float64
comptime R1 = Vec[DT, 1]
comptime RD = Dual[DT, 1]


def vec_params[NP: Int](p: Array[Float64, NP]) -> Array[R1, NP]:
    var r = Array[R1, NP](fill=R1.splat(0))
    for j in range(NP):
        r[j] = R1.splat(p[j])
    return r^


def lane0(v: SIMD) -> Float64:
    return Float64(v[0].cast[DType.float64]())


def crr_american_put(S: Float64, K: Float64, r: Float64, sigma: Float64, T: Float64, n: Int) -> Float64:
    var dt = T / Float64(n); var up = exp(sigma * sqrt(dt)); var dn = 1.0 / up
    var q = (exp(r * dt) - dn) / (up - dn); var disc = exp(-r * dt)
    var v = List[Float64]()
    for i in range(n + 1):
        v.append(max(K - S * up ** Float64(i) * dn ** Float64(n - i), 0.0))
    for step in range(n - 1, -1, -1):
        for i in range(step + 1):
            var cont = disc * (q * v[i + 1] + (1 - q) * v[i])
            v[i] = max(cont, K - S * up ** Float64(i) * dn ** Float64(step - i))
    return v[0]


def check(mut fails: Int, label: String, err: Float64, tol: Float64):
    var ok = err <= tol
    print("   ", label, "max err", err, "(tol", tol, ")", "OK" if ok else "FAIL")
    if not ok:
        fails += 1


def verify[M: FdModel](mut fails: Int, label: String, p: Array[Float64, M.NP], lo: Float64, hi: Float64,
                       nx: Int, nt: Int, fd_tol: Float64):
    print(label, nx, "x", nt)
    var r = fd_greeks[M, R1](vec_params(p), lo, hi, nx, nt)
    # dot test: forward tangent of the price through the scheme vs the reverse sweep
    var pd = Array[RD, M.NP](fill=RD.splat(0))
    var gv = 0.0
    for j in range(M.NP):
        var v = 0.3 + 0.1 * Float64(j)
        pd[j] = RD(SIMD[DT, 1](p[j]), SIMD[DT, 1](v))
        gv += lane0(r.grad[j].value()) * v
    var d = fd_greeks[M, RD](pd, lo, hi, nx, nt)
    check(fails, "dot test: Dual price tangent vs grad . v (rel)", abs(lane0(d.price.tangent()) - gv) / abs(gv), 1e-12)
    # gradient vs central differences of the price-only solve
    var e = 0.0
    for j in range(M.NP):
        var h = 1e-6 * max(abs(p[j]), 1e-2)
        var pp = p.copy(); pp[j] += h
        var pm = p.copy(); pm[j] -= h
        var fdv = (lane0(fd_price[M, R1](vec_params(pp), lo, hi, nx, nt).value())
                   - lane0(fd_price[M, R1](vec_params(pm), lo, hi, nx, nt).value())) / (2 * h)
        var g = lane0(r.grad[j].value())
        e = max(e, abs(g - fdv) / max(abs(fdv), 1e-3))
    check(fails, "gradient vs central differences (rel)", e, fd_tol)


def main() raises:
    var fails = 0
    var p = Array[Float64, 5](fill=0)
    p[0] = 100; p[1] = 0.01; p[2] = 0.2; p[3] = 1.0; p[4] = 100
    var lo = log(100.0) - 6 * 0.2; var hi = log(100.0) + 6 * 0.2
    verify[BsEuroCall](fails, "European call", p, lo, hi, 201, 200, 1e-6)
    verify[BsAmericanPut](fails, "American put", p, lo, hi, 201, 200, 1e-6)

    var e = fd_greeks[BsEuroCall, R1](vec_params(p), lo, hi, 801, 800)
    var bs: List[Float64] = [8.433318690109608, 0.5596176923702425, 47.52845054691464, 39.447933090788894, 4.420077814548036, -0.4752845054691464]
    var worst = abs(lane0(e.price.value()) - bs[0]) / bs[0]
    for j in range(5):
        worst = max(worst, abs(lane0(e.grad[j].value()) - bs[j + 1]) / abs(bs[j + 1]))
    print("European call 801x800 vs Black-Scholes")
    check(fails, "price and 5 Greeks (rel)", worst, 2e-5)
    var a = lane0(fd_price[BsAmericanPut, R1](vec_params(p), lo, hi, 801, 800).value())
    var crr = crr_american_put(100, 100, 0.01, 0.2, 1.0, 4000)
    print("American put 801x800 vs CRR(4000):", a, crr)
    check(fails, "price (rel)", abs(a - crr) / crr, 5e-5)

    # 4 strikes in 4 lanes == 4 single solves; fd_price == fd_greeks price
    comptime R4 = Vec[DT, 4]
    var pv = Array[R4, 5](fill=R4.splat(0))
    pv[0] = R4.splat(100); pv[1] = R4.splat(0.01); pv[2] = R4.splat(0.2); pv[3] = R4.splat(1.0)
    pv[4] = R4(SIMD[DT, 4](90, 100, 110, 120))
    var bt = fd_greeks[BsAmericanPut, R4](pv, lo, hi, 201, 200)
    var po = fd_price[BsAmericanPut, R4](pv, lo, hi, 201, 200)
    var lane_err = 0.0
    for k in range(4):
        var q = p.copy(); q[4] = Float64(pv[4].value()[k])
        var sgl = fd_greeks[BsAmericanPut, R1](vec_params(q), lo, hi, 201, 200)
        var sp = lane0(sgl.price.value())
        lane_err = max(lane_err, abs(Float64(bt.price.value()[k]) - sp) / sp, abs(Float64(po.value()[k]) - sp) / sp)
        for j in range(5):
            var sg = lane0(sgl.grad[j].value())
            lane_err = max(lane_err, abs(Float64(bt.grad[j].value()[k]) - sg) / (abs(sg) + 1e-12))
    print("4 strikes in lanes vs single solves")
    check(fails, "price and gradient (rel)", lane_err, 1e-12)
    if fails > 0:
        abort("test_fd_engine FAILED")
    print("test_fd_engine PASSED")
