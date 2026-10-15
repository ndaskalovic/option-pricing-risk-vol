"""FD engine: compiled == tape oracle (same scheme source, f64); European call vs Black-Scholes;
American put vs a CRR binomial tree.

Run from repo root: mojo run -I . tests/test_fd_engine.mojo
"""
from std.os import abort
from std.math import exp, sqrt, log
from autodiff.ct import FdModel, fd_compiled, fd_batch, fd_price, fd_tape
from models.bs_fd import BsEuroCall, BsAmericanPut

comptime W = 2
comptime DT = DType.float64


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


def compare[M: FdModel](label: StaticString, params: Array[Float64, M.NP], lo: Float64, hi: Float64, nx: Int, nt: Int) -> Int:
    var c = fd_compiled[M, W, DT](params, lo, hi, nx, nt)
    var t = fd_tape[M, W, DT](params, lo, hi, nx, nt)
    print(label); print("  compiled", c); print("  tape    ", t)
    var bad = 0
    if abs(c.price - t.price) > 1e-10 * abs(t.price): bad += 1; print("  price mismatch")
    for j in range(M.NP):
        if abs(c.grad[j] - t.grad[j]) > 1e-9 * (abs(t.grad[j]) + 1e-9):
            print("  grad mismatch", j); bad += 1
    return bad


def main() raises:
    var p = Array[Float64, 5](fill=0)
    p[0] = 100; p[1] = 0.01; p[2] = 0.2; p[3] = 1.0; p[4] = 100
    var lo = log(100.0) - 6 * 0.2; var hi = log(100.0) + 6 * 0.2
    var failures = compare[BsEuroCall]("European call, small grid", p, lo, hi, 61, 40)
    failures += compare[BsAmericanPut]("American put, small grid", p, lo, hi, 61, 40)

    var e = fd_compiled[BsEuroCall, W, DT](p, lo, hi, 801, 800)
    var bs: List[Float64] = [8.433318690109608, 0.5596176923702425, 47.52845054691464, 39.447933090788894, 4.420077814548036, -0.4752845054691464]
    var ours: List[Float64] = [e.price, e.grad[0], e.grad[1], e.grad[2], e.grad[3], e.grad[4]]
    print("European call 801x800 vs Black-Scholes:")
    for i in range(6):
        var rel = abs(ours[i] - bs[i]) / abs(bs[i])
        print("  ", ours[i], bs[i], "rel err", rel)
        if rel > 2e-3: failures += 1
    var a = fd_compiled[BsAmericanPut, W, DT](p, lo, hi, 801, 800)
    var crr = crr_american_put(100, 100, 0.01, 0.2, 1.0, 4000)
    print("American put 801x800:", a.price, " CRR(4000):", crr, " rel err", abs(a.price - crr) / crr)
    if abs(a.price - crr) / crr > 2e-3: failures += 1

    # batch: 4 strikes in one solve == 4 single solves; fd_price == price of the full run
    var pv = Array[SIMD[DType.float64, 4], 5](fill=0)
    pv[0] = 100; pv[1] = 0.01; pv[2] = 0.2; pv[3] = 1.0; pv[4] = SIMD[DType.float64, 4](90, 100, 110, 120)
    var bt = fd_batch[BsAmericanPut, 4, DType.float64](pv, lo, hi, 201, 200)
    var po = fd_price[BsAmericanPut, 4, DType.float64](pv, lo, hi, 201, 200)
    for k in range(4):
        var q = p.copy(); q[4] = Float64(pv[4][k])
        var sgl = fd_compiled[BsAmericanPut, 1, DType.float64](q, lo, hi, 201, 200)
        if abs(bt.price[k] - sgl.price) > 1e-12 * sgl.price or abs(po[k] - sgl.price) > 1e-12 * sgl.price:
            print("  batch/price-only mismatch lane", k); failures += 1
        for j in range(5):
            if abs(bt.grad[j][k] - sgl.grad[j]) > 1e-10 * (abs(sgl.grad[j]) + 1e-12):
                print("  batch grad mismatch lane", k, "param", j); failures += 1
    print("batch (4 strikes) == single solves:", "OK" if failures == 0 else "FAIL")
    if failures > 0:
        abort("test_fd_engine FAILED")
    print("test_fd_engine PASSED")
