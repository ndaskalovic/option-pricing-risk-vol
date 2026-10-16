"""MC engine, f64: forward == reverse, gradient == finite differences, GBM call vs Black-Scholes.

Run from repo root: mojo run -I . tests/test_mc_engine.mojo
Per model (GBM call, Asian call, SVI call):
1. dot-product test: price tangent with Dual params (forward mode through the whole engine)
   == grad . v from the reverse sweep
2. gradient == central differences of the price with common random numbers (SVI's T is
   detached inside the vol by design, so it is excluded from this check)
"""
from std.os import abort
from std.memory.alloc import unsafe_alloc
from autodiff import Vec, Dual
from engines import McModel, mc_greeks
from models.gbm import GbmCall, GbmAsianCall
from models.svi import SviCall
import std.random

comptime W = 8
comptime DT = DType.float64
comptime RV = Vec[DT, W]
comptime RD = Dual[DT, W]


def vec_params[NP: Int](p: Array[Float64, NP]) -> Array[RV, NP]:
    var r = Array[RV, NP](fill=RV.splat(0))
    for j in range(NP):
        r[j] = RV.splat(p[j])
    return r^


def check(mut fails: Int, label: String, err: Float64, tol: Float64):
    var ok = err <= tol
    print("   ", label, "max rel err", err, "(tol", tol, ")", "OK" if ok else "FAIL")
    if not ok:
        fails += 1


def verify[M: McModel](mut fails: Int, label: String, p: Array[Float64, M.NP], n_steps: Int, n_paths: Int,
                       z: Pointer[Float32, _], detached: Int = -1):
    """`detached`: a parameter the model passes through `detach` somewhere (stop-gradient), so its
    AD derivative intentionally differs from a finite difference; it is left out of check 2."""
    print(label)
    var r = mc_greeks[M, RV](vec_params(p), n_steps, n_paths, z)
    print("   ", r)
    # 1. dot test
    var pd = Array[RD, M.NP](fill=RD.splat(0))
    var gv = 0.0
    for j in range(M.NP):
        var v = 0.3 + 0.1 * Float64(j)
        pd[j] = RD(SIMD[DT, W](p[j]), SIMD[DT, W](v))
        gv += r.grad[j] * v
    var d = mc_greeks[M, RD](pd, n_steps, n_paths, z)
    check(fails, "dot test: Dual price tangent vs grad . v", abs(d.price_tangent - gv) / abs(gv), 1e-12)
    # 2. central differences (same normals)
    var e = 0.0
    for j in range(M.NP):
        var h = 1e-5 * max(abs(p[j]), 1e-2)
        var pp = p.copy(); pp[j] += h
        var pm = p.copy(); pm[j] -= h
        var fdv = (mc_greeks[M, RV](vec_params(pp), n_steps, n_paths, z).price
                   - mc_greeks[M, RV](vec_params(pm), n_steps, n_paths, z).price) / (2 * h)
        if j != detached:
            e = max(e, abs(r.grad[j] - fdv) / max(abs(fdv), 1e-2))
    check(fails, "gradient vs central differences", e, 1e-4)


def main() raises:
    var n_steps = 52
    var n_paths = 200_000 // W
    var z = unsafe_alloc[Float32](n_paths * n_steps * W)
    std.random.seed(42)
    std.random.randn[DType.float32](z, n_paths * n_steps * W)
    var fails = 0
    var p = Array[Float64, 5](fill=0)
    p[0] = 100; p[1] = 0.01; p[2] = 0.2; p[3] = 1.0; p[4] = 100
    verify[GbmCall](fails, "GBM call (s0, r, sigma, T, K)", p, n_steps, n_paths, z)
    verify[GbmAsianCall](fails, "GBM Asian call", p, n_steps, n_paths // 4, z)
    var sp = Array[Float64, 9](fill=0)
    sp[0] = 227.7899932861328; sp[1] = 0.04; sp[2] = 0.7287671233; sp[3] = 220
    sp[4] = 5.73431376e-02; sp[5] = 6.59987208e-04; sp[6] = -1.23157254; sp[7] = 2.22873847e+02; sp[8] = 9.99943135e-02
    verify[SviCall](fails, "SVI call (s0, r, T, K, a, b, rho, m, sigma)", sp, n_steps, n_paths, z, detached=2)

    # GBM call vs Black-Scholes, within 4 standard errors
    var c = mc_greeks[GbmCall, RV](vec_params(p), n_steps, n_paths, z)
    var bs: List[Float64] = [8.433318690109608, 0.5596176923702425, 47.52845054691464, 39.447933090788894, 4.420077814548036, -0.4752845054691464]
    var tol: List[Float64] = [0.12, 0.0053, 0.44, 0.66, 0.2, 0.0044]
    var ours: List[Float64] = [c.price, c.grad[0], c.grad[1], c.grad[2], c.grad[3], c.grad[4]]
    var bad = 0
    for i in range(6):
        if abs(ours[i] - bs[i]) > tol[i]:
            print("    BS check failed", i, ours[i], bs[i]); bad += 1
    print("GBM call vs Black-Scholes (4 SE):", "OK" if bad == 0 else "FAIL")
    fails += bad
    z.unsafe_free()
    if fails > 0:
        abort("test_mc_engine FAILED")
    print("test_mc_engine PASSED")
