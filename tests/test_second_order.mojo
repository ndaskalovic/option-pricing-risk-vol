"""Second-order Greeks (forward-over-reverse): exactness, symmetry, and references.

What is asserted (and therefore claimed):
  * every AD Hessian equals central differences of the exact AAD gradient (it is the exact
    second derivative of the discrete estimator / scheme) and is symmetric;
  * FD European call: all entries within 2e-3 of the Black-Scholes Hessian except d2V/dK2
    (the strike only enters the kinked payoff), which `fd_hessian_bumped` gets within 5e-3;
  * FD American put: with smoothed exercise (eps=0.002) every entry except d2V/dK2 matches a
    1%-bump reference within 5e-3; with the hard max only the S0 row/column (gamma, vanna, ...)
    is valid;
  * MC (softplus-smoothed payoff, eps=0.5): gamma and vanna match Black-Scholes within 4 standard
    errors; the kinked payoff needs `mc_hessian_bumped`.

Run from repo root: mojo run -I . tests/test_second_order.mojo
"""
from std.os import abort
from std.math import log
from std.memory.alloc import unsafe_alloc
from engines import McModel, FdModel, Hessian, mc_compiled, mc_hessian, mc_hessian_bumped, fd_compiled, fd_hessian, fd_hessian_bumped
from models.gbm import GbmCall, GbmCallSmooth
from models.bs_fd import BsEuroCall, BsAmericanPut, BsAmericanPutSmooth
import std.random

comptime S0 = 0
comptime RATE = 1
comptime SIG = 2
comptime MAT = 3
comptime STRIKE = 4


def names() -> List[String]:
    return ["S0", "r", "sigma", "T", "K"]


def params() -> Array[Float64, 5]:
    var p = Array[Float64, 5](fill=0)
    p[0] = 100; p[1] = 0.01; p[2] = 0.2; p[3] = 1.0; p[4] = 100
    return p^


def bs() -> Array[Float64, 25]:
    """Black-Scholes Hessian over (S0, r, sigma, T, K), mpmath at 40 digits."""
    var v: List[Float64] = [0.019723966545394445, 1.9723966545394445, 0.098619832726972224, 0.029585949818091667, -0.019723966545394445, 1.9723966545394445, 149.7112149070298, -29.585949818091667, 46.066967714175774, -1.497112149070298, 0.098619832726972224, -29.585949818091667, -1.4792974909045834, 19.28017729812307, 0.29585949818091667, 0.029585949818091667, 46.066967714175774, 19.28017729812307, -2.0313904075939707, 0.014614828327388687, -0.019723966545394445, -1.497112149070298, 0.29585949818091667, 0.014614828327388687, 0.019723966545394445]
    var r = Array[Float64, 25](fill=0)
    for k in range(25):
        r[k] = v[k]
    return r^


def show(label: String, h: Hessian[5]):
    print(label)
    for i in range(5):
        var line = String("  ") + names()[i].ascii_ljust(6)
        for j in range(5):
            line += String(h.get(i, j)).ascii_rjust(24)
        print(line)


def rel(a: Float64, b: Float64) -> Float64:
    return abs(a - b) / max(abs(b), 1e-9)


def check(mut failures: Int, label: String, value: Float64, refv: Float64, tol: Float64):
    var e = rel(value, refv)
    var ok = e <= tol
    print("   ", label.ascii_ljust(46), String(value).ascii_rjust(22), String(refv).ascii_rjust(22), " rel", e, "OK" if ok else "FAIL")
    if not ok:
        failures += 1


def check_abs(mut failures: Int, label: String, value: Float64, refv: Float64, tol: Float64):
    var ok = abs(value - refv) <= tol
    print("   ", label.ascii_ljust(46), String(value).ascii_rjust(22), String(refv).ascii_rjust(22), " abs", abs(value - refv), "OK" if ok else "FAIL")
    if not ok:
        failures += 1


def check_le(mut failures: Int, label: String, value: Float64, tol: Float64):
    var ok = value <= tol
    print("   ", label.ascii_ljust(46), String(value).ascii_rjust(22), String("<= ") + String(tol), "OK" if ok else "FAIL")
    if not ok:
        failures += 1


def max_asym(h: Hessian[5]) -> Float64:
    var m = 0.0
    for i in range(5):
        for j in range(i + 1, 5):
            m = max(m, abs(h.get(i, j) - h.get(j, i)) / (abs(h.get(i, j)) + abs(h.get(j, i)) + 1e-12))
    return m


def max_rel_vs(a: Hessian[5], b: Hessian[5], floor: Float64, skip_kk: Bool) -> Float64:
    var m = 0.0
    for i in range(5):
        for j in range(5):
            if skip_kk and i == STRIKE and j == STRIKE:
                continue
            m = max(m, abs(a.get(i, j) - b.get(i, j)) / max(abs(b.get(i, j)), floor))
    return m


def main() raises:
    var p = params()
    var lo = log(100.0) - 1.2; var hi = log(100.0) + 1.2
    var bsh = bs()
    var failures = 0
    var nx = 401; var nt = 400

    # ── FD European call ──────────────────────────────────────────────
    print("FD European call", nx, "x", nt)
    var he = fd_hessian[BsEuroCall, 1, DType.float64](p, lo, hi, nx, nt)
    var tiny = fd_hessian_bumped[BsEuroCall, 1, DType.float64](p, lo, hi, nx, nt, 1e-4)
    var big = fd_hessian_bumped[BsEuroCall, 1, DType.float64](p, lo, hi, nx, nt, 1e-2)
    show("  AD Hessian", he)
    check_le(failures, "exact: max rel vs FD of AAD gradient (h=1e-4)", max_rel_vs(he, tiny, 1e-6, False), 1e-5)
    check_le(failures, "symmetry: max rel asymmetry", max_asym(he), 1e-9)
    var worst = 0.0
    for i in range(5):
        for j in range(5):
            if i == STRIKE and j == STRIKE:
                continue
            worst = max(worst, rel(he.get(i, j), bsh[i * 5 + j]))
    check_le(failures, "vs Black-Scholes, max rel (all but d2/dK2)", worst, 2e-3)
    check(failures, "gamma", he.get(S0, S0), bsh[0], 2e-3)
    check(failures, "vanna d2/dS0dsigma", he.get(S0, SIG), bsh[2], 2e-3)
    check(failures, "volga d2/dsigma2", he.get(SIG, SIG), bsh[12], 2e-3)
    check(failures, "d2/dK2 via fd_hessian_bumped (1%)", big.get(STRIKE, STRIKE), bsh[24], 5e-3)
    print("    (AD d2/dK2 =", he.get(STRIKE, STRIKE), ": the strike only enters through the kinked payoff)")

    # ── FD American put ───────────────────────────────────────────────
    print("FD American put", nx, "x", nt)
    comptime AS = BsAmericanPutSmooth[0.002]
    var ha = fd_hessian[AS, 1, DType.float64](p, lo, hi, nx, nt)
    var ha_tiny = fd_hessian_bumped[AS, 1, DType.float64](p, lo, hi, nx, nt, 1e-4)
    var ha_big = fd_hessian_bumped[AS, 1, DType.float64](p, lo, hi, nx, nt, 1e-2)
    show("  AD Hessian, smoothed exercise eps=0.002", ha)
    check_le(failures, "exact: max rel vs FD of AAD gradient (h=1e-4)", max_rel_vs(ha, ha_tiny, 1e-3, False), 2e-3)
    check_le(failures, "symmetry", max_asym(ha), 1e-9)
    check_le(failures, "vs 1%-bump reference, max rel (all but d2/dK2)", max_rel_vs(ha, ha_big, 1e-3, True), 5e-3)
    var hard = fd_hessian[BsAmericanPut, 1, DType.float64](p, lo, hi, nx, nt)
    var hard_big = fd_hessian_bumped[BsAmericanPut, 1, DType.float64](p, lo, hi, nx, nt, 1e-2)
    var hw = 0.0
    for j in range(5):
        hw = max(hw, rel(hard.get(S0, j), hard_big.get(S0, j)))
    check_le(failures, "hard max: S0 row vs 1%-bump reference", hw, 2e-3)
    check(failures, "hard max gamma vs smoothed gamma", hard.get(S0, S0), ha.get(S0, S0), 2e-3)
    check(failures, "d2/dK2 via fd_hessian_bumped (hard max)", hard_big.get(STRIKE, STRIKE), ha_big.get(STRIKE, STRIKE), 1e-2)
    print("    price: hard max", hard.price, " smoothed", ha.price, " (smoothing shifts the price; use the hard max for price / first order)")

    # ── MC ────────────────────────────────────────────────────────────
    var n_steps = 52
    var n_paths = 200_000 // 8
    var z = unsafe_alloc[Float32](n_paths * n_steps * 8)
    std.random.seed(42)
    std.random.randn[DType.float32](z, n_paths * n_steps * 8)
    print("MC 200k x 52, softplus-smoothed payoff eps=0.5")
    var hm = mc_hessian[GbmCallSmooth, 8, DType.float64](p, n_steps, n_paths, z)
    var hm_tiny = mc_hessian_bumped[GbmCallSmooth, 8, DType.float64](p, n_steps, n_paths, z, 1e-4)
    show("  AD Hessian", hm)
    check_le(failures, "exact: max rel vs FD of AAD gradient (h=1e-4)", max_rel_vs(hm, hm_tiny, 1e-6, False), 1e-5)
    check_le(failures, "symmetry", max_asym(hm), 1e-9)
    # 4 single-run standard errors (measured over 8 seeds): gamma 1.4e-4, vanna 3.1e-3
    check_abs(failures, "gamma vs Black-Scholes (4 SE)", hm.get(S0, S0), bsh[0], 6e-4)
    check_abs(failures, "vanna vs Black-Scholes (4 SE)", hm.get(S0, SIG), bsh[2], 1.3e-2)
    var hk = mc_hessian[GbmCall, 8, DType.float64](p, n_steps, n_paths, z)
    var hkb = mc_hessian_bumped[GbmCall, 8, DType.float64](p, n_steps, n_paths, z, 1e-2)
    print("    kinked payoff: AD (pathwise) gamma", hk.get(S0, S0), "<- misses the Dirac term at the strike")
    check_abs(failures, "kinked payoff: gamma via mc_hessian_bumped (1%)", hkb.get(S0, S0), bsh[0], 1.5e-3)
    z.unsafe_free()

    if failures > 0:
        abort("test_second_order FAILED")
    print("test_second_order PASSED")
