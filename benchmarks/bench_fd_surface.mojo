"""Full American-put chains: 2000 underliers x 20 maturities x 25 strikes = 1M options, single core.

Run from repo root: mojo run -I . benchmarks/bench_fd_surface.mojo
One FD solve per underlier (`fd_surface`, unit-strike model on log(S/K)), f64:
  values   Vec, 4 underliers per batch: price, delta, gamma, theta, dV/dK for every option
  Dual     2 underliers x (d/dsigma, d/dr) per batch: the above + vega and rho for every option
  bumped   Vec + 4 bumped solves for vega and rho (for comparison)
  book     Vec, 4 underliers per batch: book value + d/d(r, q, sigma) per underlier (one reverse sweep)
Every output is consumed.
"""
from std.math import exp, log, sqrt, ceil
from autodiff.ct import Vec, Dual, FdSurface, SurfacePoint, fd_surface, fd_surface_book
from models.bs_fd import BsPutMoneyness
import std.random, std.time

comptime F = DType.float64
comptime NU = 2000
comptime NM = 20
comptime NK = 25
comptime RATE = 0.04
comptime M = BsPutMoneyness[True]


struct Market(Movable):
    var S: List[Float64]
    var sig: List[Float64]
    var q: List[Float64]
    var K: List[Float64]      # NU * NK, the same strikes for every maturity
    var pos: List[Float64]    # NU * NM * NK book positions
    var T: List[Float64]
    var order: List[Int]      # underliers sorted by sigma (similar grids per batch)

    def __init__(out self):
        std.random.seed(42)
        self.S = List[Float64](); self.sig = List[Float64](); self.q = List[Float64]()
        self.K = List[Float64](); self.pos = List[Float64](); self.T = List[Float64](); self.order = List[Int]()
        for _ in range(NU):
            var s = min(max(exp(log(80.0) + 1.0 * std.random.randn_float64()), 5.0), 2000.0)
            var v = min(max(exp(log(0.30) + 0.40 * std.random.randn_float64()), 0.10), 1.20)
            var dq = 0.0 if std.random.random_float64() < 0.4 else std.random.random_float64(0.0, 0.05)
            self.S.append(s); self.sig.append(v); self.q.append(dq)
            var inc = 0.5 if s < 25 else (1.0 if s < 100 else (2.5 if s < 250 else 5.0))
            for k in range(NK):
                var m = 0.5 + Float64(k) / Float64(NK - 1)
                self.K.append(max(inc, Float64(Int(s * m / inc + 0.5)) * inc))
        for _ in range(NU * NM * NK):
            self.pos.append(std.random.random_float64(-10.0, 10.0))
        for w in range(1, 7):
            self.T.append(Float64(w) / 52.0)
        for mo in [2, 3, 4, 5, 6, 7, 8, 9, 12, 15, 18, 24, 30, 36]:
            self.T.append(Float64(mo) / 12.0)
        for i in range(NU):
            self.order.append(i)
        for i in range(1, NU):
            var j = i
            while j > 0 and self.sig[self.order[j - 1]] > self.sig[self.order[j]]:
                var t = self.order[j]; self.order[j] = self.order[j - 1]; self.order[j - 1] = t
                j -= 1


def grid(mk: Market, und: List[Int], h_target: Float64) -> Tuple[Float64, Float64, Int]:
    """x = log(S/K) range covering every strike of the batch +- 4.5 sd at the longest maturity."""
    var lo = 0.0; var hi = 0.0; var smax = 0.0
    for u in und:
        lo = min(lo, log(mk.S[u] / mk.K[u * NK + NK - 1])); hi = max(hi, log(mk.S[u] / mk.K[u * NK]))
        smax = max(smax, mk.sig[u])
    var z = 4.5 * smax * sqrt(mk.T[NM - 1])
    var nx = Int(ceil((hi - lo + 2 * z) / h_target)) + 1
    return (lo - z, hi + z, nx)


def _now() -> Int:
    return Int(std.time.perf_counter_ns())


def lanes[W: Int](mk: Market, und: List[Int], f: List[Float64]) -> SIMD[F, W]:
    var v = SIMD[F, W](0)
    for l in range(W):
        v[l] = f[und[l]]
    return v


def run_values[W: Int](mk: Market, h: Float64, dt: Float64) -> Float64:
    comptime R = Vec[F, W]
    var chk = 0.0
    for b in range(NU // W):
        var und = List[Int]()
        for l in range(W):
            und.append(mk.order[b * W + l])
        var g = grid(mk, und, h)
        var p = Array[R, 3](fill=R.splat(0))
        p[0] = R.splat(RATE); p[1] = R(lanes[W](mk, und, mk.q)); p[2] = R(lanes[W](mk, und, mk.sig))
        var s = fd_surface[M, R](p, g[0], g[1], g[2], mk.T, dt)
        var S = lanes[W](mk, und, mk.S)
        for m in range(NM):
            for k in range(NK):
                var K = SIMD[F, W](0)
                for l in range(W):
                    K[l] = mk.K[und[l] * NK + k]
                var o = s.option(m, S, K)
                chk += (o[0].value() + o[1].value() + o[2].value() + o[3].value() + o[4].value()).reduce_add()
        s.free()
    return chk


def run_dual[W: Int](mk: Market, h: Float64, dt: Float64) -> Float64:
    """W lanes = W/2 underliers x (vega lane, rho lane)."""
    comptime R = Dual[F, W]
    var chk = 0.0
    var dsig = SIMD[F, W](0); var dr = SIMD[F, W](0)
    for l in range(W):
        dsig[l] = 1.0 if l % 2 == 0 else 0.0; dr[l] = 1.0 - dsig[l]
    for b in range(NU // (W // 2)):
        var und = List[Int]()
        for l in range(W):
            und.append(mk.order[b * (W // 2) + l // 2])
        var g = grid(mk, und, h)
        var p = Array[R, 3](fill=R.splat(0))
        p[0] = R(SIMD[F, W](RATE), dr)
        p[1] = R(lanes[W](mk, und, mk.q), SIMD[F, W](0))
        p[2] = R(lanes[W](mk, und, mk.sig), dsig)
        var s = fd_surface[M, R](p, g[0], g[1], g[2], mk.T, dt)
        var S = lanes[W](mk, und, mk.S)
        for m in range(NM):
            for k in range(NK):
                var K = SIMD[F, W](0)
                for l in range(W):
                    K[l] = mk.K[und[l] * NK + k]
                var o = s.option(m, S, K)
                # lanes 0, 2: values + vega; lanes 1, 3: rho (values repeated)
                chk += (o[0].value() + o[1].value() + o[2].value() + o[3].value() + o[4].value()
                        + o[0].tangent()).reduce_add()
        s.free()
    return chk


def run_bumped(mk: Market, h: Float64, dt: Float64) -> Float64:
    comptime W = 4
    comptime R = Vec[F, W]
    var chk = 0.0
    for b in range(NU // W):
        var und = List[Int]()
        for l in range(W):
            und.append(mk.order[b * W + l])
        var g = grid(mk, und, h)
        var sig = lanes[W](mk, und, mk.sig)
        var surfs = List[FdSurface[R]]()
        for bi in range(5):
            var p = Array[R, 3](fill=R.splat(0))
            p[0] = R.splat(RATE + (1e-4 if bi == 3 else (-1e-4 if bi == 4 else 0.0)))
            p[1] = R(lanes[W](mk, und, mk.q))
            p[2] = R(sig * (1.01 if bi == 1 else (0.99 if bi == 2 else 1.0)))
            surfs.append(fd_surface[M, R](p, g[0], g[1], g[2], mk.T, dt))
        var S = lanes[W](mk, und, mk.S)
        for m in range(NM):
            for k in range(NK):
                var K = SIMD[F, W](0)
                for l in range(W):
                    K[l] = mk.K[und[l] * NK + k]
                var o = surfs[0].option(m, S, K)
                var vu = surfs[1].option(m, S, K)[0].value(); var vd = surfs[2].option(m, S, K)[0].value()
                var ru = surfs[3].option(m, S, K)[0].value(); var rd = surfs[4].option(m, S, K)[0].value()
                var vega = (vu - vd) / (0.02 * sig); var rho = (ru - rd) / 2e-4
                chk += (o[0].value() + o[1].value() + o[2].value() + o[3].value() + o[4].value() + vega + rho).reduce_add()
        for bi in range(5):
            surfs[bi].free()
    return chk


def run_book(mk: Market, h: Float64, dt: Float64) -> Float64:
    comptime W = 4
    comptime R = Vec[F, W]
    var chk = 0.0
    for b in range(NU // W):
        var und = List[Int]()
        for l in range(W):
            und.append(mk.order[b * W + l])
        var g = grid(mk, und, h)
        var p = Array[R, 3](fill=R.splat(0))
        p[0] = R.splat(RATE); p[1] = R(lanes[W](mk, und, mk.q)); p[2] = R(lanes[W](mk, und, mk.sig))
        var S = lanes[W](mk, und, mk.S)
        var pts = List[SurfacePoint[F, W]]()
        for m in range(NM):
            for k in range(NK):
                var K = SIMD[F, W](0); var w = SIMD[F, W](0)
                for l in range(W):
                    K[l] = mk.K[und[l] * NK + k]; w[l] = mk.pos[und[l] * NM * NK + m * NK + k] * K[l]
                pts.append(SurfacePoint[F, W](m, log(S / K), w))
        var bk = fd_surface_book[M, R](p, g[0], g[1], g[2], mk.T, dt, pts)
        chk += (bk.value.value() + bk.grad[0].value() + bk.grad[1].value() + bk.grad[2].value()).reduce_add()
        bk.surface.free()
    return chk


def widths():
    """Values and Dual at 4 vs 8 lanes (h 0.01): wider batches give no extra throughput."""
    var mk = Market()
    var h = 0.01; var dt = 1.0 / 250
    var b4 = 1e9; var b8 = 1e9; var d4 = 1e9; var d8 = 1e9; var chk = 0.0
    for _ in range(3):
        var t0 = _now(); chk += run_values[4](mk, h, dt)
        var t1 = _now(); chk += run_values[8](mk, h, dt)
        var t2 = _now(); chk += run_dual[4](mk, h, dt)
        var t3 = _now(); chk += run_dual[8](mk, h, dt)
        var t4 = _now()
        b4 = min(b4, Float64(t1 - t0) * 1e-9); b8 = min(b8, Float64(t2 - t1) * 1e-9)
        d4 = min(d4, Float64(t3 - t2) * 1e-9); d8 = min(d8, Float64(t4 - t3) * 1e-9)
    print("SIMD width, h 0.01: values W=4", b4, "s | W=8", b8, "s | Dual W=4 (2 und x 2 dirs)", d4, "s | Dual W=8 (4 und x 2 dirs)", d8, "s | chk", chk)


def main() raises:
    var mk = Market()
    print("2000 underliers x 20 maturities x 25 strikes = 1M American puts; one FD solve per underlier, f64, single core")
    for cfg in [(0.02, 1.0 / 100), (0.01, 1.0 / 250), (0.005, 1.0 / 500)]:
        var h = cfg[0]; var dt = cfg[1]
        var best = List[Float64](length=4, fill=1e9); var chk = 0.0
        for _ in range(3):
            var t0 = _now(); chk += run_values[4](mk, h, dt)
            var t1 = _now(); chk += run_dual[4](mk, h, dt)
            var t2 = _now(); chk += run_bumped(mk, h, dt)
            var t3 = _now(); chk += run_book(mk, h, dt)
            var t4 = _now()
            best[0] = min(best[0], Float64(t1 - t0) * 1e-9); best[1] = min(best[1], Float64(t2 - t1) * 1e-9)
            best[2] = min(best[2], Float64(t3 - t2) * 1e-9); best[3] = min(best[3], Float64(t4 - t3) * 1e-9)
        print("h", h, "dt", dt, "| values (price, delta, gamma, theta, dV/dK)", best[0], "s | + vega, rho via Dual", best[1],
              "s (", best[1] / best[0], "x) | via bumps", best[2], "s (", best[2] / best[0], "x) | book + 3 sensitivities/underlier", best[3],
              "s (", best[3] / best[0], "x) | chk", chk)
    widths()
