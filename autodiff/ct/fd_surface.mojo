"""FD surface engine: one solve gives the solution at every maturity, and (for a unit-strike model
on x = log(S/K)) every strike.

Same PDE, operator and scheme as `fd.mojo` (theta scheme, M = I - th dt L LU-factorised per time
block, cell-averaged payoff, exercise after each solve, V_xx = 0 boundaries). Differences:
- The time grid is a list of blocks so that every maturity lands exactly on a step. The first block
  can start with Rannacher steps (implicit half-steps) that damp the payoff kink, which matters for
  short maturities.
- The solution u and its time derivative u_tau = L u (0 where exercised) are saved at every
  maturity instead of being
  interpolated once at the spot. The model's `spot` and `maturity` are unused; the time grid is
  data, not differentiated.
Value type R: `Vec` gives values; `Dual` gives values plus one forward tangent per lane at every
grid node, so one pass yields e.g. the vega of every option (and rho in another lane).
`fd_surface_book` adds a reverse sweep for a weighted sum of surface values (a book), giving its
gradient with respect to every model parameter in one sweep.
For a unit-strike model, V(S, K, T) = K u(log(S/K), T); `FdSurface.option` maps a read to price
and grid Greeks.
"""
from std.math import ceil, log
from std.os import abort
from .real import Real
from .compiled import Compiled
from .fd import FdModel, _CoefK, _PayK, _Buf, _exercise, _exercise_weight, GAUSS_X, GAUSS_W


@fieldwise_init
struct _Block(ImplicitlyCopyable, Movable):
    var dt: Float64
    var th: Float64
    var steps: Int
    var level0: Int      # global time level at the start of the block
    var mat: Int         # maturity index reached at the end of the block, or -1


def _schedule(maturities: List[Float64], dt_target: Float64, rannacher: Int) -> List[_Block]:
    """Each maturity segment gets ceil(len / dt_target) equal steps (at least rannacher + 1).
    The first `rannacher` Crank-Nicolson steps are replaced by 2*rannacher implicit half-steps."""
    var bl = List[_Block]()
    var t_prev = 0.0
    var lev = 0
    for m in range(len(maturities)):
        var seg = maturities[m] - t_prev
        if seg <= 0:
            abort("fd_surface: maturities must be positive and strictly increasing")
        var ns = max(rannacher + 1, Int(ceil(seg / dt_target)))
        var dt = seg / Float64(ns)
        var start = 0
        if m == 0 and rannacher > 0:
            bl.append(_Block(dt * 0.5, 1.0, 2 * rannacher, lev, -1))
            lev += 2 * rannacher
            start = rannacher
        bl.append(_Block(dt, 0.5, ns - start, lev, m))
        lev += ns - start
        t_prev = maturities[m]
    return bl^


@always_inline
def _factor[R: Real](
    nx: Int, dt: Float64, th: Float64, l: _Buf[R], dd: _Buf[R], u: _Buf[R],
    es: _Buf[R], ed: _Buf[R], eu: _Buf[R], inv_m: _Buf[R], cpr: _Buf[R], lmul: _Buf[R], qT: _Buf[R],
):
    """E = I + (1-th) dt L and the LU factors of M = I - th dt L (as in `_fd_core`)."""
    var ew = dt * (1 - th); var iw = dt * th
    for i in range(nx):
        es[i] = ew * l[i]; ed[i] = 1.0 + ew * dd[i]; eu[i] = ew * u[i]
    var piv = 1.0 - iw * dd[0]
    inv_m[0] = 1.0 / piv; cpr[0] = -(iw * u[0]) * inv_m[0]; lmul[0] = R.splat(0); qT[0] = R.splat(0)
    for i in range(1, nx):
        var mprev = -(iw * u[i - 1])
        lmul[i] = -(iw * l[i]) * inv_m[i - 1]
        piv = (1.0 - iw * dd[i]) - lmul[i] * mprev
        inv_m[i] = 1.0 / piv
        cpr[i] = -(iw * u[i]) * inv_m[i]
        qT[i] = mprev * inv_m[i]


@fieldwise_init
struct SurfacePoint[DT: DType, W: Int](ImplicitlyCopyable, Movable):
    """A weighted read of the surface: maturity index m, grid coordinate x and weight w per lane."""
    var m: Int
    var x: SIMD[Self.DT, Self.W]
    var w: SIMD[Self.DT, Self.W]


struct FdSurface[R: Real](Movable):
    """u(x, T_m) and u_tau(x, T_m) on the grid for every maturity m."""
    var x_lo: Float64
    var h: Float64
    var nx: Int
    var nm: Int
    var slices: _Buf[Self.R]
    var taus: _Buf[Self.R]       # u_tau = a u_xx + b u_x - c u (discrete L u); 0 on exercised nodes

    def __init__(out self, x_lo: Float64, h: Float64, nx: Int, nm: Int):
        self.x_lo = x_lo; self.h = h; self.nx = nx; self.nm = nm
        self.slices = _Buf[Self.R](nm * nx); self.taus = _Buf[Self.R](nm * nx)

    def free(self):
        self.slices.free(); self.taus.free()

    @always_inline
    def _weights(self, x: SIMD[Self.R.DT, Self.R.W], mut j: List[Int]) -> SIMD[Self.R.DT, Self.R.W]:
        """Nearest interior node per lane (into j) and the offset t = (x - x_j) / h."""
        comptime W = Self.R.W
        var xj = SIMD[Self.R.DT, W](0)
        for lane in range(W):
            var xl = Float64(x[lane].cast[DType.float64]())
            var jj = min(max(Int((xl - self.x_lo) / self.h + 0.5), 2), self.nx - 3)
            j[lane] = jj
            xj[lane] = Scalar[Self.R.DT](self.x_lo + Float64(jj) * self.h)
        return (x - xj) / Scalar[Self.R.DT](self.h)

    def _read(self, buf: _Buf[Self.R], m: Int, x: SIMD[Self.R.DT, Self.R.W]) -> Array[Self.R, 3]:
        """u (quadratic through the nearest 3 nodes), u_x (its slope) and u_xx (second differences
        at the nearest node and its neighbour on x's side, linearly interpolated: O(h^2))."""
        comptime W = Self.R.W
        var j = List[Int](length=W, fill=0)
        var ts = self._weights(x, j)
        var t = Self.R.lift(ts)
        var at = Self.R.lift(abs(ts))
        var base = m * self.nx
        var vm = Self.R.splat(0); var vc = Self.R.splat(0); var vp = Self.R.splat(0)
        var vo = Self.R.splat(0)      # outer node on x's side: j+2 if t >= 0 else j-2
        var vi = Self.R.splat(0)      # inner neighbour on x's side: j+1 or j-1
        for lane in range(W):
            var jl = j[lane]
            vm.set_lane(lane, buf[base + jl - 1], lane)
            vc.set_lane(lane, buf[base + jl], lane)
            vp.set_lane(lane, buf[base + jl + 1], lane)
            var sgn = 1 if ts[lane] >= 0 else -1
            vi.set_lane(lane, buf[base + jl + sgn], lane)
            vo.set_lane(lane, buf[base + jl + 2 * sgn], lane)
        var h2 = self.h * self.h
        var out = Array[Self.R, 3](fill=Self.R.splat(0))
        out[0] = t * (t - 1.0) * 0.5 * vm + (1.0 - t * t) * vc + t * (t + 1.0) * 0.5 * vp
        out[1] = ((t - 0.5) * vm - 2.0 * t * vc + (t + 0.5) * vp) / self.h
        var d2c = (vm - 2.0 * vc + vp) / h2
        var d2n = (vc - 2.0 * vi + vo) / h2
        out[2] = (1.0 - at) * d2c + at * d2n
        return out^

    def read(self, m: Int, x: SIMD[Self.R.DT, Self.R.W]) -> Array[Self.R, 3]:
        """u, u_x, u_xx at maturity index m, per-lane coordinate x (quadratic interpolation)."""
        return self._read(self.slices, m, x)

    def option(self, m: Int, S: SIMD[Self.R.DT, Self.R.W], K: SIMD[Self.R.DT, Self.R.W]) -> Array[Self.R, 5]:
        """Unit-strike model on x = log(S/K): price, delta, gamma, theta (dV/dt, calendar), dV/dK.
        Theta comes from the PDE (u_tau = L u on continuation nodes), so it has no dt error."""
        comptime assert Self.R.DT.is_floating_point()
        var x = log(S / K)
        var r = self._read(self.slices, m, x)
        var rt = self._read(self.taus, m, x)
        var k = Self.R.lift(K); var s = Self.R.lift(S)
        var out = Array[Self.R, 5](fill=Self.R.splat(0))
        out[0] = k * r[0]
        out[1] = k * r[1] / s
        out[2] = k * (r[2] - r[1]) / (s * s)
        out[3] = -(k * rt[0])
        out[4] = r[0] - r[1]
        return out^


def _surface_core[M: FdModel, R: Real, grads: Bool](
    p: Array[R, M.NP], x_lo: Float64, x_hi: Float64, nx: Int, maturities: List[Float64],
    dt_target: Float64, rannacher: Int, points: List[SurfacePoint[R.DT, R.W]], mut a_p: Array[R, M.NP],
) -> FdSurface[R]:
    """Forward solve saving every maturity; with `grads`, also the reverse sweep of
    sum_k w_k u(x_k, T_{m_k}) into a_p (same per-step adjoint as `_fd_core`, per-block dt and theta)."""
    comptime Coef = Compiled[_CoefK[M]]
    comptime Pay = Compiled[_PayK[M]]
    comptime W = R.W
    comptime DT = R.DT
    var hcell = (x_hi - x_lo) / Float64(nx - 1)
    var zero = R.splat(0)
    var nm = len(maturities)

    # operator L = (l, dd, u), exercise values, cell-averaged payoff (as in `_fd_core`)
    var l = _Buf[R](nx); var dd = _Buf[R](nx); var u = _Buf[R](nx); var intr = _Buf[R](nx)
    var cur = _Buf[R](nx)
    for i in range(nx):
        var xi = Array[R, 1](fill=R.splat(x_lo + Float64(i) * hcell))
        var k = Coef.eval[R](p, xi)
        var a = k[0]; var b = k[1]; var c = k[2]
        if i == 0:
            l[i] = zero; dd[i] = -b / hcell - c; u[i] = b / hcell
        elif i == nx - 1:
            l[i] = -b / hcell; dd[i] = b / hcell - c; u[i] = zero
        else:
            l[i] = a / (hcell * hcell) - b / (2 * hcell)
            dd[i] = -2.0 * a / (hcell * hcell) - c
            u[i] = a / (hcell * hcell) + b / (2 * hcell)
        intr[i] = Pay.eval[R](p, xi)[0]
        var acc = R.splat(0)
        comptime for g in range(4):
            comptime gx = GAUSS_X[g]
            comptime gw = GAUSS_W[g]
            var xg = Array[R, 1](fill=R.splat(x_lo + Float64(i) * hcell + gx * hcell))
            acc += Pay.eval[R](p, xg)[0] * gw
        cur[i] = acc

    var bl = _schedule(maturities, dt_target, rannacher)
    var last = bl[len(bl) - 1]
    var nlev = last.level0 + last.steps + 1
    var es = _Buf[R](nx); var ed = _Buf[R](nx); var eu = _Buf[R](nx)
    var inv_m = _Buf[R](nx); var cpr = _Buf[R](nx); var lmul = _Buf[R](nx); var qT = _Buf[R](nx)
    var dp = _Buf[R](nx)
    var Ys = _Buf[R](nlev * nx if grads else 1)
    var surf = FdSurface[R](x_lo, hcell, nx, nm)
    comptime if grads:
        for i in range(nx):
            Ys[i] = cur[i]

    # forward: per block factorise once, step, save the slice at each maturity
    for bi in range(len(bl)):
        var b = bl[bi]
        _factor[R](nx, b.dt, b.th, l, dd, u, es, ed, eu, inv_m, cpr, lmul, qT)
        for s in range(b.steps):
            var vprev = zero; var vcur = cur[0]; var dprev = zero
            for i in range(nx):
                var vnext = cur[i + 1] if i < nx - 1 else zero
                var rhs = es[i] * vprev + ed[i] * vcur + eu[i] * vnext
                var d = rhs - lmul[i] * dprev
                dp[i] = d * inv_m[i]
                dprev = d; vprev = vcur; vcur = vnext
            var ynext = zero
            var base = (b.level0 + s + 1) * nx
            for jj in range(nx):
                var i = nx - 1 - jj
                var y = dp[i] - cpr[i] * ynext
                comptime if grads:
                    Ys[base + i] = y
                comptime if M.AMERICAN:
                    cur[i] = _exercise[R, M.EXERCISE_EPS](y, intr[i])
                else:
                    cur[i] = y
                ynext = y
        if b.mat >= 0:
            for i in range(nx):
                surf.slices[b.mat * nx + i] = cur[i]
                var lu = dd[i] * cur[i]
                if i > 0:
                    lu += l[i] * cur[i - 1]
                if i < nx - 1:
                    lu += u[i] * cur[i + 1]
                comptime if M.AMERICAN:
                    lu = R.select_gt(cur[i], intr[i], lu, zero)
                surf.taus[b.mat * nx + i] = lu

    comptime if grads:
        var aV = _Buf[R](nx)
        var al = _Buf[R](nx); var ad = _Buf[R](nx); var au = _Buf[R](nx); var aintr = _Buf[R](nx)
        var wv = _Buf[R](nx)
        var jl = List[Int](length=W, fill=0)
        for bb in range(len(bl)):
            var bi = len(bl) - 1 - bb
            var b = bl[bi]
            var th = b.th
            # seed the book's reads of this maturity (aV is the adjoint of V at the block's last level)
            if b.mat >= 0:
                for pt in points:
                    if pt.m != b.mat:
                        continue
                    var t = surf._weights(pt.x, jl)
                    var w0 = t * (t - 1.0) * 0.5 * pt.w; var w1 = (1.0 - t * t) * pt.w; var w2 = t * (t + 1.0) * 0.5 * pt.w
                    for lane in range(W):
                        var j = jl[lane]
                        var q = aV[j - 1]; q.set_lane(lane, q + R.lift(SIMD[DT, W](w0[lane])), lane); aV[j - 1] = q
                        q = aV[j]; q.set_lane(lane, q + R.lift(SIMD[DT, W](w1[lane])), lane); aV[j] = q
                        q = aV[j + 1]; q.set_lane(lane, q + R.lift(SIMD[DT, W](w2[lane])), lane); aV[j + 1] = q
            _factor[R](nx, b.dt, th, l, dd, u, es, ed, eu, inv_m, cpr, lmul, qT)
            for ss in range(b.steps):
                var n = b.level0 + b.steps - 1 - ss
                var base1 = (n + 1) * nx; var base0 = n * nx
                # U^T w = aY (forward), with the exercise mask
                var wprev = zero
                for i in range(nx):
                    var ay = aV[i]
                    comptime if M.AMERICAN:
                        var wy = _exercise_weight[R, M.EXERCISE_EPS](Ys[base1 + i], intr[i])
                        aintr[i] += ay * (1.0 - wy)
                        ay = ay * wy
                    var w = ay * inv_m[i] - qT[i] * wprev
                    wv[i] = w
                    wprev = w

                @always_inline
                def zval(j: Int) capturing -> R:
                    var vj = Ys[base0 + j]
                    comptime if M.AMERICAN:
                        if n > 0:   # level 0 is the averaged payoff itself (no exercise applied)
                            vj = _exercise[R, M.EXERCISE_EPS](vj, intr[j])
                    return Ys[base1 + j] * th + vj * (1 - th)

                # L^T a_rhs = w (backward), fused dt-scaled operator adjoints and aV = E^T a_rhs
                var r1 = zero; var r2 = zero
                var zp = zero
                var zc = zval(nx - 1)
                for jj in range(nx):
                    var i = nx - 1 - jj
                    var lm = lmul[i + 1] if i < nx - 1 else zero
                    var r = wv[i] - lm * r1
                    var rd = r * b.dt
                    var zm = zval(i - 1) if i > 0 else zero
                    al[i] += rd * zm
                    ad[i] += rd * zc
                    au[i] += rd * zp
                    zp = zc; zc = zm
                    if i < nx - 1:
                        var esn = es[i + 2] if i < nx - 2 else zero
                        aV[i + 1] = ed[i + 1] * r1 + eu[i] * r + esn * r2
                    r2 = r1; r1 = r
                aV[0] = ed[0] * r1 + (es[1] * r2 if nx > 1 else zero)

        # back to parameters: payoff (initial values and exercise values) and coefficients
        for i in range(nx):
            var xi = Array[R, 1](fill=R.splat(x_lo + Float64(i) * hcell))
            comptime for g in range(4):
                comptime gx = GAUSS_X[g]
                comptime gw = GAUSS_W[g]
                var xg = Array[R, 1](fill=R.splat(x_lo + Float64(i) * hcell + gx * hcell))
                var ayg = Array[R, 1](fill=aV[i] * gw)
                _ = Pay.vjp[R](p, xg, ayg, a_p)
            comptime if M.AMERICAN:
                var ay = Array[R, 1](fill=aintr[i])
                _ = Pay.vjp[R](p, xi, ay, a_p)
            var ak = Array[R, 3](fill=zero)
            if i == 0:
                ak[1] = (au[i] - ad[i]) / hcell; ak[2] = -ad[i]
            elif i == nx - 1:
                ak[1] = (ad[i] - al[i]) / hcell; ak[2] = -ad[i]
            else:
                ak[0] = (al[i] - 2.0 * ad[i] + au[i]) / (hcell * hcell)
                ak[1] = (au[i] - al[i]) / (2 * hcell)
                ak[2] = -ad[i]
            _ = Coef.vjp[R](p, xi, ak, a_p)
        aV.free(); al.free(); ad.free(); au.free(); aintr.free(); wv.free()

    l.free(); dd.free(); u.free(); intr.free(); cur.free(); es.free(); ed.free(); eu.free()
    inv_m.free(); cpr.free(); lmul.free(); qT.free(); dp.free(); Ys.free()
    return surf^


def fd_surface[M: FdModel, R: Real](
    p: Array[R, M.NP], x_lo: Float64, x_hi: Float64, nx: Int, maturities: List[Float64],
    dt_target: Float64, rannacher: Int = 2,
) -> FdSurface[R]:
    """Solution at every maturity in one solve. R = Vec: values; R = Dual: values + tangents
    (one direction per lane). Caller frees the result."""
    var none = List[SurfacePoint[R.DT, R.W]]()
    var a = Array[R, M.NP](fill=R.splat(0))
    return _surface_core[M, R, False](p, x_lo, x_hi, nx, maturities, dt_target, rannacher, none, a)


struct FdBook[R: Real, NP: Int](Movable):
    var value: Self.R                   # sum_k w_k u(x_k, T_{m_k}), per lane
    var grad: Array[Self.R, Self.NP]    # d value / d params, per lane
    var surface: FdSurface[Self.R]

    def __init__(out self, value: Self.R, var grad: Array[Self.R, Self.NP], var surface: FdSurface[Self.R]):
        self.value = value
        self.grad = grad^
        self.surface = surface^


def fd_surface_book[M: FdModel, R: Real](
    p: Array[R, M.NP], x_lo: Float64, x_hi: Float64, nx: Int, maturities: List[Float64],
    dt_target: Float64, points: List[SurfacePoint[R.DT, R.W]], rannacher: Int = 2,
) -> FdBook[R, M.NP]:
    """A book of surface reads, sum_k w_k u(x_k, T_{m_k}), and its gradient w.r.t. every model
    parameter in one reverse sweep (per lane). For a unit-strike model, w_k = position_k * K_k.
    R = Dual gives forward-over-reverse (gradient tangents = Hessian-vector products of the book).
    Caller frees `surface`."""
    var a = Array[R, M.NP](fill=R.splat(0))
    var surf = _surface_core[M, R, True](p, x_lo, x_hi, nx, maturities, dt_target, rannacher, points, a)
    var v = R.splat(0)
    for pt in points:
        v += R.lift(pt.w) * surf.read(pt.m, pt.x)[0]
    return FdBook[R, M.NP](v, a^, surf^)
