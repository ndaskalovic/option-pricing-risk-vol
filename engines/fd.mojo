"""1-D finite-difference engine (theta scheme, log-space grid): price + gradient.

PDE in time-to-maturity tau on x = log S:   V_tau = a(x) V_xx + b(x) V_x - c(x) V,
V(x, 0) = payoff(x). A model is small functions written once against `ADNum`:
    coeffs(x, p) -> (a, b, c)      per grid node
    payoff(x, p) -> V              per grid node (also the exercise value if AMERICAN)
    spot(p)      -> x0             today's log-spot (price is interpolated there)
    maturity(p)  -> T
The engine owns the grid, time stepping, tridiagonal solves, early exercise and a
hand-written reverse sweep (transposed solves); the model functions are compiled
(`Compiled`) and only their small vector-Jacobian products run in the reverse sweep.
Boundaries: V_xx = 0 with one-sided V_x (keeps the system tridiagonal).
The initial condition is the payoff averaged over each grid cell (4-point Gauss), which
smooths the strike kink; early exercise compares with the pointwise payoff.
The grid is fixed (not differentiated); the price and spot sensitivity come from
quadratic interpolation at x0.
SIMD lanes are independent problems (e.g. different parameter sets).
"""
from std.memory.alloc import unsafe_alloc
from autodiff import ADNum, Kernel, Compiled, Real


trait FdModel:
    comptime NP: Int
    comptime AMERICAN: Bool
    comptime EXERCISE_EPS: Float64
    """Width of the smoothed early-exercise max (0 = hard max). The hard max makes the
    discrete price only piecewise smooth in parameters that move the exercise boundary
    (sigma, r, K, T), so second-order Greeks need EXERCISE_EPS > 0; first order does not."""

    @staticmethod
    def coeffs[N: ADNum](x: N, p: Array[N, Self.NP]) -> Array[N, 3]: ...

    @staticmethod
    def payoff[N: ADNum](x: N, p: Array[N, Self.NP]) -> N: ...

    @staticmethod
    def spot[N: ADNum](p: Array[N, Self.NP]) -> N: ...

    @staticmethod
    def maturity[N: ADNum](p: Array[N, Self.NP]) -> N: ...


# ── adapters ────────────────────────────────────────────────────────

struct _CoefK[M: FdModel](Kernel):
    comptime NI = Self.M.NP
    comptime ND = 1          # x (grid coordinate: data, not differentiated)
    comptime NO = 3

    @staticmethod
    def eval[N: ADNum](x: Array[N, Self.NI], d: Array[N, Self.ND]) -> Array[N, Self.NO]:
        var p = Array[N, Self.M.NP](fill=x[0])
        comptime for j in range(Self.M.NP):
            p[j] = x[j]
        return Self.M.coeffs[N](d[0], p)


struct _PayK[M: FdModel](Kernel):
    comptime NI = Self.M.NP
    comptime ND = 1
    comptime NO = 1

    @staticmethod
    def eval[N: ADNum](x: Array[N, Self.NI], d: Array[N, Self.ND]) -> Array[N, Self.NO]:
        var p = Array[N, Self.M.NP](fill=x[0])
        comptime for j in range(Self.M.NP):
            p[j] = x[j]
        return Array[N, 1](fill=Self.M.payoff[N](d[0], p))


struct _SpotK[M: FdModel](Kernel):
    comptime NI = Self.M.NP
    comptime ND = 1
    comptime NO = 2          # spot, maturity

    @staticmethod
    def eval[N: ADNum](x: Array[N, Self.NI], d: Array[N, Self.ND]) -> Array[N, Self.NO]:
        var p = Array[N, Self.M.NP](fill=x[0])
        comptime for j in range(Self.M.NP):
            p[j] = x[j]
        var y = Array[N, 2](fill=x[0])
        y[0] = Self.M.spot[N](p)
        y[1] = Self.M.maturity[N](p)
        return y^



struct _Buf[R: Real](ImplicitlyCopyable, Movable):
    """Plain heap array of runtime values."""
    var p: Pointer[Self.R, MutUntrackedOrigin]

    def __init__(out self, n: Int, fill: Self.R = Self.R.splat(0)):
        self.p = unsafe_alloc[Self.R](max(n, 1))
        for i in range(n):
            self.p[unsafe_offset=i] = fill

    @always_inline
    def __getitem__(self, i: Int) -> Self.R:
        return self.p[unsafe_offset=i]

    @always_inline
    def __setitem__(self, i: Int, v: Self.R):
        self.p[unsafe_offset=i] = v

    def free(self):
        self.p.unsafe_free()


# 4-point Gauss-Legendre on [-1/2, 1/2] (cell averaging)
comptime GAUSS_X = [-0.4305681557970263, -0.16999052179242813, 0.16999052179242813, 0.4305681557970263]
comptime GAUSS_W = [0.17392742256872692, 0.32607257743127307, 0.32607257743127307, 0.17392742256872692]


struct FdResult[R: Real, NP: Int](Movable):
    """Price and gradient per lane (each lane is an independent problem)."""
    var price: Self.R
    var grad: Array[Self.R, Self.NP]

    def __init__(out self, price: Self.R, var grad: Array[Self.R, Self.NP]):
        self.price = price
        self.grad = grad^


@always_inline
def _vmax[R: Real](a: R, b: R) -> R:
    return R.select_gt(a, b, a, b)


@always_inline
def _exercise[R: Real, eps: Float64](y: R, g: R) -> R:
    """max(y, g), or its softplus smoothing g + eps*log(1 + exp((y - g)/eps)) when eps > 0."""
    comptime if eps == 0.0:
        return _vmax(y, g)
    else:
        var d = y - g
        var ad = R.select_gt(d, R.splat(0), d, -d)
        return g + d.relu() + eps * (1.0 + (-ad / eps).exp()).log()


@always_inline
def _exercise_weight[R: Real, eps: Float64](y: R, g: R) -> R:
    """d max(y, g) / dy  (the derivative w.r.t. g is 1 minus this)."""
    comptime if eps == 0.0:
        return R.select_gt(y, g, R.splat(1), R.splat(0))
    else:
        var d = y - g
        var ad = R.select_gt(d, R.splat(0), d, -d)
        var e = (-ad / eps).exp()
        return R.select_gt(d, R.splat(0), 1.0 / (1.0 + e), e / (1.0 + e))


struct _Scheme[M: FdModel, R: Real]:
    """One theta-scheme solve on a fixed grid: operator L = (l, dd, u), exercise values,
    cell-averaged payoff, per-block E and LU(M) factors, and the reverse-sweep buffers.
    `_fd_core` (one maturity) and `fd_surface` (many) drive the same forward and reverse steps.

    Per step (level n -> n+1): rhs = E V, M Y = rhs (unit-lower L, then U with one FMA on the
    chain), V = max(Y, payoff) if American. Only Y is stored. Reverse per step: U^T w = mask*aV,
    L^T a_rhs = w fused with the operator adjoints aL_ij += a_rhs_i (th Y^{n+1}_j + (1-th) V^n_j)
    (accumulated without the dt factor; see `rescale`) and aV^n = E^T a_rhs."""
    comptime Coef = Compiled[_CoefK[Self.M]]
    comptime Pay = Compiled[_PayK[Self.M]]
    var nx: Int
    var x_lo: Float64
    var h: Float64
    var l: _Buf[Self.R]; var dd: _Buf[Self.R]; var u: _Buf[Self.R]; var intr: _Buf[Self.R]; var cur: _Buf[Self.R]
    var es: _Buf[Self.R]; var ed: _Buf[Self.R]; var eu: _Buf[Self.R]
    var inv_m: _Buf[Self.R]; var cpr: _Buf[Self.R]; var lmul: _Buf[Self.R]; var qT: _Buf[Self.R]; var dp: _Buf[Self.R]
    var aV: _Buf[Self.R]; var al: _Buf[Self.R]; var ad: _Buf[Self.R]; var au: _Buf[Self.R]
    var aintr: _Buf[Self.R]; var wv: _Buf[Self.R]

    def __init__(out self, p: Array[Self.R, Self.M.NP], x_lo: Float64, x_hi: Float64, nx: Int, grads: Bool):
        comptime V = Self.R
        var h = (x_hi - x_lo) / Float64(nx - 1)
        var zero = V.splat(0)
        self.nx = nx; self.x_lo = x_lo; self.h = h
        self.l = _Buf[V](nx); self.dd = _Buf[V](nx); self.u = _Buf[V](nx); self.intr = _Buf[V](nx); self.cur = _Buf[V](nx)
        self.es = _Buf[V](nx); self.ed = _Buf[V](nx); self.eu = _Buf[V](nx)
        self.inv_m = _Buf[V](nx); self.cpr = _Buf[V](nx); self.lmul = _Buf[V](nx); self.qT = _Buf[V](nx); self.dp = _Buf[V](nx)
        var na = nx if grads else 1
        self.aV = _Buf[V](na); self.al = _Buf[V](na); self.ad = _Buf[V](na); self.au = _Buf[V](na)
        self.aintr = _Buf[V](na); self.wv = _Buf[V](na)
        for i in range(nx):
            var xi = Array[V, 1](fill=V.splat(x_lo + Float64(i) * h))
            var k = Self.Coef.eval[V](p, xi)
            var a = k[0]; var b = k[1]; var c = k[2]
            if i == 0:
                self.l[i] = zero; self.dd[i] = -b / h - c; self.u[i] = b / h
            elif i == nx - 1:
                self.l[i] = -b / h; self.dd[i] = b / h - c; self.u[i] = zero
            else:
                self.l[i] = a / (h * h) - b / (2 * h)
                self.dd[i] = -2.0 * a / (h * h) - c
                self.u[i] = a / (h * h) + b / (2 * h)
            self.intr[i] = Self.Pay.eval[V](p, xi)[0]
            var acc = V.splat(0)
            comptime for g in range(4):
                comptime gx = GAUSS_X[g]
                comptime gw = GAUSS_W[g]
                var xg = Array[V, 1](fill=V.splat(x_lo + Float64(i) * h + gx * h))
                acc += Self.Pay.eval[V](p, xg)[0] * gw
            self.cur[i] = acc

    def free(self):
        self.l.free(); self.dd.free(); self.u.free(); self.intr.free(); self.cur.free()
        self.es.free(); self.ed.free(); self.eu.free()
        self.inv_m.free(); self.cpr.free(); self.lmul.free(); self.qT.free(); self.dp.free()
        self.aV.free(); self.al.free(); self.ad.free(); self.au.free(); self.aintr.free(); self.wv.free()

    def factor(self, dt: Self.R, th: Float64):
        """E = I + (1-th) dt L and the LU factors of M = I - th dt L (no divisions in the steps)."""
        var nx = self.nx
        var l = self.l; var dd = self.dd; var u = self.u
        var es = self.es; var ed = self.ed; var eu = self.eu
        var inv_m = self.inv_m; var cpr = self.cpr; var lmul = self.lmul; var qT = self.qT
        var ew = dt * (1 - th); var iw = dt * th
        for i in range(nx):
            es[i] = ew * l[i]; ed[i] = 1.0 + ew * dd[i]; eu[i] = ew * u[i]
        var piv = 1.0 - iw * dd[0]
        inv_m[0] = 1.0 / piv; cpr[0] = -(iw * u[0]) * inv_m[0]
        lmul[0] = Self.R.splat(0); qT[0] = Self.R.splat(0)
        for i in range(1, nx):
            var mprev = -(iw * u[i - 1])                     # superdiagonal of M, row i-1
            lmul[i] = -(iw * l[i]) * inv_m[i - 1]
            piv = (1.0 - iw * dd[i]) - lmul[i] * mprev
            inv_m[i] = 1.0 / piv
            cpr[i] = -(iw * u[i]) * inv_m[i]
            qT[i] = mprev * inv_m[i]

    @always_inline
    def step[store: Bool](self, Ys: _Buf[Self.R], base: Int):
        """One step; with `store`, Y (pre-exercise) goes to Ys[base .. base + nx)."""
        var nx = self.nx
        var cur = self.cur; var dp = self.dp; var intr = self.intr
        var es = self.es; var ed = self.ed; var eu = self.eu
        var inv_m = self.inv_m; var cpr = self.cpr; var lmul = self.lmul
        var zero = Self.R.splat(0)
        var vprev = zero; var vcur = cur[0]; var dprev = zero
        for i in range(nx):
            var vnext = cur[i + 1] if i < nx - 1 else zero
            var rhs = es[i] * vprev + ed[i] * vcur + eu[i] * vnext
            var d = rhs - lmul[i] * dprev
            dp[i] = d * inv_m[i]
            dprev = d; vprev = vcur; vcur = vnext
        var ynext = zero
        for jj in range(nx):
            var i = nx - 1 - jj
            var y = dp[i] - cpr[i] * ynext
            comptime if store:
                Ys[base + i] = y
            comptime if Self.M.AMERICAN:
                cur[i] = _exercise[Self.R, Self.M.EXERCISE_EPS](y, intr[i])
            else:
                cur[i] = y
            ynext = y

    @always_inline
    def step_back(self, Ys: _Buf[Self.R], n: Int, th: Float64):
        """Reverse of step n (level n -> n+1): aV (level n+1) -> aV (level n)."""
        var nx = self.nx
        var aV = self.aV; var wv = self.wv; var intr = self.intr; var aintr = self.aintr
        var al = self.al; var ad = self.ad; var au = self.au
        var es = self.es; var ed = self.ed; var eu = self.eu
        var inv_m = self.inv_m; var lmul = self.lmul; var qT = self.qT
        var zero = Self.R.splat(0)
        var base1 = (n + 1) * nx; var base0 = n * nx
        # U^T w = aY (forward), with the exercise mask: w_i = aY_i/m_i - qT_i w_{i-1}
        var wprev = zero
        for i in range(nx):
            var ay = aV[i]
            comptime if Self.M.AMERICAN:
                var wy = _exercise_weight[Self.R, Self.M.EXERCISE_EPS](Ys[base1 + i], intr[i])
                aintr[i] += ay * (1.0 - wy)
                ay = ay * wy
            var w = ay * inv_m[i] - qT[i] * wprev
            wv[i] = w
            wprev = w

        @always_inline
        def zval(j: Int) capturing -> Self.R:
            var vj = Ys[base0 + j]
            comptime if Self.M.AMERICAN:
                if n > 0:   # level 0 is the averaged payoff itself (no exercise applied)
                    vj = _exercise[Self.R, Self.M.EXERCISE_EPS](vj, intr[j])
            return Ys[base1 + j] * th + vj * (1 - th)

        # L^T a_rhs = w (backward): r_i = w_i - lmul_{i+1} r_{i+1}; fused operator adjoints and
        # aV_{i+1} = ed_{i+1} r_{i+1} + eu_i r_i + es_{i+2} r_{i+2} (lagged by one node)
        var r1 = zero; var r2 = zero
        var zp = zero
        var zc = zval(nx - 1)
        for jj in range(nx):
            var i = nx - 1 - jj
            var lm = lmul[i + 1] if i < nx - 1 else zero
            var r = wv[i] - lm * r1
            var zm = zval(i - 1) if i > 0 else zero
            al[i] += r * zm
            ad[i] += r * zc
            au[i] += r * zp
            zp = zc; zc = zm
            if i < nx - 1:
                var esn = es[i + 2] if i < nx - 2 else zero
                aV[i + 1] = ed[i + 1] * r1 + eu[i] * r + esn * r2
            r2 = r1; r1 = r
        aV[0] = ed[0] * r1 + (es[1] * r2 if nx > 1 else zero)

    def operator_dot(self) -> Self.R:
        """sum_i (al l + ad d + au u): the dt adjoint per unit dt, before `rescale`."""
        var s = Self.R.splat(0)
        for i in range(self.nx):
            s += self.al[i] * self.l[i] + self.ad[i] * self.dd[i] + self.au[i] * self.u[i]
        return s

    def rescale(self, f: Self.R):
        """Multiply the accumulated operator adjoints by f. One block: f = dt at the end. Several
        blocks (reverse order): f = dt_b / dt_{b-1} between blocks and dt_first at the end, so that
        sum_b dt_b A_b = dt_0 (A_0 + (dt_1 / dt_0)(A_1 + ...))."""
        for i in range(self.nx):
            self.al[i] = self.al[i] * f; self.ad[i] = self.ad[i] * f; self.au[i] = self.au[i] * f

    def to_params(self, p: Array[Self.R, Self.M.NP], mut a_p: Array[Self.R, Self.M.NP]):
        """Payoff (initial values and exercise values) and coefficient vjps into a_p."""
        comptime V = Self.R
        var h = self.h
        for i in range(self.nx):
            var xi = Array[V, 1](fill=V.splat(self.x_lo + Float64(i) * h))
            comptime for g in range(4):
                comptime gx = GAUSS_X[g]
                comptime gw = GAUSS_W[g]
                var xg = Array[V, 1](fill=V.splat(self.x_lo + Float64(i) * h + gx * h))
                var ayg = Array[V, 1](fill=self.aV[i] * gw)
                _ = Self.Pay.vjp[V](p, xg, ayg, a_p)
            comptime if Self.M.AMERICAN:
                var ay = Array[V, 1](fill=self.aintr[i])
                _ = Self.Pay.vjp[V](p, xi, ay, a_p)
            var al = self.al[i]; var ad = self.ad[i]; var au = self.au[i]
            var ak = Array[V, 3](fill=V.splat(0))
            if i == 0:
                ak[1] = (au - ad) / h; ak[2] = -ad
            elif i == self.nx - 1:
                ak[1] = (ad - al) / h; ak[2] = -ad
            else:
                ak[0] = (al - 2.0 * ad + au) / (h * h)
                ak[1] = (au - al) / (2 * h)
                ak[2] = -ad
            _ = Self.Coef.vjp[V](p, xi, ak, a_p)


def _fd_core[M: FdModel, R: Real, grads: Bool](
    p: Array[R, M.NP], x_lo: Float64, x_hi: Float64, nx: Int, nt: Int, theta: Float64
) -> FdResult[R, M.NP]:
    """nt equal steps to the model's maturity, price interpolated at the model's spot."""
    comptime Spot = Compiled[_SpotK[M]]
    comptime W = R.W
    comptime DT = R.DT
    var none = Array[R, 1](fill=R.splat(0))
    var sm = Spot.eval[R](p, none)
    var x0 = sm[0]
    var dt = sm[1] / Float64(nt)
    var sc = _Scheme[M, R](p, x_lo, x_hi, nx, grads)
    var h = sc.h
    sc.factor(dt, theta)
    var Ys = _Buf[R]((nt + 1) * nx if grads else 1)
    comptime if grads:
        for i in range(nx):
            Ys[i] = sc.cur[i]
    for n in range(nt):
        sc.step[grads](Ys, (n + 1) * nx)

    # price: quadratic interpolation at x0 around the nearest node (cell chosen per lane by value)
    var cur = sc.cur
    var xj = SIMD[DT, W](0)
    var jl = List[Int](length=W, fill=0)
    var vm = R.splat(0); var vc = R.splat(0); var vp = R.splat(0)
    for lane in range(W):
        var xl = Float64(x0.value()[lane].cast[DType.float64]())
        var j = min(max(Int((xl - x_lo) / h + 0.5), 1), nx - 2)
        jl[lane] = j
        xj[lane] = Scalar[DT](x_lo + Float64(j) * h)
        vm.set_lane(lane, cur[j - 1], lane); vc.set_lane(lane, cur[j], lane); vp.set_lane(lane, cur[j + 1], lane)
    var t = (x0 - R.lift(xj)) / h
    var w0 = t * (t - 1.0) * 0.5; var w1 = 1.0 - t * t; var w2 = t * (t + 1.0) * 0.5
    var price = w0 * vm + w1 * vc + w2 * vp

    var a_p = Array[R, M.NP](fill=R.splat(0))
    comptime if grads:
        var a_x0 = ((t - 0.5) * vm - 2.0 * t * vc + (t + 0.5) * vp) / h
        var aV = sc.aV
        for lane in range(W):
            var j = jl[lane]
            var q = aV[j - 1]; q.set_lane(lane, q + w0, lane); aV[j - 1] = q
            q = aV[j]; q.set_lane(lane, q + w1, lane); aV[j] = q
            q = aV[j + 1]; q.set_lane(lane, q + w2, lane); aV[j + 1] = q
        for nn in range(nt):
            sc.step_back(Ys, nt - 1 - nn, theta)
        var a_dt = sc.operator_dot()
        sc.rescale(dt)
        sc.to_params(p, a_p)
        var asm = Array[R, 2](fill=R.splat(0))
        asm[0] = a_x0; asm[1] = a_dt / Float64(nt)
        _ = Spot.vjp[R](p, none, asm, a_p)
    sc.free(); Ys.free()
    return FdResult[R, M.NP](price, a_p^)


def fd_greeks[M: FdModel, R: Real](
    params: Array[R, M.NP], x_lo: Float64, x_hi: Float64, nx: Int, nt: Int, theta: Float64 = 0.5
) -> FdResult[R, M.NP]:
    """Price and gradient per lane, on a shared grid (lanes = independent parameter sets).
    R = Dual (parameters seeded with a direction v) also gives grad . v and H v as tangents."""
    return _fd_core[M, R, True](params, x_lo, x_hi, nx, nt, theta)


def fd_price[M: FdModel, R: Real](
    params: Array[R, M.NP], x_lo: Float64, x_hi: Float64, nx: Int, nt: Int, theta: Float64 = 0.5
) -> R:
    """Price only: no history stored, no reverse sweep."""
    return _fd_core[M, R, False](params, x_lo, x_hi, nx, nt, theta).price
