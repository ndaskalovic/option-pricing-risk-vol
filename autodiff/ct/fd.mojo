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
from ..number import Number
from ..tape import Tape
from .adnum import ADNum, maximum, smooth_relu
from .trace import Kernel
from .compiled import Compiled
from .tapenum import TapeNum
from .real import Real, Vec, Dual
from .mc import Hessian


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


@fieldwise_init
struct FdResult[NP: Int](Movable, Writable):
    var price: Float64
    var grad: Array[Float64, Self.NP]

    def write_to(self, mut writer: Some[Writer]):
        writer.write("price=", self.price, " grad=[")
        for j in range(Self.NP):
            writer.write(self.grad[j], ", " if j < Self.NP - 1 else "]")


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


def _lane0(v: SIMD) -> Float64:
    return Float64(v[0].cast[DType.float64]())


struct _FdOut[R: Real, NP: Int](Movable):
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


def _fd_core[M: FdModel, R: Real, grads: Bool](
    p: Array[R, M.NP], x_lo: Float64, x_hi: Float64, nx: Int, nt: Int, theta: Float64
) -> _FdOut[R, M.NP]:
    """Theta scheme with M = I - th*dt*L factorised once (LU, no divisions in the loops),
    recurrences carried in registers, only the pre-exercise solution Y stored per level,
    and a fused two-sweep reverse pass per step. Generic over the value type R:
    Vec -> price + gradient; Dual -> also the directional derivative of the gradient
    (forward-over-reverse through every solve)."""
    comptime Coef = Compiled[_CoefK[M]]
    comptime Pay = Compiled[_PayK[M]]
    comptime Spot = Compiled[_SpotK[M]]
    comptime W = R.W
    comptime DT = R.DT
    var none = Array[R, 1](fill=R.splat(0))
    var sm = Spot.eval[R](p, none)
    var x0 = sm[0]
    var dt = sm[1] / Float64(nt)
    var hcell = (x_hi - x_lo) / Float64(nx - 1)
    var th = theta
    var zero = R.splat(0)

    # operator L = (l, dd, u), exercise values, cell-averaged payoff
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

    # explicit operator E = I + (1-th) dt L ; implicit M = I - th dt L = LU (factorised once)
    var es = _Buf[R](nx); var ed = _Buf[R](nx); var eu = _Buf[R](nx); var mc = _Buf[R](nx)
    var inv_m = _Buf[R](nx); var cpr = _Buf[R](nx); var lmul = _Buf[R](nx); var qT = _Buf[R](nx)
    var ew = dt * (1 - th); var iw = dt * th
    for i in range(nx):
        es[i] = ew * l[i]; ed[i] = 1.0 + ew * dd[i]; eu[i] = ew * u[i]
        mc[i] = -(iw * u[i])
    var piv = 1.0 - iw * dd[0]
    inv_m[0] = 1.0 / piv; cpr[0] = mc[0] * inv_m[0]; lmul[0] = zero; qT[0] = zero
    for i in range(1, nx):
        lmul[i] = -(iw * l[i]) * inv_m[i - 1]                # L multiplier (M sub = -iw l)
        piv = (1.0 - iw * dd[i]) - lmul[i] * mc[i - 1]
        inv_m[i] = 1.0 / piv
        cpr[i] = mc[i] * inv_m[i]
        qT[i] = mc[i - 1] * inv_m[i]                         # U^T sweep multiplier

    # forward time stepping; store Y (pre-exercise) per level for the reverse sweep
    var Ys = _Buf[R]((nt + 1) * nx if grads else 1)
    var dp = _Buf[R](nx)
    comptime if grads:
        for i in range(nx):
            Ys[i] = cur[i]
    for n in range(nt):
        # rhs = E V and forward elimination (unit-lower L), stored scaled by 1/m off the chain
        var vprev = zero; var vcur = cur[0]; var dprev = zero
        for i in range(nx):
            var vnext = cur[i + 1] if i < nx - 1 else zero
            var rhs = es[i] * vprev + ed[i] * vcur + eu[i] * vnext
            var d = rhs - lmul[i] * dprev
            dp[i] = d * inv_m[i]
            dprev = d; vprev = vcur; vcur = vnext
        # back substitution: y_i = d'_i/m_i - c'_i y_{i+1}  (one FMA on the chain)
        var ynext = zero
        var base = (n + 1) * nx
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

    # price: quadratic interpolation at x0 around the nearest node (cell chosen per lane by value)
    var xj = SIMD[DT, W](0)
    var jl = List[Int](length=W, fill=0)
    var vm = zero; var vc = zero; var vp = zero
    for lane in range(W):
        var xl = Float64(x0.value()[lane].cast[DType.float64]())
        var j = min(max(Int((xl - x_lo) / hcell + 0.5), 1), nx - 2)
        jl[lane] = j
        xj[lane] = Scalar[DT](x_lo + Float64(j) * hcell)
        vm.set_lane(lane, cur[j - 1], lane); vc.set_lane(lane, cur[j], lane); vp.set_lane(lane, cur[j + 1], lane)
    var t = (x0 - R.lift(xj)) / hcell
    var w0 = t * (t - 1.0) * 0.5; var w1 = 1.0 - t * t; var w2 = t * (t + 1.0) * 0.5
    var price = w0 * vm + w1 * vc + w2 * vp

    var a_p = Array[R, M.NP](fill=R.splat(0))
    comptime if grads:
        var a_x0 = ((t - 0.5) * vm - 2.0 * t * vc + (t + 0.5) * vp) / hcell
        var aV = _Buf[R](nx)
        for lane in range(W):
            var j = jl[lane]
            var q = aV[j - 1]; q.set_lane(lane, q + w0, lane); aV[j - 1] = q
            q = aV[j]; q.set_lane(lane, q + w1, lane); aV[j] = q
            q = aV[j + 1]; q.set_lane(lane, q + w2, lane); aV[j + 1] = q

        # Reverse sweep. Per step n (level n+1 -> n), with aY = mask * aV (American):
        #   M^T a_rhs = aY  via  U^T w = aY (forward), L^T a_rhs = w (backward)
        #   operator adjoints (without the dt factor, applied at the end):
        #     aL_{i,j} += a_rhs_i * (th * Y^{n+1}_j + (1-th) * V^n_j)
        #   aV^n = E^T a_rhs
        var al = _Buf[R](nx); var ad = _Buf[R](nx); var au = _Buf[R](nx); var aintr = _Buf[R](nx)
        var wv = _Buf[R](nx)
        for nn in range(nt):
            var n = nt - 1 - nn
            var base1 = (n + 1) * nx; var base0 = n * nx
            # U^T w = aY (forward):  w_i = aY_i/m_i - qT_i w_{i-1}
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

            # L^T a_rhs = w (backward): r_i = w_i - lmul_{i+1} r_{i+1}; fused operator adjoints
            # and aV_{i+1} = ed_{i+1} r_{i+1} + eu_i r_i + es_{i+2} r_{i+2} (lagged by one node)
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

        # dt-scaled operator adjoints; a_dt = sum_i (al l + ad d + au u) (unscaled sums)
        var a_dt = zero
        for i in range(nx):
            a_dt += al[i] * l[i] + ad[i] * dd[i] + au[i] * u[i]
            al[i] = al[i] * dt; ad[i] = ad[i] * dt; au[i] = au[i] * dt

        # back to parameters: payoff (initial values and exercise values), coefficients, spot, maturity
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
        var asm = Array[R, 2](fill=zero)
        asm[0] = a_x0; asm[1] = a_dt / Float64(nt)
        _ = Spot.vjp[R](p, none, asm, a_p)
        al.free(); ad.free(); au.free(); aintr.free(); wv.free(); aV.free()

    l.free(); dd.free(); u.free(); intr.free(); cur.free(); es.free(); ed.free(); eu.free()
    mc.free(); inv_m.free(); cpr.free(); lmul.free(); qT.free(); Ys.free(); dp.free()
    return _FdOut[R, M.NP](price, a_p^)


struct FdLanes[DT: DType, W: Int, NP: Int](Movable):
    """Per-lane price and gradient (SIMD lanes = independent problems)."""
    var price: SIMD[Self.DT, Self.W]
    var grad: Array[SIMD[Self.DT, Self.W], Self.NP]

    def __init__(out self, price: SIMD[Self.DT, Self.W], var grad: Array[SIMD[Self.DT, Self.W], Self.NP]):
        self.price = price
        self.grad = grad^


def _params[NP: Int, R: Real](params: Array[SIMD[R.DT, R.W], NP]) -> Array[R, NP]:
    var p = Array[R, NP](fill=R.splat(0))
    comptime for j in range(NP):
        p[j] = R.lift(params[j])
    return p^


def _broadcast[NP: Int, W: Int, DT: DType](params: Array[Float64, NP]) -> Array[SIMD[DT, W], NP]:
    var p = Array[SIMD[DT, W], NP](fill=0)
    comptime for j in range(NP):
        p[j] = SIMD[DT, W](params[j])
    return p^


def fd_batch[M: FdModel, W: Int, DT: DType](
    params: Array[SIMD[DT, W], M.NP], x_lo: Float64, x_hi: Float64, nx: Int, nt: Int, theta: Float64 = 0.5
) -> FdLanes[DT, W, M.NP]:
    """W independent problems at once (one parameter set per lane, shared grid): per-lane price + gradient."""
    comptime R = Vec[DT, W]
    var r = _fd_core[M, R, True](_params[M.NP, R](params), x_lo, x_hi, nx, nt, theta)
    var g = Array[SIMD[DT, W], M.NP](fill=0)
    comptime for j in range(M.NP):
        g[j] = r.grad[j].value()
    return FdLanes[DT, W, M.NP](r.price.value(), g^)


def fd_compiled[M: FdModel, W: Int, DT: DType](
    params: Array[Float64, M.NP], x_lo: Float64, x_hi: Float64, nx: Int, nt: Int, theta: Float64 = 0.5
) -> FdResult[M.NP]:
    """Price + gradient for one parameter set (all lanes identical; result is lane 0)."""
    var r = fd_batch[M, W, DT](_broadcast[M.NP, W, DT](params), x_lo, x_hi, nx, nt, theta)
    var g = Array[Float64, M.NP](fill=0)
    comptime for j in range(M.NP):
        g[j] = _lane0(r.grad[j])
    return FdResult[M.NP](_lane0(r.price), g^)


def fd_price[M: FdModel, W: Int, DT: DType](
    params: Array[SIMD[DT, W], M.NP], x_lo: Float64, x_hi: Float64, nx: Int, nt: Int, theta: Float64 = 0.5
) -> SIMD[DT, W]:
    """Price only (no history stored, no reverse sweep)."""
    comptime R = Vec[DT, W]
    return _fd_core[M, R, False](_params[M.NP, R](params), x_lo, x_hi, nx, nt, theta).price.value()


@fieldwise_init
struct FdSecond[NP: Int](Movable, Writable):
    """Price, gradient, directional derivative grad.v and Hessian-vector product H v."""
    var price: Float64
    var grad: Array[Float64, Self.NP]
    var dprice: Float64
    var hv: Array[Float64, Self.NP]

    def write_to(self, mut writer: Some[Writer]):
        writer.write("price=", self.price, " dprice=", self.dprice, " hv=[")
        for j in range(Self.NP):
            writer.write(self.hv[j], ", " if j < Self.NP - 1 else "]")


def fd_hvp[M: FdModel, W: Int, DT: DType](
    params: Array[Float64, M.NP], v: Array[Float64, M.NP], x_lo: Float64, x_hi: Float64, nx: Int, nt: Int,
    theta: Float64 = 0.5,
) -> FdSecond[M.NP]:
    """One forward-over-reverse pass through the whole scheme: price, gradient and H v.
    For American exercise the exercise boundary is held at its computed position
    (derivative of the discrete scheme with the exercise mask fixed)."""
    comptime R = Dual[DT, W]
    var p = Array[R, M.NP](fill=R.splat(0))
    comptime for j in range(M.NP):
        p[j] = R(SIMD[DT, W](params[j]), SIMD[DT, W](v[j]))
    var r = _fd_core[M, R, True](p, x_lo, x_hi, nx, nt, theta)
    var g = Array[Float64, M.NP](fill=0)
    var hv = Array[Float64, M.NP](fill=0)
    comptime for j in range(M.NP):
        g[j] = _lane0(r.grad[j].value())
        hv[j] = _lane0(r.grad[j].tangent())
    return FdSecond[M.NP](_lane0(r.price.value()), g^, _lane0(r.price.tangent()), hv^)


def fd_hessian[M: FdModel, W: Int, DT: DType](
    params: Array[Float64, M.NP], x_lo: Float64, x_hi: Float64, nx: Int, nt: Int, theta: Float64 = 0.5
) -> Hessian[M.NP]:
    """Full Hessian: one forward-over-reverse pass per parameter direction (column)."""
    var h = Array[Float64, M.NP * M.NP](fill=0)
    var price = 0.0
    var grad = Array[Float64, M.NP](fill=0)
    for j in range(M.NP):
        var e = Array[Float64, M.NP](fill=0)
        e[j] = 1.0
        var r = fd_hvp[M, W, DT](params, e, x_lo, x_hi, nx, nt, theta)
        for i in range(M.NP):
            h[i * M.NP + j] = r.hv[i]
        price = r.price
        grad = r.grad.copy()
    return Hessian[M.NP](price, grad^, h^)


def fd_hessian_bumped[M: FdModel, W: Int, DT: DType](
    params: Array[Float64, M.NP], x_lo: Float64, x_hi: Float64, nx: Int, nt: Int,
    rel_bump: Float64 = 1e-2, theta: Float64 = 0.5,
) -> Hessian[M.NP]:
    """Central differences of the (exact, first-order) AAD gradient: 2*NP first-order solves.
    Not AD. Use for entries AD cannot see through kinks: d2/dK2 (the strike only enters the
    kinked payoff) and, with a hard exercise max, directions that move the exercise boundary."""
    var h = Array[Float64, M.NP * M.NP](fill=0)
    var base = fd_compiled[M, W, DT](params, x_lo, x_hi, nx, nt, theta)
    for j in range(M.NP):
        var e = rel_bump * max(abs(params[j]), 1e-2)
        var pp = params.copy(); pp[j] += e
        var pm = params.copy(); pm[j] -= e
        var gp = fd_compiled[M, W, DT](pp, x_lo, x_hi, nx, nt, theta)
        var gm = fd_compiled[M, W, DT](pm, x_lo, x_hi, nx, nt, theta)
        for i in range(M.NP):
            h[i * M.NP + j] = (gp.grad[i] - gm.grad[i]) / (2 * e)
    return Hessian[M.NP](base.price, base.grad.copy(), h^)


def fd_generic[M: FdModel, N: ADNum](p: Array[N, M.NP], x_lo: Float64, x_hi: Float64, nx: Int, nt: Int,
                                     theta: Float64, lane_x0: Float64) -> N:
    """The same scheme written once against ADNum (runs on the tape: the oracle).
    `lane_x0` is the numeric spot coordinate (selects the interpolation cell)."""
    var x0 = M.spot[N](p)
    var dt = M.maturity[N](p) / Float64(nt)
    var h = (x_hi - x_lo) / Float64(nx - 1)
    var l = List[N](); var dd = List[N](); var u = List[N](); var intr = List[N]()
    for i in range(nx):
        var xi = x0.constant(x_lo + Float64(i) * h)
        var k = M.coeffs[N](xi, p)
        if i == 0:
            l.append(x0.constant(0)); dd.append(-k[1] / h - k[2]); u.append(k[1] / h)
        elif i == nx - 1:
            l.append(-k[1] / h); dd.append(k[1] / h - k[2]); u.append(x0.constant(0))
        else:
            l.append(k[0] / (h * h) - k[1] / (2 * h)); dd.append(-2.0 * k[0] / (h * h) - k[2]); u.append(k[0] / (h * h) + k[1] / (2 * h))
        intr.append(M.payoff[N](xi, p))
    var Vc = List[N]()
    for i in range(nx):
        var acc = x0.constant(0)
        comptime for g in range(4):
            comptime gx = GAUSS_X[g]
            comptime gw = GAUSS_W[g]
            acc = acc + gw * M.payoff[N](x0.constant(x_lo + Float64(i) * h + gx * h), p)
        Vc.append(acc)
    for _ in range(nt):
        var cp = List[N](); var rp = List[N]()
        for i in range(nx):
            var lv = dd[i] * Vc[i]
            if i > 0: lv = lv + l[i] * Vc[i - 1]
            if i < nx - 1: lv = lv + u[i] * Vc[i + 1]
            var rhs = Vc[i] + (1 - theta) * dt * lv
            var sub = -theta * dt * l[i]; var dia = 1.0 - theta * dt * dd[i]; var sup = -theta * dt * u[i]
            if i == 0:
                cp.append(sup / dia); rp.append(rhs / dia)
            else:
                var m = dia - sub * cp[i - 1]
                cp.append(sup / m); rp.append((rhs - sub * rp[i - 1]) / m)
        var Y = rp.copy()
        for jj in range(1, nx):
            var i = nx - 1 - jj
            Y[i] = rp[i] - cp[i] * Y[i + 1]
        for i in range(nx):
            comptime if M.AMERICAN:
                comptime if M.EXERCISE_EPS == 0.0:
                    Vc[i] = maximum(Y[i], intr[i])
                else:
                    Vc[i] = intr[i] + smooth_relu(Y[i] - intr[i], M.EXERCISE_EPS)
            else:
                Vc[i] = Y[i]
    var f = (lane_x0 - x_lo) / h
    var j = min(max(Int(f + 0.5), 1), nx - 2)
    var t = (x0 - (x_lo + Float64(j) * h)) / h
    return 0.5 * t * (t - 1.0) * Vc[j - 1] + (1.0 - t * t) * Vc[j] + 0.5 * t * (t + 1.0) * Vc[j + 1]


def fd_tape[M: FdModel, W: Int, DT: DType](
    params: Array[Float64, M.NP], x_lo: Float64, x_hi: Float64, nx: Int, nt: Int, theta: Float64 = 0.5
) -> FdResult[M.NP]:
    comptime T = TapeNum[W, DT]
    comptime V = SIMD[DT, W]
    var tape = Tape[W, DT](nx * (nt * 40 + 40) + 1024)
    var p = Array[T, M.NP](fill=T(Number[W, DT](0, tape)))
    comptime for j in range(M.NP):
        p[j] = T(Number[W, DT](V(params[j]), tape))
    var x0v = M.spot[T](p).n.value
    var y = fd_generic[M, T](p, x_lo, x_hi, nx, nt, theta, Float64(x0v[0].cast[DType.float64]()))
    y.n.propagate_adjoints(1)
    var g = Array[Float64, M.NP](fill=0)
    comptime for j in range(M.NP):
        g[j] = _lane0(p[j].n.grad())
    var price = _lane0(y.n.value)
    tape.free()
    return FdResult[M.NP](price, g^)
