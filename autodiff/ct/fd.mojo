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
from .adnum import ADNum, maximum
from .trace import Kernel
from .compiled import Compiled
from .tapenum import TapeNum


trait FdModel:
    comptime NP: Int
    comptime AMERICAN: Bool

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


struct _Buf[DT: DType, W: Int]:
    """Plain heap array of SIMD values."""
    var p: Pointer[SIMD[Self.DT, Self.W], MutUntrackedOrigin]

    def __init__(out self, n: Int, fill: SIMD[Self.DT, Self.W] = 0):
        self.p = unsafe_alloc[SIMD[Self.DT, Self.W]](max(n, 1))
        for i in range(n):
            self.p[unsafe_offset=i] = fill

    @always_inline
    def __getitem__(self, i: Int) -> SIMD[Self.DT, Self.W]:
        return self.p[unsafe_offset=i]

    @always_inline
    def __setitem__(self, i: Int, v: SIMD[Self.DT, Self.W]):
        self.p[unsafe_offset=i] = v

    def free(self):
        self.p.unsafe_free()


# 4-point Gauss-Legendre on [-1/2, 1/2] (cell averaging)
comptime GAUSS_X = [-0.4305681557970263, -0.16999052179242813, 0.16999052179242813, 0.4305681557970263]
comptime GAUSS_W = [0.17392742256872692, 0.32607257743127307, 0.32607257743127307, 0.17392742256872692]


@always_inline
def _quad_weights(t: Float64) -> Tuple[Float64, Float64, Float64, Float64, Float64, Float64]:
    """Quadratic Lagrange weights at offset t (in cells) from the middle node, and their t-derivatives."""
    return (0.5 * t * (t - 1), 1 - t * t, 0.5 * t * (t + 1), t - 0.5, -2 * t, t + 0.5)


def _lane0(v: SIMD) -> Float64:
    return Float64(v[0].cast[DType.float64]())


struct FdLanes[DT: DType, W: Int, NP: Int](Movable):
    """Per-lane price and gradient (SIMD lanes = independent problems)."""
    var price: SIMD[Self.DT, Self.W]
    var grad: Array[SIMD[Self.DT, Self.W], Self.NP]

    def __init__(out self, price: SIMD[Self.DT, Self.W], var grad: Array[SIMD[Self.DT, Self.W], Self.NP]):
        self.price = price
        self.grad = grad^


def _fd_core[M: FdModel, W: Int, DT: DType, grads: Bool](
    p: Array[SIMD[DT, W], M.NP], x_lo: Float64, x_hi: Float64, nx: Int, nt: Int, theta: Float64
) -> FdLanes[DT, W, M.NP]:
    """Theta scheme with M = I - th*dt*L factorised once (LU, no divisions in the loops),
    recurrences carried in registers, only the pre-exercise solution Y stored per level,
    and a fused two-sweep reverse pass per step."""
    comptime assert DT.is_floating_point(), "needs a floating-point dtype"
    comptime V = SIMD[DT, W]
    comptime Coef = Compiled[_CoefK[M]]
    comptime Pay = Compiled[_PayK[M]]
    comptime Spot = Compiled[_SpotK[M]]
    var none = Array[V, 1](fill=0)
    var sm = Spot.eval[W, DT](p, none)
    var x0 = sm[0]
    var dt = sm[1] / V(Float64(nt))
    var hcell = (x_hi - x_lo) / Float64(nx - 1)
    var dx = V(hcell)
    var th = V(theta)

    # operator L = (l, dd, u), exercise values, cell-averaged payoff
    var l = _Buf[DT, W](nx); var dd = _Buf[DT, W](nx); var u = _Buf[DT, W](nx); var intr = _Buf[DT, W](nx)
    var cur = _Buf[DT, W](nx)
    for i in range(nx):
        var xi = Array[V, 1](fill=V(x_lo + Float64(i) * hcell))
        var k = Coef.eval[W, DT](p, xi)
        var a = k[0]; var b = k[1]; var c = k[2]
        if i == 0:
            l[i] = 0; dd[i] = -b / dx - c; u[i] = b / dx
        elif i == nx - 1:
            l[i] = -b / dx; dd[i] = b / dx - c; u[i] = 0
        else:
            l[i] = a / (dx * dx) - b / (2 * dx); dd[i] = -2 * a / (dx * dx) - c; u[i] = a / (dx * dx) + b / (2 * dx)
        intr[i] = Pay.eval[W, DT](p, xi)[0]
        var acc = V(0)
        comptime for g in range(4):
            comptime gx = GAUSS_X[g]
            comptime gw = GAUSS_W[g]
            var xg = Array[V, 1](fill=V(x_lo + Float64(i) * hcell + gx * hcell))
            acc += V(gw) * Pay.eval[W, DT](p, xg)[0]
        cur[i] = acc

    # explicit operator E = I + (1-th) dt L ; implicit M = I - th dt L = LU (factorised once)
    var es = _Buf[DT, W](nx); var ed = _Buf[DT, W](nx); var eu = _Buf[DT, W](nx)
    var ma = _Buf[DT, W](nx); var mc = _Buf[DT, W](nx)          # M sub / sup
    var inv_m = _Buf[DT, W](nx); var cpr = _Buf[DT, W](nx); var lmul = _Buf[DT, W](nx); var qT = _Buf[DT, W](nx)
    var ew = (1 - th) * dt; var iw = th * dt
    for i in range(nx):
        es[i] = ew * l[i]; ed[i] = 1 + ew * dd[i]; eu[i] = ew * u[i]
        ma[i] = -iw * l[i]; mc[i] = -iw * u[i]
    var piv = 1 - iw * dd[0]
    inv_m[0] = 1 / piv; cpr[0] = mc[0] * inv_m[0]; lmul[0] = 0
    for i in range(1, nx):
        lmul[i] = ma[i] * inv_m[i - 1]                      # L multiplier
        piv = (1 - iw * dd[i]) - lmul[i] * mc[i - 1]
        inv_m[i] = 1 / piv
        cpr[i] = mc[i] * inv_m[i]
    qT[0] = 0
    for i in range(1, nx):
        qT[i] = mc[i - 1] * inv_m[i]     # U^T sweep multiplier, so its chain is a single FMA

    # forward time stepping; store Y (pre-exercise) per level for the reverse sweep
    var Ys = _Buf[DT, W]((nt + 1) * nx if grads else 1)
    var dp = _Buf[DT, W](nx)
    comptime if grads:
        for i in range(nx):
            Ys[i] = cur[i]
    for n in range(nt):
        # rhs = E V and forward elimination  d'_i = (rhs_i - lmul_i d'_{i-1})  (unit-lower L)
        var vprev = V(0); var vcur = cur[0]; var dprev = V(0)
        for i in range(nx):
            var vnext = cur[i + 1] if i < nx - 1 else V(0)
            var rhs = es[i] * vprev + ed[i] * vcur + eu[i] * vnext
            var d = rhs - lmul[i] * dprev
            dp[i] = d * inv_m[i]            # scaled off the recurrence chain
            dprev = d; vprev = vcur; vcur = vnext
        # back substitution with U (pivots inv_m, sup mc):  y_i = (d'_i - mc_i y_{i+1}) / m_i
        var ynext = V(0)
        var base = (n + 1) * nx
        for jj in range(nx):
            var i = nx - 1 - jj
            var y = dp[i] - cpr[i] * ynext
            comptime if grads:
                Ys[base + i] = y
            comptime if M.AMERICAN:
                cur[i] = max(y, intr[i])
            else:
                cur[i] = y
            ynext = y

    # price: quadratic interpolation at x0 around the nearest node (per lane)
    var price = V(0); var a_x0 = V(0)
    var aV = _Buf[DT, W](nx)
    for lane in range(W):
        var xl = Float64(x0[lane].cast[DType.float64]())
        var f = (xl - x_lo) / hcell
        var j = min(max(Int(f + 0.5), 1), nx - 2)
        var w = _quad_weights(f - Float64(j))
        var vm = cur[j - 1][lane]; var vc = cur[j][lane]; var vp = cur[j + 1][lane]
        price[lane] = Scalar[DT](w[0]) * vm + Scalar[DT](w[1]) * vc + Scalar[DT](w[2]) * vp
        comptime if grads:
            a_x0[lane] = (Scalar[DT](w[3]) * vm + Scalar[DT](w[4]) * vc + Scalar[DT](w[5]) * vp) / Scalar[DT](hcell)
            var t = aV[j - 1]; t[lane] = Scalar[DT](w[0]); aV[j - 1] = t
            t = aV[j]; t[lane] = Scalar[DT](w[1]); aV[j] = t
            t = aV[j + 1]; t[lane] = Scalar[DT](w[2]); aV[j + 1] = t

    var a_p = Array[V, M.NP](fill=0)
    comptime if grads:
        # Reverse sweep. Per step n (level n+1 -> n), with aY = mask * aV (American):
        #   M^T a_rhs = aY  via  U^T w = aY (forward), L^T a_rhs = w (backward)
        #   operator adjoints (without the dt factor, applied at the end):
        #     aL_{i,j} += a_rhs_i * (th * Y^{n+1}_j + (1-th) * V^n_j)
        #   aV^n = E^T a_rhs
        var al = _Buf[DT, W](nx); var ad = _Buf[DT, W](nx); var au = _Buf[DT, W](nx); var aintr = _Buf[DT, W](nx)
        var wv = _Buf[DT, W](nx)
        var one_m_th = 1 - th
        for nn in range(nt):
            var n = nt - 1 - nn
            var base1 = (n + 1) * nx; var base0 = n * nx
            # U^T w = aY  (forward):  w_i = (aY_i - mc_{i-1} w_{i-1}) / m_i
            var wprev = V(0)
            for i in range(nx):
                var ay = aV[i]
                comptime if M.AMERICAN:
                    var m = Ys[base1 + i].gt(intr[i])
                    aintr[i] += m.select(V(0), ay)
                    ay = m.select(ay, V(0))
                var w = ay * inv_m[i] - qT[i] * wprev
                wv[i] = w
                wprev = w
            # L^T a_rhs = w (backward):  r_i = w_i - lmul_{i+1} r_{i+1}; fused operator adjoints
            # and aV_{i+1} = ed_{i+1} r_{i+1} + eu_i r_i + es_{i+2} r_{i+2} (lagged by one node)
            var r1 = V(0); var r2 = V(0)          # r_{i+1}, r_{i+2}
            # rolling window over the stored levels: z_j = th * Y^{n+1}_j + (1-th) * V^n_j
            @always_inline
            def zval(j: Int) capturing -> V:
                var vj = Ys[base0 + j]
                comptime if M.AMERICAN:
                    if n > 0:   # level 0 is the averaged payoff itself (no exercise applied)
                        vj = max(vj, intr[j])
                return th * Ys[base1 + j] + one_m_th * vj
            var zp = V(0)
            var zc = zval(nx - 1)
            for jj in range(nx):
                var i = nx - 1 - jj
                var lm = lmul[i + 1] if i < nx - 1 else V(0)
                var r = wv[i] - lm * r1
                var zm = zval(i - 1) if i > 0 else V(0)
                # operator adjoints at row i: columns i-1, i, i+1
                al[i] += r * zm
                ad[i] += r * zc
                au[i] += r * zp
                zp = zc; zc = zm
                # aV^n at node i+1 is complete once r_i is known
                if i < nx - 1:
                    var esn = es[i + 2] if i < nx - 2 else V(0)
                    aV[i + 1] = ed[i + 1] * r1 + eu[i] * r + esn * r2
                r2 = r1; r1 = r
            aV[0] = ed[0] * r1 + (es[1] * r2 if nx > 1 else V(0))

        # dt-scaled operator adjoints; a_dt = sum_i (al l + ad d + au u) (unscaled sums)
        var a_dt = V(0)
        for i in range(nx):
            a_dt += al[i] * l[i] + ad[i] * dd[i] + au[i] * u[i]
            al[i] = al[i] * dt; ad[i] = ad[i] * dt; au[i] = au[i] * dt

        # back to parameters: payoff (initial values and exercise values), coefficients, spot, maturity
        for i in range(nx):
            var xi = Array[V, 1](fill=V(x_lo + Float64(i) * hcell))
            comptime for g in range(4):
                comptime gx = GAUSS_X[g]
                comptime gw = GAUSS_W[g]
                var xg = Array[V, 1](fill=V(x_lo + Float64(i) * hcell + gx * hcell))
                var ayg = Array[V, 1](fill=V(gw) * aV[i])
                _ = Pay.vjp[W, DT](p, xg, ayg, a_p)
            comptime if M.AMERICAN:
                var ay = Array[V, 1](fill=aintr[i])
                _ = Pay.vjp[W, DT](p, xi, ay, a_p)
            var ak = Array[V, 3](fill=0)
            if i == 0:
                ak[1] = (-ad[i] + au[i]) / dx; ak[2] = -ad[i]
            elif i == nx - 1:
                ak[1] = (-al[i] + ad[i]) / dx; ak[2] = -ad[i]
            else:
                ak[0] = (al[i] - 2 * ad[i] + au[i]) / (dx * dx)
                ak[1] = (-al[i] + au[i]) / (2 * dx)
                ak[2] = -ad[i]
            _ = Coef.vjp[W, DT](p, xi, ak, a_p)
        var asm = Array[V, 2](fill=0)
        asm[0] = a_x0; asm[1] = a_dt / V(Float64(nt))
        _ = Spot.vjp[W, DT](p, none, asm, a_p)
        al.free(); ad.free(); au.free(); aintr.free(); wv.free()

    l.free(); dd.free(); u.free(); intr.free(); cur.free(); es.free(); ed.free(); eu.free()
    ma.free(); mc.free(); inv_m.free(); cpr.free(); lmul.free(); qT.free(); Ys.free(); dp.free(); aV.free()
    return FdLanes[DT, W, M.NP](price, a_p^)


def _broadcast[NP: Int, W: Int, DT: DType](params: Array[Float64, NP]) -> Array[SIMD[DT, W], NP]:
    var p = Array[SIMD[DT, W], NP](fill=0)
    comptime for j in range(NP):
        p[j] = SIMD[DT, W](params[j])
    return p^


def fd_compiled[M: FdModel, W: Int, DT: DType](
    params: Array[Float64, M.NP], x_lo: Float64, x_hi: Float64, nx: Int, nt: Int, theta: Float64 = 0.5
) -> FdResult[M.NP]:
    """Price + gradient for one parameter set (all lanes identical; result is lane 0)."""
    var r = _fd_core[M, W, DT, True](_broadcast[M.NP, W, DT](params), x_lo, x_hi, nx, nt, theta)
    var g = Array[Float64, M.NP](fill=0)
    comptime for j in range(M.NP):
        g[j] = _lane0(r.grad[j])
    return FdResult[M.NP](_lane0(r.price), g^)


def fd_batch[M: FdModel, W: Int, DT: DType](
    params: Array[SIMD[DT, W], M.NP], x_lo: Float64, x_hi: Float64, nx: Int, nt: Int, theta: Float64 = 0.5
) -> FdLanes[DT, W, M.NP]:
    """W independent problems at once (one parameter set per lane, shared grid): per-lane price + gradient."""
    return _fd_core[M, W, DT, True](params, x_lo, x_hi, nx, nt, theta)


def fd_price[M: FdModel, W: Int, DT: DType](
    params: Array[SIMD[DT, W], M.NP], x_lo: Float64, x_hi: Float64, nx: Int, nt: Int, theta: Float64 = 0.5
) -> SIMD[DT, W]:
    """Price only (no history stored, no reverse sweep)."""
    return _fd_core[M, W, DT, False](params, x_lo, x_hi, nx, nt, theta).price


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
                Vc[i] = maximum(Y[i], intr[i])
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
