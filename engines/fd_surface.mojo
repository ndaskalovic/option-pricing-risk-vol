"""FD surface engine: one solve gives the solution at every maturity, and (for a unit-strike model
on x = log(S/K)) every strike.

Same PDE, operator and scheme as `fd.mojo` (theta scheme, M = I - th dt L LU-factorised per time
block, cell-averaged payoff, exercise after each solve, V_xx = 0 boundaries). Differences:
- The time grid is a list of blocks so that every maturity lands exactly on a step. The first block
  can start with Rannacher steps (implicit half-steps) that damp the payoff kink, which matters for
  short maturities.
- The solution u and its time derivative u_tau = L u (0 where exercised) are saved at every
  maturity instead of being interpolated once at the spot. The model's `spot` and `maturity` are
  unused; the time grid is data, not differentiated.
The forward and reverse steps are `fd._Scheme`'s, shared with the single-option engine.
Value type R: `Vec` gives values; `Dual` gives values plus one forward tangent per lane at every
grid node, so one pass yields e.g. the vega of every option (and rho in another lane).
`fd_surface_book` adds a reverse sweep for a weighted sum of surface values (a book), giving its
gradient with respect to every model parameter in one sweep.
For a unit-strike model, V(S, K, T) = K u(log(S/K), T); `FdSurface.option` maps a read to price
and grid Greeks.
"""
from std.math import ceil, log
from std.os import abort
from autodiff import Real
from .fd import FdModel, _Buf, _Scheme


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
    sum_k w_k u(x_k, T_{m_k}) into a_p."""
    comptime W = R.W
    comptime DT = R.DT
    var sc = _Scheme[M, R](p, x_lo, x_hi, nx, grads)
    var bl = _schedule(maturities, dt_target, rannacher)
    var last = bl[len(bl) - 1]
    var Ys = _Buf[R]((last.level0 + last.steps + 1) * nx if grads else 1)
    var surf = FdSurface[R](x_lo, sc.h, nx, len(maturities))
    comptime if grads:
        for i in range(nx):
            Ys[i] = sc.cur[i]

    # forward: per block factorise once, step, save u and u_tau at each maturity
    for bi in range(len(bl)):
        var b = bl[bi]
        sc.factor(R.splat(b.dt), b.th)
        for s in range(b.steps):
            sc.step[grads](Ys, (b.level0 + s + 1) * nx)
        if b.mat >= 0:
            for i in range(nx):
                var c = sc.cur[i]
                surf.slices[b.mat * nx + i] = c
                var lu = sc.dd[i] * c
                if i > 0:
                    lu += sc.l[i] * sc.cur[i - 1]
                if i < nx - 1:
                    lu += sc.u[i] * sc.cur[i + 1]
                comptime if M.AMERICAN:
                    lu = R.select_gt(c, sc.intr[i], lu, R.splat(0))
                surf.taus[b.mat * nx + i] = lu

    comptime if grads:
        var jl = List[Int](length=W, fill=0)
        for bb in range(len(bl)):
            var bi = len(bl) - 1 - bb
            var b = bl[bi]
            if bb > 0:
                sc.rescale(R.splat(bl[bi + 1].dt / b.dt))   # see `_Scheme.rescale`
            # seed the book's reads of this maturity (aV is the adjoint of V at the block's last level)
            if b.mat >= 0:
                for pt in points:
                    if pt.m != b.mat:
                        continue
                    var t = surf._weights(pt.x, jl)
                    var w0 = t * (t - 1.0) * 0.5 * pt.w; var w1 = (1.0 - t * t) * pt.w; var w2 = t * (t + 1.0) * 0.5 * pt.w
                    for lane in range(W):
                        var j = jl[lane]
                        var q = sc.aV[j - 1]; q.set_lane(lane, q + R.lift(SIMD[DT, W](w0[lane])), lane); sc.aV[j - 1] = q
                        q = sc.aV[j]; q.set_lane(lane, q + R.lift(SIMD[DT, W](w1[lane])), lane); sc.aV[j] = q
                        q = sc.aV[j + 1]; q.set_lane(lane, q + R.lift(SIMD[DT, W](w2[lane])), lane); sc.aV[j + 1] = q
            sc.factor(R.splat(b.dt), b.th)
            for ss in range(b.steps):
                sc.step_back(Ys, b.level0 + b.steps - 1 - ss, b.th)
        sc.rescale(R.splat(bl[0].dt))
        sc.to_params(p, a_p)

    sc.free(); Ys.free()
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
