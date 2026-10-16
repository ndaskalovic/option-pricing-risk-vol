"""Monte Carlo engine: price + gradient w.r.t. every parameter, from one model source.

A model is four small functions written once against `ADNum`:
    setup(p, [n_steps]) -> c          once per run   (per-step coefficients)
    init(p)             -> s0         once per run   (initial state)
    step(s, c, z)       -> s'         every step     (z: NZ normal draws)
    payoff(s, p, c)     -> value      every path
`mc_compiled` compiles each one at compile time (see `Compiled`, `Scan`);
`mc_tape` runs the same source on the runtime tape (flexible fallback and oracle).

SIMD lanes are independent paths. `normals` holds n_paths * n_steps * NZ * W draws:
path i starts at i*n_steps*NZ*W, step k at +k*NZ*W, draw j at +j*W.
"""
from std.memory.alloc import unsafe_alloc
from autodiff import Number, Tape
from autodiff.ct import ADNum, Kernel, Compiled, Scan, TapeNum, Real, Vec, Dual


trait McModel:
    comptime NP: Int   # differentiable parameters
    comptime NC: Int   # per-step coefficients produced by setup
    comptime NS: Int   # state dimension
    comptime NZ: Int   # normal draws per step

    @staticmethod
    def setup[N: ADNum](p: Array[N, Self.NP], n_steps: N) -> Array[N, Self.NC]: ...

    @staticmethod
    def init[N: ADNum](p: Array[N, Self.NP]) -> Array[N, Self.NS]: ...

    @staticmethod
    def step[N: ADNum](s: Array[N, Self.NS], c: Array[N, Self.NC], z: Array[N, Self.NZ]) -> Array[N, Self.NS]: ...

    @staticmethod
    def payoff[N: ADNum](s: Array[N, Self.NS], p: Array[N, Self.NP], c: Array[N, Self.NC]) -> N: ...


# ── adapters: model functions -> Kernels ─────────────────────────────

struct _SetupK[M: McModel](Kernel):
    comptime NI = Self.M.NP
    comptime ND = 1
    comptime NO = Self.M.NC

    @staticmethod
    def eval[N: ADNum](x: Array[N, Self.NI], d: Array[N, Self.ND]) -> Array[N, Self.NO]:
        var p = Array[N, Self.M.NP](fill=x[0])
        comptime for j in range(Self.M.NP):
            p[j] = x[j]
        return Self.M.setup[N](p, d[0])


struct _InitK[M: McModel](Kernel):
    comptime NI = Self.M.NP
    comptime ND = 1   # unused (keeps arrays non-empty)
    comptime NO = Self.M.NS

    @staticmethod
    def eval[N: ADNum](x: Array[N, Self.NI], d: Array[N, Self.ND]) -> Array[N, Self.NO]:
        var p = Array[N, Self.M.NP](fill=x[0])
        comptime for j in range(Self.M.NP):
            p[j] = x[j]
        return Self.M.init[N](p)


struct _StepK[M: McModel](Kernel):
    comptime NI = Self.M.NS + Self.M.NC
    comptime ND = Self.M.NZ
    comptime NO = Self.M.NS

    @staticmethod
    def eval[N: ADNum](x: Array[N, Self.NI], d: Array[N, Self.ND]) -> Array[N, Self.NO]:
        var s = Array[N, Self.M.NS](fill=x[0])
        var c = Array[N, Self.M.NC](fill=x[0])
        var z = Array[N, Self.M.NZ](fill=d[0])
        comptime for j in range(Self.M.NS):
            s[j] = x[j]
        comptime for j in range(Self.M.NC):
            c[j] = x[Self.M.NS + j]
        comptime for j in range(Self.M.NZ):
            z[j] = d[j]
        return Self.M.step[N](s, c, z)


struct _PayoffK[M: McModel](Kernel):
    comptime NI = Self.M.NS + Self.M.NP + Self.M.NC
    comptime ND = 1   # unused
    comptime NO = 1

    @staticmethod
    def eval[N: ADNum](x: Array[N, Self.NI], d: Array[N, Self.ND]) -> Array[N, Self.NO]:
        var s = Array[N, Self.M.NS](fill=x[0])
        var p = Array[N, Self.M.NP](fill=x[0])
        var c = Array[N, Self.M.NC](fill=x[0])
        comptime for j in range(Self.M.NS):
            s[j] = x[j]
        comptime for j in range(Self.M.NP):
            p[j] = x[Self.M.NS + j]
        comptime for j in range(Self.M.NC):
            c[j] = x[Self.M.NS + Self.M.NP + j]
        var y = Array[N, 1](fill=x[0])
        y[0] = Self.M.payoff[N](s, p, c)
        return y^


@fieldwise_init
struct McResult[NP: Int](Movable, Writable):
    var price: Float64
    var grad: Array[Float64, Self.NP]

    def write_to(self, mut writer: Some[Writer]):
        writer.write("price=", self.price, " grad=[")
        for j in range(Self.NP):
            writer.write(self.grad[j], ", " if j < Self.NP - 1 else "]")


def _lane_mean[W: Int, DT: DType](v: SIMD[DT, W], count: Int) -> Float64:
    return Float64(v.cast[DType.float64]().reduce_add()) / Float64(count * W)


struct McLanes[R: Real, NP: Int](Movable):
    """Per-lane sums over paths of the price and of its parameter adjoints."""
    var price: Self.R
    var grad: Array[Self.R, Self.NP]

    def __init__(out self, price: Self.R, var grad: Array[Self.R, Self.NP]):
        self.price = price
        self.grad = grad^


def _mc_core[M: McModel, R: Real](
    p: Array[R, M.NP], n_steps: Int, n_paths: Int, normals: Pointer[Float32, _]
) -> McLanes[R, M.NP]:
    """Pathwise reverse mode over all paths, generic over the value type R."""
    comptime S = Scan[_StepK[M], M.NS]
    comptime Setup = Compiled[_SetupK[M]]
    comptime Init = Compiled[_InitK[M]]
    comptime Pay = Compiled[_PayoffK[M]]
    var nd = Array[R, 1](fill=R.splat(Float64(n_steps)))
    var none = Array[R, 1](fill=R.splat(0))

    # path-invariant parts: once per run
    var c = Setup.eval[R](p, nd)
    var s0 = Init.eval[R](p, none)

    var res = unsafe_alloc[R](S.residual_slots(n_steps))
    var a_c = Array[R, M.NC](fill=R.splat(0))
    var a_s0 = Array[R, M.NS](fill=R.splat(0))
    var a_p = Array[R, M.NP](fill=R.splat(0))
    var acc = R.splat(0)
    var stride = n_steps * M.NZ * R.W
    for i in range(n_paths):
        var z = normals.unsafe_offset(i * stride)
        var sn = S.fwd[R](s0, c, z, n_steps, res)
        # payoff: fused value + reverse
        var px = Array[R, Pay.K.NI](fill=R.splat(0))
        comptime for j in range(M.NS):
            px[j] = sn[j]
        comptime for j in range(M.NP):
            px[M.NS + j] = p[j]
        comptime for j in range(M.NC):
            px[M.NS + M.NP + j] = c[j]
        var a_px = Array[R, Pay.K.NI](fill=R.splat(-0.0))
        var one = Array[R, 1](fill=R.splat(1))
        var y = Pay.vjp[R](px, none, one, a_px)
        acc += y[0]
        var a_sn = Array[R, M.NS](fill=R.splat(0))
        comptime for j in range(M.NS):
            a_sn[j] = a_px[j]
        comptime for j in range(M.NP):
            a_p[j] += a_px[M.NS + j]
        comptime for j in range(M.NC):
            a_c[j] += a_px[M.NS + M.NP + j]
        var a0 = S.bwd[R](z, n_steps, res, a_sn, a_c)
        comptime for j in range(M.NS):
            a_s0[j] += a0[j]
    res.unsafe_free()

    # reverse through the path-invariant parts once
    _ = Init.vjp[R](p, none, a_s0, a_p)
    _ = Setup.vjp[R](p, nd, a_c, a_p)
    return McLanes[R, M.NP](acc, a_p^)


def mc_compiled[M: McModel, W: Int, DT: DType](
    params: Array[Float64, M.NP], n_steps: Int, n_paths: Int, normals: Pointer[Float32, _]
) -> McResult[M.NP]:
    """Price + gradient. `n_paths` counts SIMD paths (W scalar paths each)."""
    comptime R = Vec[DT, W]
    var p = Array[R, M.NP](fill=R.splat(0))
    comptime for j in range(M.NP):
        p[j] = R.splat(params[j])
    var r = _mc_core[M, R](p, n_steps, n_paths, normals)
    var g = Array[Float64, M.NP](fill=0)
    comptime for j in range(M.NP):
        g[j] = _lane_mean[W, DT](r.grad[j].value(), n_paths)
    return McResult[M.NP](_lane_mean[W, DT](r.price.value(), n_paths), g^)


@fieldwise_init
struct McSecond[NP: Int](Movable, Writable):
    """Price, gradient, directional derivative grad.v and Hessian-vector product H v."""
    var price: Float64
    var grad: Array[Float64, Self.NP]
    var dprice: Float64
    var hv: Array[Float64, Self.NP]

    def write_to(self, mut writer: Some[Writer]):
        writer.write("price=", self.price, " dprice=", self.dprice, " hv=[")
        for j in range(Self.NP):
            writer.write(self.hv[j], ", " if j < Self.NP - 1 else "]")


def mc_hvp[M: McModel, W: Int, DT: DType](
    params: Array[Float64, M.NP], v: Array[Float64, M.NP], n_steps: Int, n_paths: Int,
    normals: Pointer[Float32, _],
) -> McSecond[M.NP]:
    """One forward-over-reverse pass: price, gradient and H v (cost ~2-3x first order).
    Note: pathwise second derivatives of kinked payoffs miss the Dirac term at the kink;
    use a smoothed payoff (see `smooth_relu`) for unbiased-in-the-limit MC second order."""
    comptime R = Dual[DT, W]
    var p = Array[R, M.NP](fill=R.splat(0))
    comptime for j in range(M.NP):
        p[j] = R(SIMD[DT, W](params[j]), SIMD[DT, W](v[j]))
    var r = _mc_core[M, R](p, n_steps, n_paths, normals)
    var g = Array[Float64, M.NP](fill=0)
    var hv = Array[Float64, M.NP](fill=0)
    comptime for j in range(M.NP):
        g[j] = _lane_mean[W, DT](r.grad[j].value(), n_paths)
        hv[j] = _lane_mean[W, DT](r.grad[j].tangent(), n_paths)
    return McSecond[M.NP](_lane_mean[W, DT](r.price.value(), n_paths), g^,
                          _lane_mean[W, DT](r.price.tangent(), n_paths), hv^)


struct Hessian[NP: Int](Movable, Writable):
    var price: Float64
    var grad: Array[Float64, Self.NP]
    var h: Array[Float64, Self.NP * Self.NP]   # row-major: h[i*NP + j] = d2 price / dp_i dp_j

    def __init__(out self, price: Float64, var grad: Array[Float64, Self.NP], var h: Array[Float64, Self.NP * Self.NP]):
        self.price = price
        self.grad = grad^
        self.h = h^

    def get(self, i: Int, j: Int) -> Float64:
        return self.h[i * Self.NP + j]

    def write_to(self, mut writer: Some[Writer]):
        writer.write("price=", self.price, "\n")
        for i in range(Self.NP):
            writer.write("  [")
            for j in range(Self.NP):
                writer.write(self.h[i * Self.NP + j], ", " if j < Self.NP - 1 else "]\n")


def mc_hessian[M: McModel, W: Int, DT: DType](
    params: Array[Float64, M.NP], n_steps: Int, n_paths: Int, normals: Pointer[Float32, _]
) -> Hessian[M.NP]:
    """Full Hessian: one forward-over-reverse pass per parameter direction (column)."""
    var h = Array[Float64, M.NP * M.NP](fill=0)
    var price = 0.0
    var grad = Array[Float64, M.NP](fill=0)
    for j in range(M.NP):
        var e = Array[Float64, M.NP](fill=0)
        e[j] = 1.0
        var r = mc_hvp[M, W, DT](params, e, n_steps, n_paths, normals)
        for i in range(M.NP):
            h[i * M.NP + j] = r.hv[i]
        price = r.price
        grad = r.grad.copy()
    return Hessian[M.NP](price, grad^, h^)


def mc_hessian_bumped[M: McModel, W: Int, DT: DType](
    params: Array[Float64, M.NP], n_steps: Int, n_paths: Int, normals: Pointer[Float32, _],
    rel_bump: Float64 = 1e-2,
) -> Hessian[M.NP]:
    """Central differences of the (exact, first-order) AAD gradient with common random numbers:
    H[:, j] = (grad(p + h e_j) - grad(p - h e_j)) / 2h, h = rel_bump * max(|p_j|, 1e-2).
    Not AD: 2*NP first-order runs. Valid for kinked payoffs (pathwise delta + bump is a standard
    gamma estimator), unlike pathwise second order; bias O(h^2), variance grows as h shrinks."""
    var h = Array[Float64, M.NP * M.NP](fill=0)
    var base = mc_compiled[M, W, DT](params, n_steps, n_paths, normals)
    for j in range(M.NP):
        var e = rel_bump * max(abs(params[j]), 1e-2)
        var pp = params.copy(); pp[j] += e
        var pm = params.copy(); pm[j] -= e
        var gp = mc_compiled[M, W, DT](pp, n_steps, n_paths, normals)
        var gm = mc_compiled[M, W, DT](pm, n_steps, n_paths, normals)
        for i in range(M.NP):
            h[i * M.NP + j] = (gp.grad[i] - gm.grad[i]) / (2 * e)
    return Hessian[M.NP](base.price, base.grad.copy(), h^)


def mc_tape[M: McModel, W: Int, DT: DType](
    params: Array[Float64, M.NP], n_steps: Int, n_paths: Int, normals: Pointer[Float32, _]
) -> McResult[M.NP]:
    """Same model on the runtime tape (pathwise: mark / rewind). Slower; the oracle."""
    comptime T = TapeNum[W, DT]
    comptime V = SIMD[DT, W]
    var tape = Tape[W, DT](n_steps * (16 + 16 * M.NS * M.NZ) + 4096)
    var p = Array[T, M.NP](fill=T(Number[W, DT](0, tape)))
    comptime for j in range(M.NP):
        p[j] = T(Number[W, DT](V(params[j]), tape))
    var n = T(Number[W, DT](V(Float64(n_steps)), tape))
    var c = M.setup[T](p, n)
    var s0 = M.init[T](p)
    tape.set_mark()
    var acc = V(0)
    var stride = n_steps * M.NZ * W
    for i in range(n_paths):
        var zp = normals.unsafe_offset(i * stride)
        var s = s0.copy()
        for k in range(n_steps):
            var z = Array[T, M.NZ](fill=s0[0])
            comptime for j in range(M.NZ):
                z[j] = s0[0].constant(0)
                z[j] = T(Number[W, DT](zp.unsafe_load[width=W]((k * M.NZ + j) * W).cast[DT](), tape))
            s = M.step[T](s, c, z)
        var y = M.payoff[T](s, p, c)
        y.n.propagate_adjoints(1)
        acc += y.n.value
        tape.rewind_to_mark()
    tape.propagate_mark_to_start()
    var g = Array[Float64, M.NP](fill=0)
    comptime for j in range(M.NP):
        g[j] = _lane_mean[W, DT](p[j].n.grad(), n_paths)
    var price = _lane_mean[W, DT](acc, n_paths)
    tape.free()
    return McResult[M.NP](price, g^)
