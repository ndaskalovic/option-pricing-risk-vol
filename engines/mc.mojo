"""Monte Carlo engine: price and gradient w.r.t. every model parameter (pathwise AAD).

A model is four small functions written once against `ADNum`:
    setup(p, n_steps) -> c        once per run    (per-step coefficients)
    init(p)           -> s0       once per run    (initial state)
    step(s, c, z)     -> s'       every step      (z: NZ normal draws)
    payoff(s, p, c)   -> value    every path
Each is compiled (`Compiled`, `Scan`); per path the engine runs the scan forward, a fused
payoff vjp and the scan backward, then reverses `init` and `setup` once.

SIMD lanes are paths. `normals` holds n_paths * n_steps * NZ * W draws: path i starts at
i*n_steps*NZ*W, step k at +k*NZ*W, draw j at +j*W.
Value type R: `Vec` gives price and gradient; `Dual` (parameters seeded with a direction v)
also gives their tangents, grad . v and the Hessian-vector product H v.
"""
from std.memory.alloc import unsafe_alloc
from autodiff import ADNum, Kernel, Compiled, Scan, Real


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


struct McResult[NP: Int](Movable, Writable):
    """Path means of the price and gradient, and their tangents (zero for `Vec`; for `Dual`
    seeded with direction v: price_tangent = grad . v, grad_tangent = H v)."""
    var price: Float64
    var grad: Array[Float64, Self.NP]
    var price_tangent: Float64
    var grad_tangent: Array[Float64, Self.NP]

    def __init__(out self, price: Float64, var grad: Array[Float64, Self.NP],
                 price_tangent: Float64, var grad_tangent: Array[Float64, Self.NP]):
        self.price = price
        self.grad = grad^
        self.price_tangent = price_tangent
        self.grad_tangent = grad_tangent^

    def write_to(self, mut writer: Some[Writer]):
        writer.write("price=", self.price, " grad=[")
        for j in range(Self.NP):
            writer.write(self.grad[j], ", " if j < Self.NP - 1 else "]")


def _mean[DT: DType, W: Int](v: SIMD[DT, W], n_paths: Int) -> Float64:
    return Float64(v.cast[DType.float64]().reduce_add()) / Float64(n_paths * W)


def mc_greeks[M: McModel, R: Real](
    params: Array[R, M.NP], n_steps: Int, n_paths: Int, normals: Pointer[Float32, _]
) -> McResult[M.NP]:
    """Price and gradient (path means) in one pass. `n_paths` counts SIMD paths (R.W scalar
    paths each); every lane of `params` should hold the same value."""
    comptime S = Scan[_StepK[M], M.NS]
    comptime Setup = Compiled[_SetupK[M]]
    comptime Init = Compiled[_InitK[M]]
    comptime Pay = Compiled[_PayoffK[M]]
    var nd = Array[R, 1](fill=R.splat(Float64(n_steps)))
    var none = Array[R, 1](fill=R.splat(0))

    # path-invariant parts: once per run
    var c = Setup.eval[R](params, nd)
    var s0 = Init.eval[R](params, none)

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
            px[M.NS + j] = params[j]
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
    _ = Init.vjp[R](params, none, a_s0, a_p)
    _ = Setup.vjp[R](params, nd, a_c, a_p)

    var g = Array[Float64, M.NP](fill=0)
    var gt = Array[Float64, M.NP](fill=0)
    comptime for j in range(M.NP):
        g[j] = _mean(a_p[j].value(), n_paths)
        gt[j] = _mean(a_p[j].tangent(), n_paths)
    return McResult[M.NP](_mean(acc.value(), n_paths), g^, _mean(acc.tangent(), n_paths), gt^)
