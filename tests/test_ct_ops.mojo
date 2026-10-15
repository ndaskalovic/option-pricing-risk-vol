"""Every supported operation: compiled reverse-mode == tape, primal == tape (f64, random inputs).

Run from repo root: mojo run -I . tests/test_ct_ops.mojo
"""
from std.os import abort
from autodiff import Number, Tape
from autodiff.ct import ADNum, Kernel, Compiled, TapeNum, Vec, Dual, maximum, minimum
import std.random

comptime W = 4
comptime DT = DType.float64
comptime V = SIMD[DT, W]
comptime RV = Vec[DT, W]
comptime RD = Dual[DT, W]


def vjp_vec(xs: Array[V, 4], zs: Array[V, 1], seed: Array[V, 3], mut a: Array[V, 4]) -> Array[V, 3]:
    """First-order compiled vjp on plain SIMD arrays (via the Vec value type)."""
    comptime C = Compiled[AllOps]
    var x = Array[RV, 4](fill=RV.splat(0)); var d = Array[RV, 1](fill=RV.splat(0))
    var sy = Array[RV, 3](fill=RV.splat(0)); var ax = Array[RV, 4](fill=RV.splat(0))
    for i in range(4): x[i] = RV(xs[i])
    d[0] = RV(zs[0])
    for i in range(3): sy[i] = RV(seed[i])
    var y = C.vjp[RV](x, d, sy, ax)
    for i in range(4): a[i] = ax[i].v
    var out = Array[V, 3](fill=0)
    for i in range(3): out[i] = y[i].v
    return out^


# One kernel exercising every operation, with shared subexpressions and an unused branch.
struct AllOps(Kernel):
    comptime NI = 4
    comptime ND = 1
    comptime NO = 3

    @staticmethod
    def eval[N: ADNum](x: Array[N, Self.NI], d: Array[N, Self.ND]) -> Array[N, Self.NO]:
        var a = x[0]; var b = x[1]; var c = x[2]; var e = x[3]; var z = d[0]
        var s = (a * b + c / e - a / 3.0 + 2.0 * b - (1.5 - c)) * z
        var t = (a.exp() + b.log() + c.sqrt() + (e ** 1.7) + (-a) + (1.0 / b) + (a - 0.3).relu())
        var u = maximum(a, b) * minimum(c, e) + N.select_gt(z, a.constant(0.0), a * c, b + e)
        _ = (a * b).exp()   # not an output: must not cost anything
        var y = Array[N, Self.NO](fill=a)
        y[0] = s * t
        y[1] = u / (t + 3.0) + s
        y[2] = (s + u).relu() + 0.25 * t + a.detach() * b * c.detach()
        return y^


def main() raises:
    comptime C = Compiled[AllOps]
    print("AllOps: nodes", C.N, "residuals", C.NRES)
    std.random.seed(3)
    var failures = 0
    for trial in range(20):
        var xs = Array[V, 4](fill=0)
        for i in range(4):
            for l in range(W):
                xs[i][l] = 0.2 + 1.5 * std.random.random_float64()    # positive (log/sqrt/pow)
        var zs = Array[V, 1](fill=0)
        for l in range(W):
            zs[0][l] = std.random.randn_float64()
        var seed = Array[V, 3](fill=0)
        seed[0] = V(0.7); seed[1] = V(-1.3); seed[2] = V(0.4)

        # compiled
        var a_c = Array[V, 4](fill=0)
        var y_c = vjp_vec(xs, zs, seed, a_c)

        # tape oracle running the same source: y = sum_k seed_k * out_k, one backward pass
        var tape = Tape[W, DT]()
        var xt = Array[TapeNum[W, DT], 4](fill=TapeNum[W, DT](Number[W, DT](0, tape)))
        for i in range(4):
            xt[i] = TapeNum[W, DT](Number[W, DT](xs[i], tape))
        var zt = Array[TapeNum[W, DT], 1](fill=TapeNum[W, DT](Number[W, DT](zs[0], tape)))
        var yt = AllOps.eval[TapeNum[W, DT]](xt, zt)
        var obj = yt[0] * seed[0][0] + yt[1] * seed[1][0] + yt[2] * seed[2][0]
        obj.n.propagate_adjoints()
        for k in range(3):
            if (y_c[k] - yt[k].n.value).__abs__().reduce_max() > 1e-12 * (1 + abs(y_c[k]).reduce_max()):
                print("primal mismatch out", k, y_c[k], yt[k].n.value); failures += 1
        for i in range(4):
            var g_t = xt[i].n.grad()
            if (a_c[i] - g_t).__abs__().reduce_max() > 1e-11 * (1 + abs(g_t).reduce_max()):
                print("grad mismatch x", i, "trial", trial, a_c[i], g_t); failures += 1
        tape.free()

        # second order: Dual tangents through the compiled reverse sweep (forward-over-reverse)
        # == central differences of the compiled gradient; and the Hessian is symmetric.
        # Output 2 holds the `detach` term, whose second derivatives follow stop-gradient
        # semantics (finite differences of the gradient do not), so it gets a zero seed here.
        var seed2 = seed.copy()
        seed2[2] = V(0)
        var H = Array[V, 16](fill=0)
        for j in range(4):
            var xd = Array[RD, 4](fill=RD.splat(0)); var dd = Array[RD, 1](fill=RD.splat(0))
            var sd = Array[RD, 3](fill=RD.splat(0)); var ad = Array[RD, 4](fill=RD.splat(0))
            for i in range(4):
                xd[i] = RD(xs[i], V(1.0) if i == j else V(0.0))
            dd[0] = RD(zs[0], V(0))
            for i in range(3): sd[i] = RD(seed2[i], V(0))
            _ = C.vjp[RD](xd, dd, sd, ad)
            var hstep = V(1e-6) * xs[j]
            var xp = xs.copy(); xp[j] += hstep
            var xm = xs.copy(); xm[j] -= hstep
            var gp = Array[V, 4](fill=0); var gm = Array[V, 4](fill=0)
            _ = vjp_vec(xp, zs, seed2, gp)
            _ = vjp_vec(xm, zs, seed2, gm)
            for i in range(4):
                var fdv = (gp[i] - gm[i]) / (2 * hstep)
                H[i * 4 + j] = ad[i].t
                if (ad[i].t - fdv).__abs__().reduce_max() > 2e-5 * (1 + abs(fdv).reduce_max()):
                    print("hessian mismatch", i, j, "trial", trial, ad[i].t, fdv); failures += 1
        for i in range(4):
            for j in range(i + 1, 4):
                if (H[i * 4 + j] - H[j * 4 + i]).__abs__().reduce_max() > 1e-10 * (1 + abs(H[i * 4 + j]).reduce_max()):
                    print("hessian asymmetric", i, j); failures += 1
    if failures > 0:
        abort("test_ct_ops FAILED")
    print("test_ct_ops PASSED (20 random trials x 4 lanes, all ops; first order == tape, Hessian == FD of gradient and symmetric)")
