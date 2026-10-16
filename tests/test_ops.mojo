"""Every supported operation, compiled forward and reverse, f64, random inputs.

Run from repo root: mojo run -I . tests/test_ops.mojo
1. primal == a plain-SIMD reference of the same formulas
2. dot-product test: <w, J v> (forward mode, Dual) == <v, J^T w> (reverse mode, vjp)
3. reverse gradient == central differences of the primal
4. forward-over-reverse Hessian == central differences of the gradient, and symmetric
5. stop-gradient: detach contributes its value, not its derivative
"""
from std.os import abort
from std.math import exp, log, sqrt
from autodiff import ADNum, Kernel, Compiled, Vec, Dual, maximum, minimum
import std.random

comptime W = 4
comptime DT = DType.float64
comptime V = SIMD[DT, W]
comptime RV = Vec[DT, W]
comptime RD = Dual[DT, W]


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


def reference(x: Array[V, 4], z: V) -> Array[V, 3]:
    var a = x[0]; var b = x[1]; var c = x[2]; var e = x[3]
    var s = (a * b + c / e - a / 3.0 + 2.0 * b - (1.5 - c)) * z
    var t = exp(a) + log(b) + sqrt(c) + e ** 1.7 - a + 1.0 / b + max(a - 0.3, 0)
    var u = max(a, b) * min(c, e) + z.gt(0).select(a * c, b + e)
    var y = Array[V, 3](fill=0)
    y[0] = s * t
    y[1] = u / (t + 3.0) + s
    y[2] = max(s + u, 0) + 0.25 * t + a * b * c
    return y^


struct Detach(Kernel):
    comptime NI = 2
    comptime ND = 1
    comptime NO = 1

    @staticmethod
    def eval[N: ADNum](x: Array[N, Self.NI], d: Array[N, Self.ND]) -> Array[N, Self.NO]:
        return Array[N, 1](fill=x[0].detach() * x[1])


def primal(x: Array[V, 4], z: V) -> Array[V, 3]:
    var xv = Array[RV, 4](fill=RV.splat(0))
    for i in range(4):
        xv[i] = RV(x[i])
    var y = Compiled[AllOps].eval[RV](xv, Array[RV, 1](fill=RV(z)))
    var out = Array[V, 3](fill=0)
    for k in range(3):
        out[k] = y[k].value()
    return out^


def gradient(x: Array[V, 4], z: V, w: Array[V, 3]) -> Array[V, 4]:
    """J^T w by the compiled reverse sweep."""
    var xv = Array[RV, 4](fill=RV.splat(0)); var sw = Array[RV, 3](fill=RV.splat(0))
    var a = Array[RV, 4](fill=RV.splat(0))
    for i in range(4):
        xv[i] = RV(x[i])
    for k in range(3):
        sw[k] = RV(w[k])
    _ = Compiled[AllOps].vjp[RV](xv, Array[RV, 1](fill=RV(z)), sw, a)
    var out = Array[V, 4](fill=0)
    for i in range(4):
        out[i] = a[i].value()
    return out^


def dot3(a: Array[V, 3], b: Array[V, 3]) -> V:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def dot4(a: Array[V, 4], b: Array[V, 4]) -> V:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]


def worst(err: V, scale: V) -> Float64:
    return Float64((abs(err) / (1.0 + abs(scale))).reduce_max())


def main() raises:
    comptime C = Compiled[AllOps]
    print("AllOps: nodes", C.N, "residuals", C.NRES)
    std.random.seed(3)
    var e_primal = 0.0; var e_dot = 0.0; var e_fd = 0.0; var e_hess = 0.0; var e_sym = 0.0
    for _ in range(20):
        var x = Array[V, 4](fill=0); var v = Array[V, 4](fill=0); var w = Array[V, 3](fill=0)
        var z = V(0)
        for l in range(W):
            for i in range(4):
                x[i][l] = 0.2 + 1.5 * std.random.random_float64()    # positive (log/sqrt/pow)
                v[i][l] = std.random.randn_float64()
            for k in range(3):
                w[k][l] = std.random.randn_float64()
            z[l] = std.random.randn_float64()

        # 1. primal
        var y = primal(x, z)
        var yr = reference(x, z)
        for k in range(3):
            e_primal = max(e_primal, worst(y[k] - yr[k], yr[k]))

        # 2. dot-product test: forward tangent J v (Dual eval) against reverse J^T w
        var xd = Array[RD, 4](fill=RD.splat(0))
        for i in range(4):
            xd[i] = RD(x[i], v[i])
        var yd = C.eval[RD](xd, Array[RD, 1](fill=RD(z, V(0))))
        var jv = Array[V, 3](fill=0)
        for k in range(3):
            jv[k] = yd[k].tangent()
        var jtw = gradient(x, z, w)
        e_dot = max(e_dot, worst(dot3(w, jv) - dot4(v, jtw), dot3(w, jv)))

        # 3. gradient vs central differences of the primal; output 2 holds the detach term
        #    (stop-gradient is not what finite differences measure), so it gets weight 0
        var w2 = w.copy()
        w2[2] = V(0)
        var g = gradient(x, z, w2)
        for i in range(4):
            var h = V(1e-6) * x[i]
            var xp = x.copy(); xp[i] += h
            var xm = x.copy(); xm[i] -= h
            var fdv = (dot3(w2, primal(xp, z)) - dot3(w2, primal(xm, z))) / (2 * h)
            e_fd = max(e_fd, worst(g[i] - fdv, fdv))

        # 4. Hessian of <w2, y> by forward-over-reverse vs central differences of the gradient
        var H = Array[V, 16](fill=0)
        for j in range(4):
            var xj = Array[RD, 4](fill=RD.splat(0)); var sd = Array[RD, 3](fill=RD.splat(0))
            var ad = Array[RD, 4](fill=RD.splat(0))
            for i in range(4):
                xj[i] = RD(x[i], V(1.0) if i == j else V(0.0))
            for k in range(3):
                sd[k] = RD(w2[k], V(0))
            _ = C.vjp[RD](xj, Array[RD, 1](fill=RD(z, V(0))), sd, ad)
            var h = V(1e-6) * x[j]
            var xp = x.copy(); xp[j] += h
            var xm = x.copy(); xm[j] -= h
            var gp = gradient(xp, z, w2); var gm = gradient(xm, z, w2)
            for i in range(4):
                var fdv = (gp[i] - gm[i]) / (2 * h)
                H[i * 4 + j] = ad[i].tangent()
                e_hess = max(e_hess, worst(ad[i].tangent() - fdv, fdv))
        for i in range(4):
            for j in range(i + 1, 4):
                e_sym = max(e_sym, worst(H[i * 4 + j] - H[j * 4 + i], H[i * 4 + j]))

    # 5. detach: y = stop(a) * b  =>  dy/da = 0, dy/db = a
    var xs = Array[RV, 2](fill=RV.splat(0))
    xs[0] = RV.splat(1.7); xs[1] = RV.splat(0.3)
    var ag = Array[RV, 2](fill=RV.splat(0))
    _ = Compiled[Detach].vjp[RV](xs, Array[RV, 1](fill=RV.splat(0)), Array[RV, 1](fill=RV.splat(1)), ag)
    var e_detach = Float64(abs(ag[0].value()).reduce_max() + abs(ag[1].value() - 1.7).reduce_max())

    var fails = 0
    var names = ["primal vs reference", "dot test (forward vs reverse)", "gradient vs central differences",
                 "Hessian vs FD of gradient", "Hessian symmetry", "detach (stop-gradient)"]
    var errs = [e_primal, e_dot, e_fd, e_hess, e_sym, e_detach]
    var tols = [1e-13, 1e-13, 1e-6, 2e-5, 1e-12, 0.0]   # FD checks are rounding-limited (h = 1e-6)
    for i in range(6):
        var ok = errs[i] <= tols[i]
        print("   ", names[i], "max err", errs[i], "(tol", tols[i], ")", "OK" if ok else "FAIL")
        if not ok:
            fails += 1
    if fails > 0:
        abort("test_ops FAILED")
    print("test_ops PASSED")
