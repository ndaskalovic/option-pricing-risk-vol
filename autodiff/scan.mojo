"""`Scan`: reverse-mode AD through a runtime-length loop  s_{k+1} = K(s_k, p; d_k).

The step kernel `K` is compiled (see `Compiled`); the loop itself runs at runtime.
Forward saves `Compiled[K].NRES` values per step into a caller buffer; backward walks
the steps in reverse, carrying the state adjoint and accumulating the parameter adjoint.
Data for step k (e.g. normal draws) is read from `data + k*ND*W`, element j at `+ j*W`.
Generic over the value type `R: Real` (Vec: first order; Dual: forward-over-reverse).
"""
from .trace import Kernel
from .compiled import Compiled
from .real import Real


struct Scan[K: Kernel, NS: Int]:
    """K's inputs are [state (NS) ..., params (NP) ...]; its outputs are the next state (NS)."""
    comptime NP = Self.K.NI - Self.NS
    comptime C = Compiled[Self.K]
    comptime NRES = Self.C.NRES

    @staticmethod
    def residual_slots(n_steps: Int) -> Int:
        return max(1, n_steps * Self.NRES)

    @staticmethod
    @always_inline
    def _data[R: Real](data: Pointer[Float32, _], k: Int) -> Array[R, Self.K.ND]:
        var d = Array[R, Self.K.ND](fill=R.splat(0))
        comptime for j in range(Self.K.ND):
            d[j] = R.lift(data.unsafe_load[width=R.W]((k * Self.K.ND + j) * R.W).cast[R.DT]())
        return d^

    @staticmethod
    @always_inline
    def fwd[R: Real](
        s0: Array[R, Self.NS], p: Array[R, Self.NP],
        data: Pointer[Float32, _], n_steps: Int, res: Pointer[R, MutUntrackedOrigin],
    ) -> Array[R, Self.NS]:
        comptime assert Self.K.NO == Self.NS, "step kernel must return the state"
        var x = Array[R, Self.K.NI](fill=R.splat(0))
        comptime for j in range(Self.NS):
            x[j] = s0[j]
        comptime for j in range(Self.NP):
            x[Self.NS + j] = p[j]
        for k in range(n_steps):
            var y = Self.C.fwd[R](x, Self._data[R](data, k), res.unsafe_offset(k * Self.NRES))
            comptime for j in range(Self.NS):
                x[j] = y[j]
        var s = Array[R, Self.NS](fill=R.splat(0))
        comptime for j in range(Self.NS):
            s[j] = x[j]
        return s^

    @staticmethod
    @always_inline
    def bwd[R: Real](
        data: Pointer[Float32, _], n_steps: Int, res: Pointer[R, MutUntrackedOrigin],
        a_sn: Array[R, Self.NS], mut a_p: Array[R, Self.NP],
    ) -> Array[R, Self.NS]:
        """Returns the adjoint of the initial state; accumulates a_p += d(out)/dp."""
        var a_s = a_sn.copy()
        for kk in range(n_steps):
            var k = n_steps - 1 - kk
            var a_x = Array[R, Self.K.NI](fill=R.splat(-0.0))   # -0.0: additive identity, folds away
            var a_y = Array[R, Self.K.NO](fill=R.splat(0))
            comptime for j in range(Self.NS):
                a_y[j] = a_s[j]
            Self.C.bwd[R](Self._data[R](data, k), res.unsafe_offset(k * Self.NRES), a_y, a_x)
            comptime for j in range(Self.NS):
                a_s[j] = a_x[j]
            comptime for j in range(Self.NP):
                a_p[j] += a_x[Self.NS + j]
        return a_s^
