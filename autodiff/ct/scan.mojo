"""`Scan`: reverse-mode AD through a runtime-length loop  s_{k+1} = K(s_k, p; d_k).

The step kernel `K` is compiled (see `Compiled`); the loop itself runs at runtime.
Forward saves `Compiled[K].NRES` values per step into a caller buffer; backward walks
the steps in reverse, carrying the state adjoint and accumulating the parameter adjoint.
Data for step k (e.g. normal draws) is read from `data + k*ND*W`, element j at `+ j*W`.
"""
from .trace import Kernel
from .compiled import Compiled


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
    def _data[W: Int, DT: DType](data: Pointer[Float32, _], k: Int) -> Array[SIMD[DT, W], Self.K.ND]:
        var d = Array[SIMD[DT, W], Self.K.ND](fill=0)
        comptime for j in range(Self.K.ND):
            d[j] = data.unsafe_load[width=W]((k * Self.K.ND + j) * W).cast[DT]()
        return d^

    @staticmethod
    @always_inline
    def fwd[W: Int, DT: DType](
        s0: Array[SIMD[DT, W], Self.NS], p: Array[SIMD[DT, W], Self.NP],
        data: Pointer[Float32, _], n_steps: Int, res: Pointer[SIMD[DT, W], MutUntrackedOrigin],
    ) -> Array[SIMD[DT, W], Self.NS]:
        comptime assert Self.K.NO == Self.NS, "step kernel must return the state"
        var x = Array[SIMD[DT, W], Self.K.NI](fill=0)
        comptime for j in range(Self.NS):
            x[j] = s0[j]
        comptime for j in range(Self.NP):
            x[Self.NS + j] = p[j]
        for k in range(n_steps):
            var y = Self.C.fwd[W, DT](x, Self._data[W, DT](data, k), res.unsafe_offset(k * Self.NRES))
            comptime for j in range(Self.NS):
                x[j] = y[j]
        var s = Array[SIMD[DT, W], Self.NS](fill=0)
        comptime for j in range(Self.NS):
            s[j] = x[j]
        return s^

    @staticmethod
    @always_inline
    def bwd[W: Int, DT: DType](
        data: Pointer[Float32, _], n_steps: Int, res: Pointer[SIMD[DT, W], MutUntrackedOrigin],
        a_sn: Array[SIMD[DT, W], Self.NS], mut a_p: Array[SIMD[DT, W], Self.NP],
    ) -> Array[SIMD[DT, W], Self.NS]:
        """Returns the adjoint of the initial state; accumulates a_p += d(out)/dp."""
        comptime V = SIMD[DT, W]
        var a_s = a_sn.copy()
        for kk in range(n_steps):
            var k = n_steps - 1 - kk
            var a_x = Array[V, Self.K.NI](fill=V(-0.0))   # -0.0: additive identity, folds away
            var a_y = Array[V, Self.K.NO](fill=0)
            comptime for j in range(Self.NS):
                a_y[j] = a_s[j]
            Self.C.bwd[W, DT](Self._data[W, DT](data, k), res.unsafe_offset(k * Self.NRES), a_y, a_x)
            comptime for j in range(Self.NS):
                a_s[j] = a_x[j]
            comptime for j in range(Self.NP):
                a_p[j] += a_x[Self.NS + j]
        return a_s^
