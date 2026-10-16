"""Runtime value types for compiled kernels and engines.

`Real` is the arithmetic the compiled code and the engines are written against:
    Vec[DT, W]   one SIMD vector (W lanes = W paths / problems): first-order AAD
    Dual[DT, W]  value + one tangent (forward mode); running the reverse sweep on Dual
                 values gives forward-over-reverse, i.e. Hessian-vector products.
Comparisons only exist through `select_gt` (by value), as in `ADNum`.
"""
from std.math import exp, log, sqrt


trait Real(ImplicitlyCopyable, Movable, Deinitable):
    comptime DT: DType
    comptime W: Int

    @staticmethod
    def splat(k: Float64) -> Self:
        """Constant (zero tangent)."""
        ...

    @staticmethod
    def lift(v: SIMD[Self.DT, Self.W]) -> Self:
        """Constant vector (zero tangent)."""
        ...

    def value(self) -> SIMD[Self.DT, Self.W]: ...
    def tangent(self) -> SIMD[Self.DT, Self.W]: ...

    def __add__(self, rhs: Self) -> Self: ...
    def __add__(self, rhs: Float64) -> Self: ...
    def __radd__(self, lhs: Float64) -> Self: ...
    def __sub__(self, rhs: Self) -> Self: ...
    def __sub__(self, rhs: Float64) -> Self: ...
    def __rsub__(self, lhs: Float64) -> Self: ...
    def __mul__(self, rhs: Self) -> Self: ...
    def __mul__(self, rhs: Float64) -> Self: ...
    def __rmul__(self, lhs: Float64) -> Self: ...
    def __truediv__(self, rhs: Self) -> Self: ...
    def __truediv__(self, rhs: Float64) -> Self: ...
    def __rtruediv__(self, lhs: Float64) -> Self: ...
    def __neg__(self) -> Self: ...
    def __iadd__(mut self, rhs: Self): ...
    def __isub__(mut self, rhs: Self): ...
    def exp(self) -> Self: ...
    def log(self) -> Self: ...
    def sqrt(self) -> Self: ...
    def powk(self, k: Float64) -> Self: ...
    def relu(self) -> Self: ...

    @staticmethod
    def select_gt(a: Self, b: Self, x: Self, y: Self) -> Self:
        """`x` where a > b (by value), else `y`."""
        ...

    def set_lane(mut self, lane: Int, src: Self, src_lane: Int):
        """Copy one lane (value and tangent) from `src`."""
        ...


struct Vec[dt: DType, w: Int](Real, TrivialRegisterPassable):
    comptime DT = Self.dt
    comptime W = Self.w
    comptime S = SIMD[Self.dt, Self.w]
    var v: Self.S

    @always_inline
    def __init__(out self, v: Self.S):
        self.v = v

    @staticmethod
    @always_inline
    def splat(k: Float64) -> Self: return Self(Self.S(k))
    @staticmethod
    @always_inline
    def lift(v: SIMD[Self.DT, Self.W]) -> Self: return Self(rebind[Self.S](v))
    @always_inline
    def value(self) -> SIMD[Self.DT, Self.W]: return rebind[SIMD[Self.DT, Self.W]](self.v)
    @always_inline
    def tangent(self) -> SIMD[Self.DT, Self.W]: return SIMD[Self.DT, Self.W](0)

    @always_inline
    def __add__(self, r: Self) -> Self: return Self(self.v + r.v)
    @always_inline
    def __add__(self, r: Float64) -> Self: return Self(self.v + Self.S(r))
    @always_inline
    def __radd__(self, l: Float64) -> Self: return Self(Self.S(l) + self.v)
    @always_inline
    def __sub__(self, r: Self) -> Self: return Self(self.v - r.v)
    @always_inline
    def __sub__(self, r: Float64) -> Self: return Self(self.v - Self.S(r))
    @always_inline
    def __rsub__(self, l: Float64) -> Self: return Self(Self.S(l) - self.v)
    @always_inline
    def __mul__(self, r: Self) -> Self: return Self(self.v * r.v)
    @always_inline
    def __mul__(self, r: Float64) -> Self: return Self(self.v * Self.S(r))
    @always_inline
    def __rmul__(self, l: Float64) -> Self: return Self(Self.S(l) * self.v)
    @always_inline
    def __truediv__(self, r: Self) -> Self: return Self(self.v / r.v)
    @always_inline
    def __truediv__(self, r: Float64) -> Self: return Self(self.v / Self.S(r))
    @always_inline
    def __rtruediv__(self, l: Float64) -> Self: return Self(Self.S(l) / self.v)
    @always_inline
    def __neg__(self) -> Self: return Self(-self.v)
    @always_inline
    def __iadd__(mut self, r: Self): self.v += r.v
    @always_inline
    def __isub__(mut self, r: Self): self.v -= r.v
    @always_inline
    def exp(self) -> Self:
        comptime assert Self.dt.is_floating_point()
        return Self(exp(self.v))
    @always_inline
    def log(self) -> Self:
        comptime assert Self.dt.is_floating_point()
        return Self(log(self.v))
    @always_inline
    def sqrt(self) -> Self:
        comptime assert Self.dt.is_floating_point()
        return Self(sqrt(self.v))
    @always_inline
    def powk(self, k: Float64) -> Self: return Self(self.v ** Self.S(k))
    @always_inline
    def relu(self) -> Self: return Self(max(self.v, Self.S(0)))

    @staticmethod
    @always_inline
    def select_gt(a: Self, b: Self, x: Self, y: Self) -> Self:
        return Self(a.v.gt(b.v).select(x.v, y.v))

    @always_inline
    def set_lane(mut self, lane: Int, src: Self, src_lane: Int):
        self.v[lane] = src.v[src_lane]


struct Dual[dt: DType, w: Int](Real, TrivialRegisterPassable):
    """Value and one tangent direction, lane-wise."""
    comptime DT = Self.dt
    comptime W = Self.w
    comptime S = SIMD[Self.dt, Self.w]
    var v: Self.S
    var t: Self.S

    @always_inline
    def __init__(out self, v: Self.S, t: Self.S):
        self.v = v
        self.t = t

    @staticmethod
    @always_inline
    def splat(k: Float64) -> Self: return Self(Self.S(k), Self.S(0))
    @staticmethod
    @always_inline
    def lift(v: SIMD[Self.DT, Self.W]) -> Self: return Self(rebind[Self.S](v), Self.S(0))
    @always_inline
    def value(self) -> SIMD[Self.DT, Self.W]: return rebind[SIMD[Self.DT, Self.W]](self.v)
    @always_inline
    def tangent(self) -> SIMD[Self.DT, Self.W]: return rebind[SIMD[Self.DT, Self.W]](self.t)

    @always_inline
    def __add__(self, r: Self) -> Self: return Self(self.v + r.v, self.t + r.t)
    @always_inline
    def __add__(self, r: Float64) -> Self: return Self(self.v + Self.S(r), self.t)
    @always_inline
    def __radd__(self, l: Float64) -> Self: return Self(Self.S(l) + self.v, self.t)
    @always_inline
    def __sub__(self, r: Self) -> Self: return Self(self.v - r.v, self.t - r.t)
    @always_inline
    def __sub__(self, r: Float64) -> Self: return Self(self.v - Self.S(r), self.t)
    @always_inline
    def __rsub__(self, l: Float64) -> Self: return Self(Self.S(l) - self.v, -self.t)
    @always_inline
    def __mul__(self, r: Self) -> Self: return Self(self.v * r.v, self.v * r.t + self.t * r.v)
    @always_inline
    def __mul__(self, r: Float64) -> Self: return Self(self.v * Self.S(r), self.t * Self.S(r))
    @always_inline
    def __rmul__(self, l: Float64) -> Self: return Self(Self.S(l) * self.v, Self.S(l) * self.t)
    @always_inline
    def __truediv__(self, r: Self) -> Self:
        var q = self.v / r.v
        return Self(q, (self.t - q * r.t) / r.v)
    @always_inline
    def __truediv__(self, r: Float64) -> Self: return Self(self.v / Self.S(r), self.t / Self.S(r))
    @always_inline
    def __rtruediv__(self, l: Float64) -> Self:
        var q = Self.S(l) / self.v
        return Self(q, -q * self.t / self.v)
    @always_inline
    def __neg__(self) -> Self: return Self(-self.v, -self.t)
    @always_inline
    def __iadd__(mut self, r: Self):
        self.v += r.v
        self.t += r.t
    @always_inline
    def __isub__(mut self, r: Self):
        self.v -= r.v
        self.t -= r.t
    @always_inline
    def exp(self) -> Self:
        comptime assert Self.dt.is_floating_point()
        var e = exp(self.v)
        return Self(e, e * self.t)
    @always_inline
    def log(self) -> Self:
        comptime assert Self.dt.is_floating_point()
        return Self(log(self.v), self.t / self.v)
    @always_inline
    def sqrt(self) -> Self:
        comptime assert Self.dt.is_floating_point()
        var s = sqrt(self.v)
        return Self(s, Self.S(0.5) * self.t / s)
    @always_inline
    def powk(self, k: Float64) -> Self:
        var kk = Self.S(k)
        return Self(self.v ** kk, kk * self.v ** (kk - 1) * self.t)
    @always_inline
    def relu(self) -> Self:
        var m = self.v.gt(Self.S(0))
        return Self(m.select(self.v, Self.S(0)), m.select(self.t, Self.S(0)))

    @staticmethod
    @always_inline
    def select_gt(a: Self, b: Self, x: Self, y: Self) -> Self:
        var m = a.v.gt(b.v)
        return Self(m.select(x.v, y.v), m.select(x.t, y.t))

    @always_inline
    def set_lane(mut self, lane: Int, src: Self, src_lane: Int):
        self.v[lane] = src.v[src_lane]
        self.t[lane] = src.t[src_lane]
