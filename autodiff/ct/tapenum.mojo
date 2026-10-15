"""`TapeNum`: the `ADNum` API on top of the runtime tape (`Number`).

Lets the same kernel source run on the tape: the flexible fallback for code that
does not fit compile-time capture, and the oracle that tests compiled kernels.
"""
from std.math import exp, log, sqrt
from ..number import Number
from ..tape import Tape
from .adnum import ADNum


struct TapeNum[W: Int, DT: DType](ADNum, TrivialRegisterPassable):
    var n: Number[Self.W, Self.DT]

    @always_inline
    def __init__(out self, n: Number[Self.W, Self.DT]):
        self.n = n

    @always_inline
    def _k(self, k: Float64) -> SIMD[Self.DT, Self.W]:
        return SIMD[Self.DT, Self.W](k)

    def constant(self, k: Float64) -> Self:
        return Self(Number[Self.W, Self.DT](self._k(k), self.n.tape))

    def __add__(self, rhs: Self) -> Self: return Self(self.n + rhs.n)
    def __add__(self, rhs: Float64) -> Self:
        return Self(Number[Self.W, Self.DT].unary(self.n.value + self._k(rhs), self.n, 1))
    def __radd__(self, lhs: Float64) -> Self: return self + lhs
    def __sub__(self, rhs: Self) -> Self: return Self(self.n - rhs.n)
    def __sub__(self, rhs: Float64) -> Self:
        return Self(Number[Self.W, Self.DT].unary(self.n.value - self._k(rhs), self.n, 1))
    def __rsub__(self, lhs: Float64) -> Self:
        return Self(Number[Self.W, Self.DT].unary(self._k(lhs) - self.n.value, self.n, -1))
    def __mul__(self, rhs: Self) -> Self: return Self(self.n * rhs.n)
    def __mul__(self, rhs: Float64) -> Self: return Self(self.n * self._k(rhs))
    def __rmul__(self, lhs: Float64) -> Self: return self * lhs
    def __truediv__(self, rhs: Self) -> Self: return Self(self.n / rhs.n)
    def __truediv__(self, rhs: Float64) -> Self: return Self(self.n * self._k(1.0 / rhs))
    def __rtruediv__(self, lhs: Float64) -> Self:
        var v = self._k(lhs) / self.n.value
        return Self(Number[Self.W, Self.DT].unary(v, self.n, -v / self.n.value))
    def __neg__(self) -> Self:
        return Self(Number[Self.W, Self.DT].unary(-self.n.value, self.n, -1))
    def __pow__(self, k: Float64) -> Self:
        var kk = self._k(k)
        return Self(Number[Self.W, Self.DT].unary(self.n.value ** kk, self.n, kk * self.n.value ** (kk - 1)))
    def exp(self) -> Self: return Self(self.n.expon())
    def log(self) -> Self: return Self(self.n.log())
    def sqrt(self) -> Self: return Self(self.n.sqrt())
    def detach(self) -> Self:
        return Self(Number[Self.W, Self.DT](self.n.value, self.n.tape))

    def relu(self) -> Self:
        var m = self.n.value.gt(0).select(SIMD[Self.DT, Self.W](1), SIMD[Self.DT, Self.W](0))
        return Self(Number[Self.W, Self.DT].unary(self.n.value * m, self.n, m))

    @staticmethod
    def select_gt(a: Self, b: Self, x: Self, y: Self) -> Self:
        var m = a.n.value.gt(b.n.value).select(SIMD[Self.DT, Self.W](1), SIMD[Self.DT, Self.W](0))
        return Self(x.n * m + y.n * (1 - m))
