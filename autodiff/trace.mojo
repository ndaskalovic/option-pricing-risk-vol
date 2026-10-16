"""Compile-time graph capture: `trace[K]()` runs a kernel on `TraceNum` inside the compiler."""
from .ops import *
from .adnum import ADNum


trait Kernel:
    """A pure, branch-free function x (differentiable) , d (data) -> y.

    Write `eval` once, generic over `N: ADNum`; `Compiled[K]` turns it into
    straight-line forward and reverse code at compile time.
    """
    comptime NI: Int  # differentiable inputs
    comptime ND: Int  # data inputs (no derivative), e.g. normal draws
    comptime NO: Int  # outputs

    @staticmethod
    def eval[N: ADNum](x: Array[N, Self.NI], d: Array[N, Self.ND]) -> Array[N, Self.NO]: ...


struct TraceNum(ADNum):
    """Records each operation as a graph node instead of computing it."""
    var i: Int
    var g: Pointer[List[Node], MutUntrackedOrigin]

    def __init__(out self, i: Int, g: Pointer[List[Node], MutUntrackedOrigin]):
        self.i = i
        self.g = g

    def _push(self, op: Int, a: Int, b: Int = -1, c: Int = -1, d: Int = -1, k: Float64 = 0) -> Self:
        self.g[].append(Node(op, a, b, c, d, k))
        return Self(len(self.g[]) - 1, self.g)

    def constant(self, k: Float64) -> Self:
        return self._push(CONST, -1, k=k)

    def __add__(self, rhs: Self) -> Self: return self._push(ADD, self.i, rhs.i)
    def __add__(self, rhs: Float64) -> Self: return self + self.constant(rhs)
    def __radd__(self, lhs: Float64) -> Self: return self.constant(lhs) + self
    def __sub__(self, rhs: Self) -> Self: return self._push(SUB, self.i, rhs.i)
    def __sub__(self, rhs: Float64) -> Self: return self - self.constant(rhs)
    def __rsub__(self, lhs: Float64) -> Self: return self.constant(lhs) - self
    def __mul__(self, rhs: Self) -> Self: return self._push(MUL, self.i, rhs.i)
    def __mul__(self, rhs: Float64) -> Self: return self * self.constant(rhs)
    def __rmul__(self, lhs: Float64) -> Self: return self.constant(lhs) * self
    def __truediv__(self, rhs: Self) -> Self: return self._push(DIV, self.i, rhs.i)
    def __truediv__(self, rhs: Float64) -> Self: return self * self.constant(1.0 / rhs)
    def __rtruediv__(self, lhs: Float64) -> Self: return self.constant(lhs) / self
    def __neg__(self) -> Self: return self._push(NEG, self.i)
    def __pow__(self, k: Float64) -> Self: return self._push(POWK, self.i, k=k)
    def exp(self) -> Self: return self._push(EXP, self.i)
    def log(self) -> Self: return self._push(LOG, self.i)
    def sqrt(self) -> Self: return self._push(SQRT, self.i)
    def relu(self) -> Self: return self._push(RELU, self.i)
    def detach(self) -> Self: return self._push(STOP, self.i)

    @staticmethod
    def select_gt(a: Self, b: Self, x: Self, y: Self) -> Self:
        return a._push(SELGT, a.i, b.i, x.i, y.i)


def trace[K: Kernel]() -> List[Node]:
    """Run `K.eval` on tracers and return the graph (inputs, data, ops, then OUT nodes)."""
    var g = List[Node]()
    var p = Pointer(to=g).unsafe_origin_cast[MutUntrackedOrigin]()
    var x = Array[TraceNum, K.NI](fill=TraceNum(-1, p))
    var d = Array[TraceNum, K.ND](fill=TraceNum(-1, p))
    for i in range(K.NI):
        g.append(Node(INPUT, i, -1, -1, -1, 0))
        x[i] = TraceNum(len(g) - 1, p)
    for i in range(K.ND):
        g.append(Node(DATA, i, -1, -1, -1, 0))
        d[i] = TraceNum(len(g) - 1, p)
    var y = K.eval[TraceNum](x, d)
    for o in range(K.NO):
        g.append(Node(OUT, y[o].i, o, -1, -1, 0))
    return g^
