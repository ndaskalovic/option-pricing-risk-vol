"""Compile-time reverse-mode AD over a traced kernel graph.

`Compiled[K]` traces `K` inside the compiler, analyses the graph (which nodes need
adjoints, which forward values the reverse sweep reads) and exposes straight-line
`eval`, `fwd` (saves only the needed residuals), `bwd`, and fused `vjp`. All loops
here are `comptime for` over the graph, so the emitted code has no tape, no
indices and no branches on node type.
"""
from std.math import exp, log, sqrt
from .ops import *
from .trace import Kernel, trace


def _is_leaf(op: Int) -> Bool:
    return op == INPUT or op == DATA or op == CONST or op == OUT


def _active(g: List[Node]) -> List[Bool]:
    """Node depends on a differentiable input."""
    var act = List[Bool](length=len(g), fill=False)
    for i in range(len(g)):
        var n = g[i]
        if n.op == INPUT:
            act[i] = True
        elif n.op == STOP:
            act[i] = False
        elif not _is_leaf(n.op):
            var r = False
            if n.a >= 0: r = r or act[n.a]
            if n.b >= 0: r = r or act[n.b]
            if n.c >= 0: r = r or act[n.c]
            if n.d >= 0: r = r or act[n.d]
            act[i] = r
    return act^


def _useful(g: List[Node]) -> List[Bool]:
    """Node reaches an output."""
    var use = List[Bool](length=len(g), fill=False)
    for j in range(len(g)):
        var i = len(g) - 1 - j
        var n = g[i]
        if n.op == OUT:
            use[n.a] = True
        elif use[i] and not _is_leaf(n.op):
            if n.op == SELGT:
                use[n.c] = True   # mask operands a, b get no adjoint
                use[n.d] = True
            else:
                if n.a >= 0: use[n.a] = True
                if n.b >= 0: use[n.b] = True
    return use^


def _need(g: List[Node]) -> List[Bool]:
    """Node needs an adjoint: active and on a path to an output."""
    var act = _active(g)
    var use = _useful(g)
    var need = List[Bool](length=len(g), fill=False)
    for i in range(len(g)):
        need[i] = act[i] and use[i]
    return need^


def _reads(g: List[Node]) -> List[Bool]:
    """Forward values the reverse sweep reads."""
    var need = _need(g)
    var rd = List[Bool](length=len(g), fill=False)
    for i in range(len(g)):
        var n = g[i]
        if not need[i] or _is_leaf(n.op):
            continue
        if n.op == MUL:
            if need[n.a]: rd[n.b] = True
            if need[n.b]: rd[n.a] = True
        elif n.op == DIV:
            if need[n.a] or need[n.b]: rd[n.b] = True
            if need[n.b]: rd[i] = True
        elif n.op == EXP or n.op == SQRT:
            rd[i] = True
        elif n.op == LOG or n.op == POWK or n.op == RELU:
            rd[n.a] = True
        elif n.op == SELGT:
            rd[n.a] = True
            rd[n.b] = True
    return rd^


def _residuals(g: List[Node]) -> List[Int]:
    """Values to save in `fwd`: read by the reverse sweep and not cheaply rematerialised."""
    var rd = _reads(g)
    var out = List[Int]()
    for i in range(len(g)):
        if rd[i] and g[i].op != DATA and g[i].op != CONST:
            out.append(i)
    return out^


struct Compiled[K: Kernel]:
    comptime G = trace[Self.K]()
    comptime N = len(Self.G)
    comptime NEED = _need(Self.G)
    comptime RES = _residuals(Self.G)
    comptime NRES = len(Self.RES)

    @staticmethod
    @always_inline
    def _forward[W: Int, DT: DType](
        x: Array[SIMD[DT, W], Self.K.NI], d: Array[SIMD[DT, W], Self.K.ND],
        mut val: Array[SIMD[DT, W], Self.N],
    ):
        comptime assert DT.is_floating_point(), "kernels need a floating-point dtype"
        comptime V = SIMD[DT, W]
        comptime for i in range(Self.N):
            comptime n = Self.G[i]
            comptime if n.op == INPUT:
                val[i] = x[n.a]
            elif n.op == DATA:
                val[i] = d[n.a]
            elif n.op == CONST:
                val[i] = V(n.k)
            elif n.op == ADD:
                val[i] = val[n.a] + val[n.b]
            elif n.op == SUB:
                val[i] = val[n.a] - val[n.b]
            elif n.op == MUL:
                val[i] = val[n.a] * val[n.b]
            elif n.op == DIV:
                val[i] = val[n.a] / val[n.b]
            elif n.op == NEG:
                val[i] = -val[n.a]
            elif n.op == EXP:
                val[i] = exp(val[n.a])
            elif n.op == LOG:
                val[i] = log(val[n.a])
            elif n.op == SQRT:
                val[i] = sqrt(val[n.a])
            elif n.op == POWK:
                val[i] = val[n.a] ** V(n.k)
            elif n.op == RELU:
                val[i] = max(val[n.a], V(0))
            elif n.op == SELGT:
                val[i] = val[n.a].gt(val[n.b]).select(val[n.c], val[n.d])
            elif n.op == STOP:
                val[i] = val[n.a]

    @staticmethod
    @always_inline
    def _outputs[W: Int, DT: DType](val: Array[SIMD[DT, W], Self.N]) -> Array[SIMD[DT, W], Self.K.NO]:
        var y = Array[SIMD[DT, W], Self.K.NO](fill=0)
        comptime for i in range(Self.N):
            comptime n = Self.G[i]
            comptime if n.op == OUT:
                y[n.b] = val[n.a]
        return y^

    @staticmethod
    @always_inline
    def _reverse[W: Int, DT: DType](
        val: Array[SIMD[DT, W], Self.N], a_y: Array[SIMD[DT, W], Self.K.NO],
        mut a_x: Array[SIMD[DT, W], Self.K.NI],
    ):
        """Accumulate a_x += J^T a_y. Adjoints start at -0.0 (the IEEE additive identity),
        so the first `+=` into each adjoint folds to a plain assignment."""
        comptime assert DT.is_floating_point(), "kernels need a floating-point dtype"
        comptime V = SIMD[DT, W]
        var adj = Array[V, Self.N](fill=V(-0.0))
        comptime for i in range(Self.N):
            comptime n = Self.G[i]
            comptime if n.op == OUT:
                comptime if Self.NEED[n.a]:
                    adj[n.a] += a_y[n.b]
        comptime for j in range(Self.N):
            comptime i = Self.N - 1 - j
            comptime n = Self.G[i]
            comptime if not Self.NEED[i] or n.op == OUT:
                pass
            elif n.op == INPUT:
                a_x[n.a] += adj[i]
            elif n.op == ADD:
                comptime if Self.NEED[n.a]: adj[n.a] += adj[i]
                comptime if Self.NEED[n.b]: adj[n.b] += adj[i]
            elif n.op == SUB:
                comptime if Self.NEED[n.a]: adj[n.a] += adj[i]
                comptime if Self.NEED[n.b]: adj[n.b] -= adj[i]
            elif n.op == MUL:
                comptime if Self.NEED[n.a]: adj[n.a] += adj[i] * val[n.b]
                comptime if Self.NEED[n.b]: adj[n.b] += adj[i] * val[n.a]
            elif n.op == DIV:
                comptime if Self.NEED[n.a]: adj[n.a] += adj[i] / val[n.b]
                comptime if Self.NEED[n.b]: adj[n.b] -= adj[i] * val[i] / val[n.b]
            elif n.op == NEG:
                adj[n.a] -= adj[i]
            elif n.op == EXP:
                adj[n.a] += adj[i] * val[i]
            elif n.op == LOG:
                adj[n.a] += adj[i] / val[n.a]
            elif n.op == SQRT:
                adj[n.a] += adj[i] * V(0.5) / val[i]
            elif n.op == POWK:
                adj[n.a] += adj[i] * V(n.k) * val[n.a] ** V(n.k - 1)
            elif n.op == RELU:
                adj[n.a] += val[n.a].gt(V(0)).select(adj[i], V(0))
            elif n.op == SELGT:
                var m = val[n.a].gt(val[n.b])
                comptime if Self.NEED[n.c]: adj[n.c] += m.select(adj[i], V(0))
                comptime if Self.NEED[n.d]: adj[n.d] += m.select(V(0), adj[i])

    # ── public API ───────────────────────────────────────────────────────

    @staticmethod
    @always_inline
    def eval[W: Int, DT: DType](
        x: Array[SIMD[DT, W], Self.K.NI], d: Array[SIMD[DT, W], Self.K.ND]
    ) -> Array[SIMD[DT, W], Self.K.NO]:
        """Primal only."""
        var val = Array[SIMD[DT, W], Self.N](fill=0)
        Self._forward(x, d, val)
        return Self._outputs(val)

    @staticmethod
    @always_inline
    def fwd[W: Int, DT: DType](
        x: Array[SIMD[DT, W], Self.K.NI], d: Array[SIMD[DT, W], Self.K.ND],
        res: Pointer[SIMD[DT, W], MutUntrackedOrigin],
    ) -> Array[SIMD[DT, W], Self.K.NO]:
        """Primal, saving the NRES values `bwd` needs into res[0..NRES)."""
        var val = Array[SIMD[DT, W], Self.N](fill=0)
        Self._forward(x, d, val)
        comptime for r in range(Self.NRES):
            comptime j = Self.RES[r]
            res[unsafe_offset=r] = val[j]
        return Self._outputs(val)

    @staticmethod
    @always_inline
    def bwd[W: Int, DT: DType](
        d: Array[SIMD[DT, W], Self.K.ND], res: Pointer[SIMD[DT, W], MutUntrackedOrigin],
        a_y: Array[SIMD[DT, W], Self.K.NO], mut a_x: Array[SIMD[DT, W], Self.K.NI],
    ):
        """a_x += J^T a_y, using residuals saved by `fwd` and the same data `d`."""
        comptime V = SIMD[DT, W]
        var val = Array[V, Self.N](fill=0)
        comptime for r in range(Self.NRES):
            comptime j = Self.RES[r]
            val[j] = res[unsafe_offset=r]
        comptime for i in range(Self.N):
            comptime n = Self.G[i]
            comptime if n.op == DATA:
                val[i] = d[n.a]
            elif n.op == CONST:
                val[i] = V(n.k)
        Self._reverse(val, a_y, a_x)

    @staticmethod
    @always_inline
    def vjp[W: Int, DT: DType](
        x: Array[SIMD[DT, W], Self.K.NI], d: Array[SIMD[DT, W], Self.K.ND],
        a_y: Array[SIMD[DT, W], Self.K.NO], mut a_x: Array[SIMD[DT, W], Self.K.NI],
    ) -> Array[SIMD[DT, W], Self.K.NO]:
        """Fused primal + reverse (no residual buffer). Returns y; a_x += J^T a_y."""
        var val = Array[SIMD[DT, W], Self.N](fill=0)
        Self._forward(x, d, val)
        Self._reverse(val, a_y, a_x)
        return Self._outputs(val)
