"""Compile-time reverse-mode AD over a traced kernel graph.

`Compiled[K]` traces `K` inside the compiler, analyses the graph (which nodes need
adjoints, which forward values the reverse sweep reads) and exposes straight-line
`eval`, `fwd` (saves only the needed residuals), `bwd`, and fused `vjp`. All loops
here are `comptime for` over the graph, so the emitted code has no tape, no
indices and no branches on node type.

The emitted code is generic over the runtime value type `R: Real`:
`Vec[DT, W]` gives first-order AAD; `Dual[DT, W]` runs the same forward and reverse
code in tangent mode, i.e. forward-over-reverse (Hessian-vector products).
"""
from .ops import *
from .trace import Kernel, trace
from .real import Real


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
    def _forward[R: Real](x: Array[R, Self.K.NI], d: Array[R, Self.K.ND], mut val: Array[R, Self.N]):
        comptime for i in range(Self.N):
            comptime n = Self.G[i]
            comptime if n.op == INPUT:
                val[i] = x[n.a]
            elif n.op == DATA:
                val[i] = d[n.a]
            elif n.op == CONST:
                val[i] = R.splat(n.k)
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
                val[i] = val[n.a].exp()
            elif n.op == LOG:
                val[i] = val[n.a].log()
            elif n.op == SQRT:
                val[i] = val[n.a].sqrt()
            elif n.op == POWK:
                val[i] = val[n.a].powk(n.k)
            elif n.op == RELU:
                val[i] = val[n.a].relu()
            elif n.op == SELGT:
                val[i] = R.select_gt(val[n.a], val[n.b], val[n.c], val[n.d])
            elif n.op == STOP:
                val[i] = R.lift(val[n.a].value())

    @staticmethod
    @always_inline
    def _outputs[R: Real](val: Array[R, Self.N]) -> Array[R, Self.K.NO]:
        var y = Array[R, Self.K.NO](fill=R.splat(0))
        comptime for i in range(Self.N):
            comptime n = Self.G[i]
            comptime if n.op == OUT:
                y[n.b] = val[n.a]
        return y^

    @staticmethod
    @always_inline
    def _reverse[R: Real](val: Array[R, Self.N], a_y: Array[R, Self.K.NO], mut a_x: Array[R, Self.K.NI]):
        """Accumulate a_x += J^T a_y. Adjoints start at -0.0 (the IEEE additive identity),
        so the first `+=` into each adjoint folds to a plain assignment."""
        var adj = Array[R, Self.N](fill=R.splat(-0.0))
        var zero = R.splat(0)
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
                var ai = adj[i]   # local copy: element-to-element updates must not alias
                a_x[n.a] += ai
            elif n.op == ADD:
                var ai = adj[i]
                comptime if Self.NEED[n.a]: adj[n.a] += ai
                comptime if Self.NEED[n.b]: adj[n.b] += ai
            elif n.op == SUB:
                var ai = adj[i]
                comptime if Self.NEED[n.a]: adj[n.a] += ai
                comptime if Self.NEED[n.b]: adj[n.b] -= ai
            elif n.op == MUL:
                var ai = adj[i]
                comptime if Self.NEED[n.a]: adj[n.a] += ai * val[n.b]
                comptime if Self.NEED[n.b]: adj[n.b] += ai * val[n.a]
            elif n.op == DIV:
                var ai = adj[i]
                comptime if Self.NEED[n.a]: adj[n.a] += ai / val[n.b]
                comptime if Self.NEED[n.b]: adj[n.b] -= ai * val[i] / val[n.b]
            elif n.op == NEG:
                var ai = adj[i]
                adj[n.a] -= ai
            elif n.op == EXP:
                var ai = adj[i]
                adj[n.a] += ai * val[i]
            elif n.op == LOG:
                var ai = adj[i]
                adj[n.a] += ai / val[n.a]
            elif n.op == SQRT:
                var ai = adj[i]
                adj[n.a] += ai * 0.5 / val[i]
            elif n.op == POWK:
                var ai = adj[i]
                adj[n.a] += ai * n.k * val[n.a].powk(n.k - 1)
            elif n.op == RELU:
                var ai = adj[i]
                adj[n.a] += R.select_gt(val[n.a], zero, ai, zero)
            elif n.op == SELGT:
                var ai = adj[i]
                comptime if Self.NEED[n.c]: adj[n.c] += R.select_gt(val[n.a], val[n.b], ai, zero)
                comptime if Self.NEED[n.d]: adj[n.d] += R.select_gt(val[n.a], val[n.b], zero, ai)

    # ── public API (generic over the value type R) ───────────────────────

    @staticmethod
    @always_inline
    def eval[R: Real](x: Array[R, Self.K.NI], d: Array[R, Self.K.ND]) -> Array[R, Self.K.NO]:
        """Primal only."""
        var val = Array[R, Self.N](fill=R.splat(0))
        Self._forward(x, d, val)
        return Self._outputs(val)

    @staticmethod
    @always_inline
    def fwd[R: Real](
        x: Array[R, Self.K.NI], d: Array[R, Self.K.ND], res: Pointer[R, MutUntrackedOrigin]
    ) -> Array[R, Self.K.NO]:
        """Primal, saving the NRES values `bwd` needs into res[0..NRES)."""
        var val = Array[R, Self.N](fill=R.splat(0))
        Self._forward(x, d, val)
        comptime for r in range(Self.NRES):
            comptime j = Self.RES[r]
            res[unsafe_offset=r] = val[j]
        return Self._outputs(val)

    @staticmethod
    @always_inline
    def bwd[R: Real](
        d: Array[R, Self.K.ND], res: Pointer[R, MutUntrackedOrigin],
        a_y: Array[R, Self.K.NO], mut a_x: Array[R, Self.K.NI],
    ):
        """a_x += J^T a_y, using residuals saved by `fwd` and the same data `d`."""
        var val = Array[R, Self.N](fill=R.splat(0))
        comptime for r in range(Self.NRES):
            comptime j = Self.RES[r]
            val[j] = res[unsafe_offset=r]
        comptime for i in range(Self.N):
            comptime n = Self.G[i]
            comptime if n.op == DATA:
                val[i] = d[n.a]
            elif n.op == CONST:
                val[i] = R.splat(n.k)
        Self._reverse(val, a_y, a_x)

    @staticmethod
    @always_inline
    def vjp[R: Real](
        x: Array[R, Self.K.NI], d: Array[R, Self.K.ND],
        a_y: Array[R, Self.K.NO], mut a_x: Array[R, Self.K.NI],
    ) -> Array[R, Self.K.NO]:
        """Fused primal + reverse (no residual buffer). Returns y; a_x += J^T a_y.
        With R = Dual, the tangent of a_x is the directional derivative of J^T a_y."""
        var val = Array[R, Self.N](fill=R.splat(0))
        Self._forward(x, d, val)
        Self._reverse(val, a_y, a_x)
        return Self._outputs(val)
