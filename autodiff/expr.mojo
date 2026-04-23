"""
Expression templates for operation fusion in AAD.

Instead of recording one tape node per arithmetic operation, expression
templates build a lazy compile-time expression tree. When the expression
is finally recorded (via `record()`), it collapses into a single tape
node with N inputs and N partial derivatives — reducing tape traffic by
3-5x for typical MC inner loops.

The backward pass in propagate_adjoints remains fully generic — it just
processes fewer, wider nodes.
"""

from std.math import exp
from .tape import Tape

# Maximum number of leaf inputs per fused node. Matches FAT_NODE_MAX in tape.
comptime MAX_EXPR_LEAVES = 4


# ── ExprNumber: result of recording an expression onto the tape ──────

struct ExprNumber[width: Int = 1, type: DType = DType.float32](
    TrivialRegisterPassable, Writable,
):
    """A Number-like value produced by recording an expression template.

    Holds the computed value and tape index. Can be used to read gradients
    or as a leaf in further expressions.
    """
    var value: SIMD[Self.type, Self.width]
    var idx: Int
    var tape: Tape[Self.width, Self.type]

    @always_inline("nodebug")
    def __init__(out self, value: SIMD[Self.type, Self.width], idx: Int, tape: Tape[Self.width, Self.type]):
        self.value = value
        self.idx = idx
        self.tape = tape

    @always_inline("nodebug")
    def to_leaf(self) -> Leaf[Self.width, Self.type]:
        """Convert to a Leaf for use in further expression trees."""
        return Leaf[Self.width, Self.type](self.value, self.idx)

    def grad(self) -> SIMD[Self.type, Self.width]:
        return self.tape.adjoint(self.idx)

    def mean_grad(self) -> SIMD[Self.type, 1]:
        return self.tape.adjoint(self.idx).reduce_add() / Scalar[Self.type](Self.width)

    @always_inline("nodebug")
    def propagate_adjoints(self, starting_grad: SIMD[Self.type, Self.width] = 1.0):
        self.tape.propagate_adjoints(self.idx, starting_grad)

    def write_to(self, mut writer: Some[Writer]):
        writer.write(
            "<ExprNumber[", Self.width, ", ", Self.type, "]: value=", self.value, ">",
        )


# ── Leaf: wraps a tape-recorded Number ───────────────────────────────

struct Leaf[width: Int = 1, type: DType = DType.float32](TrivialRegisterPassable):
    """A leaf expression referencing an existing tape variable."""
    var value: SIMD[Self.type, Self.width]
    var tape_idx: Int

    @always_inline("nodebug")
    def __init__(out self, value: SIMD[Self.type, Self.width], tape_idx: Int):
        self.value = value
        self.tape_idx = tape_idx

    @always_inline("nodebug")
    def eval(self) -> SIMD[Self.type, Self.width]:
        return self.value

    @always_inline("nodebug")
    def propagate_partials(
        self,
        seed: SIMD[Self.type, Self.width],
        mut indices: InlineArray[Int, 4],
        mut derivs: InlineArray[SIMD[Self.type, Self.width], 4],
        mut count: Int,
    ):
        """Propagate seed to this leaf, accumulating (index, partial) pairs."""
        for i in range(count):
            if indices[i] == self.tape_idx:
                derivs[i] = derivs[i] + seed
                return
        indices[count] = self.tape_idx
        derivs[count] = seed
        count += 1

    @always_inline("nodebug")
    def record(
        self, tape: Tape[Self.width, Self.type]
    ) -> ExprNumber[Self.width, Self.type]:
        """Record as a single-input fused node (trivial case)."""
        var indices = InlineArray[Int, 4](fill=0)
        var derivs = InlineArray[SIMD[Self.type, Self.width], 4](
            fill=SIMD[Self.type, Self.width](0)
        )
        var count: Int = 0
        self.propagate_partials(SIMD[Self.type, Self.width](1.0), indices, derivs, count)
        tape.push_fused(indices, derivs, count)
        return ExprNumber[Self.width, Self.type](self.value, len(tape) - 1, tape)


# ── AddExpr: Leaf + Leaf ─────────────────────────────────────────────

struct AddExpr[
    width: Int = 1,
    type: DType = DType.float32,
](TrivialRegisterPassable):
    """Lazy addition of two Leaf expressions."""
    var lhs: Leaf[Self.width, Self.type]
    var rhs: Leaf[Self.width, Self.type]

    @always_inline("nodebug")
    def __init__(out self, lhs: Leaf[Self.width, Self.type], rhs: Leaf[Self.width, Self.type]):
        self.lhs = lhs
        self.rhs = rhs

    @always_inline("nodebug")
    def eval(self) -> SIMD[Self.type, Self.width]:
        return self.lhs.eval() + self.rhs.eval()

    @always_inline("nodebug")
    def propagate_partials(
        self,
        seed: SIMD[Self.type, Self.width],
        mut indices: InlineArray[Int, 4],
        mut derivs: InlineArray[SIMD[Self.type, Self.width], 4],
        mut count: Int,
    ):
        self.lhs.propagate_partials(seed, indices, derivs, count)
        self.rhs.propagate_partials(seed, indices, derivs, count)

    @always_inline("nodebug")
    def record(
        self, tape: Tape[Self.width, Self.type]
    ) -> ExprNumber[Self.width, Self.type]:
        var val = self.eval()
        var indices = InlineArray[Int, 4](fill=0)
        var derivs = InlineArray[SIMD[Self.type, Self.width], 4](
            fill=SIMD[Self.type, Self.width](0)
        )
        var count: Int = 0
        self.propagate_partials(SIMD[Self.type, Self.width](1.0), indices, derivs, count)
        tape.push_fused(indices, derivs, count)
        return ExprNumber[Self.width, Self.type](val, len(tape) - 1, tape)


# ── SubExpr: Leaf - Leaf ─────────────────────────────────────────────

struct SubExpr[
    width: Int = 1,
    type: DType = DType.float32,
](TrivialRegisterPassable):
    """Lazy subtraction of two Leaf expressions."""
    var lhs: Leaf[Self.width, Self.type]
    var rhs: Leaf[Self.width, Self.type]

    @always_inline("nodebug")
    def __init__(out self, lhs: Leaf[Self.width, Self.type], rhs: Leaf[Self.width, Self.type]):
        self.lhs = lhs
        self.rhs = rhs

    @always_inline("nodebug")
    def eval(self) -> SIMD[Self.type, Self.width]:
        return self.lhs.eval() - self.rhs.eval()

    @always_inline("nodebug")
    def propagate_partials(
        self,
        seed: SIMD[Self.type, Self.width],
        mut indices: InlineArray[Int, 4],
        mut derivs: InlineArray[SIMD[Self.type, Self.width], 4],
        mut count: Int,
    ):
        self.lhs.propagate_partials(seed, indices, derivs, count)
        self.rhs.propagate_partials(-seed, indices, derivs, count)

    @always_inline("nodebug")
    def record(
        self, tape: Tape[Self.width, Self.type]
    ) -> ExprNumber[Self.width, Self.type]:
        var val = self.eval()
        var indices = InlineArray[Int, 4](fill=0)
        var derivs = InlineArray[SIMD[Self.type, Self.width], 4](
            fill=SIMD[Self.type, Self.width](0)
        )
        var count: Int = 0
        self.propagate_partials(SIMD[Self.type, Self.width](1.0), indices, derivs, count)
        tape.push_fused(indices, derivs, count)
        return ExprNumber[Self.width, Self.type](val, len(tape) - 1, tape)


# ── MulExpr: Leaf * Leaf ─────────────────────────────────────────────

struct MulExpr[
    width: Int = 1,
    type: DType = DType.float32,
](TrivialRegisterPassable):
    """Lazy multiplication of two Leaf expressions."""
    var lhs: Leaf[Self.width, Self.type]
    var rhs: Leaf[Self.width, Self.type]

    @always_inline("nodebug")
    def __init__(out self, lhs: Leaf[Self.width, Self.type], rhs: Leaf[Self.width, Self.type]):
        self.lhs = lhs
        self.rhs = rhs

    @always_inline("nodebug")
    def eval(self) -> SIMD[Self.type, Self.width]:
        return self.lhs.eval() * self.rhs.eval()

    @always_inline("nodebug")
    def propagate_partials(
        self,
        seed: SIMD[Self.type, Self.width],
        mut indices: InlineArray[Int, 4],
        mut derivs: InlineArray[SIMD[Self.type, Self.width], 4],
        mut count: Int,
    ):
        var lhs_val = self.lhs.eval()
        var rhs_val = self.rhs.eval()
        self.lhs.propagate_partials(seed * rhs_val, indices, derivs, count)
        self.rhs.propagate_partials(seed * lhs_val, indices, derivs, count)

    @always_inline("nodebug")
    def record(
        self, tape: Tape[Self.width, Self.type]
    ) -> ExprNumber[Self.width, Self.type]:
        var val = self.eval()
        var indices = InlineArray[Int, 4](fill=0)
        var derivs = InlineArray[SIMD[Self.type, Self.width], 4](
            fill=SIMD[Self.type, Self.width](0)
        )
        var count: Int = 0
        self.propagate_partials(SIMD[Self.type, Self.width](1.0), indices, derivs, count)
        tape.push_fused(indices, derivs, count)
        return ExprNumber[Self.width, Self.type](val, len(tape) - 1, tape)


# ── Compound expressions for deeper nesting ──────────────────────────
#
# Since Mojo requires concrete types for struct fields (no recursive
# parametric types through AnyType easily), we provide compound types
# for common patterns. These compose the base Leaf-only expressions.

struct MulAddExpr[
    width: Int = 1,
    type: DType = DType.float32,
](TrivialRegisterPassable):
    """Lazy expression: (a + b) * c, built from 3 leaves.

    Fuses add + mul into a single tape node with 3 inputs.
    """
    var a: Leaf[Self.width, Self.type]
    var b: Leaf[Self.width, Self.type]
    var c: Leaf[Self.width, Self.type]

    @always_inline("nodebug")
    def __init__(
        out self,
        add_lhs: Leaf[Self.width, Self.type],
        add_rhs: Leaf[Self.width, Self.type],
        mul_rhs: Leaf[Self.width, Self.type],
    ):
        self.a = add_lhs
        self.b = add_rhs
        self.c = mul_rhs

    @always_inline("nodebug")
    def eval(self) -> SIMD[Self.type, Self.width]:
        return (self.a.eval() + self.b.eval()) * self.c.eval()

    @always_inline("nodebug")
    def propagate_partials(
        self,
        seed: SIMD[Self.type, Self.width],
        mut indices: InlineArray[Int, 4],
        mut derivs: InlineArray[SIMD[Self.type, Self.width], 4],
        mut count: Int,
    ):
        # f = (a + b) * c
        # df/da = c, df/db = c, df/dc = a + b
        var c_val = self.c.eval()
        var ab_val = self.a.eval() + self.b.eval()
        self.a.propagate_partials(seed * c_val, indices, derivs, count)
        self.b.propagate_partials(seed * c_val, indices, derivs, count)
        self.c.propagate_partials(seed * ab_val, indices, derivs, count)

    @always_inline("nodebug")
    def record(
        self, tape: Tape[Self.width, Self.type]
    ) -> ExprNumber[Self.width, Self.type]:
        var val = self.eval()
        var indices = InlineArray[Int, 4](fill=0)
        var derivs = InlineArray[SIMD[Self.type, Self.width], 4](
            fill=SIMD[Self.type, Self.width](0)
        )
        var count: Int = 0
        self.propagate_partials(SIMD[Self.type, Self.width](1.0), indices, derivs, count)
        tape.push_fused(indices, derivs, count)
        return ExprNumber[Self.width, Self.type](val, len(tape) - 1, tape)


struct MulMulAddExpr[
    width: Int = 1,
    type: DType = DType.float32,
](TrivialRegisterPassable):
    """Lazy expression: (a * b + c) * d, built from 4 leaves.

    Fuses mul + add + mul into a single tape node with 4 inputs.
    """
    var a: Leaf[Self.width, Self.type]
    var b: Leaf[Self.width, Self.type]
    var c: Leaf[Self.width, Self.type]
    var d: Leaf[Self.width, Self.type]

    @always_inline("nodebug")
    def __init__(
        out self,
        a: Leaf[Self.width, Self.type],
        b: Leaf[Self.width, Self.type],
        c: Leaf[Self.width, Self.type],
        d: Leaf[Self.width, Self.type],
    ):
        self.a = a
        self.b = b
        self.c = c
        self.d = d

    @always_inline("nodebug")
    def eval(self) -> SIMD[Self.type, Self.width]:
        return (self.a.eval() * self.b.eval() + self.c.eval()) * self.d.eval()

    @always_inline("nodebug")
    def propagate_partials(
        self,
        seed: SIMD[Self.type, Self.width],
        mut indices: InlineArray[Int, 4],
        mut derivs: InlineArray[SIMD[Self.type, Self.width], 4],
        mut count: Int,
    ):
        # f = (a*b + c) * d
        # df/da = b * d
        # df/db = a * d
        # df/dc = d
        # df/dd = a*b + c
        var a_val = self.a.eval()
        var b_val = self.b.eval()
        var c_val = self.c.eval()
        var d_val = self.d.eval()
        var inner = a_val * b_val + c_val
        self.a.propagate_partials(seed * b_val * d_val, indices, derivs, count)
        self.b.propagate_partials(seed * a_val * d_val, indices, derivs, count)
        self.c.propagate_partials(seed * d_val, indices, derivs, count)
        self.d.propagate_partials(seed * inner, indices, derivs, count)

    @always_inline("nodebug")
    def record(
        self, tape: Tape[Self.width, Self.type]
    ) -> ExprNumber[Self.width, Self.type]:
        var val = self.eval()
        var indices = InlineArray[Int, 4](fill=0)
        var derivs = InlineArray[SIMD[Self.type, Self.width], 4](
            fill=SIMD[Self.type, Self.width](0)
        )
        var count: Int = 0
        self.propagate_partials(SIMD[Self.type, Self.width](1.0), indices, derivs, count)
        tape.push_fused(indices, derivs, count)
        return ExprNumber[Self.width, Self.type](val, len(tape) - 1, tape)


struct MulSubExpr[
    width: Int = 1,
    type: DType = DType.float32,
](TrivialRegisterPassable):
    """Lazy expression: (a - b) * c, built from 3 leaves.

    Fuses sub + mul into a single tape node with 3 inputs.
    """
    var a: Leaf[Self.width, Self.type]
    var b: Leaf[Self.width, Self.type]
    var c: Leaf[Self.width, Self.type]

    @always_inline("nodebug")
    def __init__(
        out self,
        sub_lhs: Leaf[Self.width, Self.type],
        sub_rhs: Leaf[Self.width, Self.type],
        mul_rhs: Leaf[Self.width, Self.type],
    ):
        self.a = sub_lhs
        self.b = sub_rhs
        self.c = mul_rhs

    @always_inline("nodebug")
    def eval(self) -> SIMD[Self.type, Self.width]:
        return (self.a.eval() - self.b.eval()) * self.c.eval()

    @always_inline("nodebug")
    def propagate_partials(
        self,
        seed: SIMD[Self.type, Self.width],
        mut indices: InlineArray[Int, 4],
        mut derivs: InlineArray[SIMD[Self.type, Self.width], 4],
        mut count: Int,
    ):
        # f = (a - b) * c
        # df/da = c, df/db = -c, df/dc = a - b
        var c_val = self.c.eval()
        var ab_val = self.a.eval() - self.b.eval()
        self.a.propagate_partials(seed * c_val, indices, derivs, count)
        self.b.propagate_partials(-(seed * c_val), indices, derivs, count)
        self.c.propagate_partials(seed * ab_val, indices, derivs, count)

    @always_inline("nodebug")
    def record(
        self, tape: Tape[Self.width, Self.type]
    ) -> ExprNumber[Self.width, Self.type]:
        var val = self.eval()
        var indices = InlineArray[Int, 4](fill=0)
        var derivs = InlineArray[SIMD[Self.type, Self.width], 4](
            fill=SIMD[Self.type, Self.width](0)
        )
        var count: Int = 0
        self.propagate_partials(SIMD[Self.type, Self.width](1.0), indices, derivs, count)
        tape.push_fused(indices, derivs, count)
        return ExprNumber[Self.width, Self.type](val, len(tape) - 1, tape)


struct ExpMulExpr[
    width: Int = 1,
    type: DType = DType.float32,
](TrivialRegisterPassable):
    """Lazy expression: exp(a + b) * c, built from 3 leaves.

    Fuses add + exp + mul into a single tape node with 3 inputs.
    This pattern appears in MC paths: path * exp(drift + vsdt * rand).
    """
    var a: Leaf[Self.width, Self.type]
    var b: Leaf[Self.width, Self.type]
    var c: Leaf[Self.width, Self.type]

    @always_inline("nodebug")
    def __init__(
        out self,
        exp_add_lhs: Leaf[Self.width, Self.type],
        exp_add_rhs: Leaf[Self.width, Self.type],
        mul_rhs: Leaf[Self.width, Self.type],
    ):
        self.a = exp_add_lhs
        self.b = exp_add_rhs
        self.c = mul_rhs

    @always_inline("nodebug")
    def eval(self) -> SIMD[Self.type, Self.width]:
        comptime assert Self.type.is_floating_point(), "ExpMulExpr requires floating-point type"
        return exp(self.a.eval() + self.b.eval()) * self.c.eval()

    @always_inline("nodebug")
    def propagate_partials(
        self,
        seed: SIMD[Self.type, Self.width],
        mut indices: InlineArray[Int, 4],
        mut derivs: InlineArray[SIMD[Self.type, Self.width], 4],
        mut count: Int,
    ):
        comptime assert Self.type.is_floating_point(), "ExpMulExpr requires floating-point type"
        # f = exp(a + b) * c
        # df/da = exp(a+b) * c
        # df/db = exp(a+b) * c
        # df/dc = exp(a+b)
        var exp_val = exp(self.a.eval() + self.b.eval())
        var c_val = self.c.eval()
        self.a.propagate_partials(seed * exp_val * c_val, indices, derivs, count)
        self.b.propagate_partials(seed * exp_val * c_val, indices, derivs, count)
        self.c.propagate_partials(seed * exp_val, indices, derivs, count)

    @always_inline("nodebug")
    def record(
        self, tape: Tape[Self.width, Self.type]
    ) -> ExprNumber[Self.width, Self.type]:
        var val = self.eval()
        var indices = InlineArray[Int, 4](fill=0)
        var derivs = InlineArray[SIMD[Self.type, Self.width], 4](
            fill=SIMD[Self.type, Self.width](0)
        )
        var count: Int = 0
        self.propagate_partials(SIMD[Self.type, Self.width](1.0), indices, derivs, count)
        tape.push_fused(indices, derivs, count)
        return ExprNumber[Self.width, Self.type](val, len(tape) - 1, tape)


# ── Helper: make a Leaf from a Number ────────────────────────────────

def make_leaf[width: Int, type: DType](
    value: SIMD[type, width], tape_idx: Int
) -> Leaf[width, type]:
    """Create a Leaf expression from a Number's value and tape index."""
    return Leaf[width, type](value, tape_idx)
