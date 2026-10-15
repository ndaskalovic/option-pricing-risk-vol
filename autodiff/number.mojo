# Copyright (c) 2024 Nicholas Daskalovic
# All rights reserved.
#
# This code is proprietary and confidential. Permission is granted to view
# this source code for personal, educational purposes only. Any other use,
# including copying, modification, distribution, or use of any portion of
# this source code, in whole or in part, without express written permission
# is strictly prohibited.

"""
This file contains the custom Number type with overloaded operators for
automatic differentiation as well as SIMD support.
"""

from std.math import log, sqrt, exp
from .node import Node
from .tape import Tape, MAX_NODE_INPUTS


struct Number[width: Int = 1, type: DType = DType.float32](TrivialRegisterPassable, Writable):
    var value: SIMD[Self.type, Self.width]
    var idx: Int
    var tape: Tape[Self.width, Self.type]

    @always_inline("nodebug")
    def __init__(
        out self,
        x: SIMD[Self.type, Self.width],
        tape: Tape[Self.width, Self.type],
        push: Bool = True,
    ):
        if push:
            var n = Node[Self.width, Self.type](0, 0, 0, 0.0, 0.0)
            tape.push_back(n)
        self.value = x
        self.idx = len(tape) - 1
        self.tape = tape

    @always_inline("nodebug")
    def __getitem__(self, idx: Int) -> SIMD[Self.type, 1]:
        if idx >= Self.width:
            print(
                "Performed an out of bound access on SIMD type. Width=",
                Self.width,
                ", index=",
                idx,
                ", on",
                String.write(self),
            )
            return SIMD[Self.type, 1](0.0)
        return self.value[idx]

    @always_inline("nodebug")
    def __add__(self, rhs: Number[Self.width, Self.type]) -> Number[Self.width, Self.type]:
        var n = Node[Self.width, Self.type](2, self.idx, rhs.idx, 1.0, 1.0)
        self.tape.push_back(n)
        return Number[Self.width, Self.type](
            self.value + rhs.value, self.tape, push=False
        )

    @always_inline("nodebug")
    def __add__(self, rhs: FloatLiteral) -> Number[Self.width, Self.type]:
        var n = Node[Self.width, Self.type](1, self.idx, 0, 1.0, 0.0)
        self.tape.push_back(n)
        return Number[Self.width, Self.type](self.value + rhs, self.tape, False)

    @always_inline("nodebug")
    def __gt__(self, rhs: Number[Self.width, Self.type]) -> SIMD[DType.bool, Self.width]:
        return self.value.gt(rhs.value)

    @always_inline("nodebug")
    def __lt__(self, rhs: Number[Self.width, Self.type]) -> SIMD[DType.bool, Self.width]:
        return self.value.lt(rhs.value)

    @always_inline("nodebug")
    def __sub__(self, rhs: Number[Self.width, Self.type]) -> Number[Self.width, Self.type]:
        var n = Node[Self.width, Self.type](2, self.idx, rhs.idx, 1.0, -1.0)
        self.tape.push_back(n)
        return Number[Self.width, Self.type](self.value - rhs.value, self.tape, False)

    @always_inline("nodebug")
    def __mul__(self, rhs: Number[Self.width, Self.type]) -> Number[Self.width, Self.type]:
        var n = Node[Self.width, Self.type](2, self.idx, rhs.idx, rhs.value, self.value)
        self.tape.push_back(n)
        return Number[Self.width, Self.type](self.value * rhs.value, self.tape, False)

    @always_inline("nodebug")
    def __mul__(self, rhs: Float32) -> Number[Self.width, Self.type]:
        var n = Node[Self.width, Self.type](1, self.idx, 0, rhs.cast[Self.type](), 0.0)
        self.tape.push_back(n)
        return Number[Self.width, Self.type](self.value * rhs.cast[Self.type](), self.tape, False)

    @always_inline("nodebug")
    def __mul__(self, rhs: SIMD[Self.type, 1]) -> Number[Self.width, Self.type]:
        var n = Node[Self.width, Self.type](1, self.idx, 0, rhs, 0.0)
        self.tape.push_back(n)
        return Number[Self.width, Self.type](self.value * rhs, self.tape, False)

    @always_inline("nodebug")
    def __mul__(self, rhs: SIMD[Self.type, Self.width]) -> Number[Self.width, Self.type]:
        var n = Node[Self.width, Self.type](1, self.idx, 0, rhs, 0.0)
        self.tape.push_back(n)
        return Number[Self.width, Self.type](self.value * rhs, self.tape, False)

    @always_inline("nodebug")
    def __truediv__(self, rhs: Number[Self.width, Self.type]) -> Number[Self.width, Self.type]:
        var n = Node[Self.width, Self.type](
            2,
            self.idx,
            rhs.idx,
            1.0 / rhs.value,
            -self.value / (rhs.value**2),
        )
        self.tape.push_back(n)
        return Number[Self.width, Self.type](self.value / rhs.value, self.tape, False)

    @always_inline("nodebug")
    def __truediv__(self, rhs: FloatLiteral) -> Number[Self.width, Self.type]:
        var n = Node[Self.width, Self.type](1, self.idx, 0, 1.0 / rhs, 0.0)
        self.tape.push_back(n)
        return Number[Self.width, Self.type](self.value / rhs, self.tape, False)

    @always_inline("nodebug")
    def __truediv__(self, rhs: Int) -> Number[Self.width, Self.type]:
        var rhs_f = SIMD[Self.type, Self.width](rhs)
        var n = Node[Self.width, Self.type](1, self.idx, 0, 1.0 / rhs_f, 0.0)
        self.tape.push_back(n)
        return Number[Self.width, Self.type](self.value / rhs_f, self.tape, False)


    @always_inline("nodebug")
    def __truediv__(self, rhs: SIMD[Self.type, Self.width]) -> Number[Self.width, Self.type]:
        var n = Node[Self.width, Self.type](1, self.idx, 0, (1.0 / rhs).cast[Self.type](), 0.0)
        self.tape.push_back(n)
        return Number[Self.width, Self.type](self.value / rhs, self.tape, False)

    @always_inline("nodebug")
    def __pow__(self, rhs: FloatLiteral) -> Number[Self.width, Self.type]:
        var n = Node[Self.width, Self.type](
            1, self.idx, 0, rhs * (self.value ** (rhs - 1)), 0.0
        )
        self.tape.push_back(n)
        return Number[Self.width, Self.type](self.value**rhs, self.tape, False)

    @always_inline("nodebug")
    def log(self) -> Number[Self.width, Self.type]:
        comptime assert Self.type.is_floating_point(), "log requires floating-point type"
        var res = log(self.value)
        var n = Node[Self.width, Self.type](1, self.idx, 0, 1.0 / self.value, 0.0)
        self.tape.push_back(n)
        return Number[Self.width, Self.type](res, self.tape, False)

    @always_inline("nodebug")
    def expon(self) -> Number[Self.width, Self.type]:
        comptime assert Self.type.is_floating_point(), "expon requires floating-point type"
        var res = exp(self.value)
        var n = Node[Self.width, Self.type](1, self.idx, 0, res, 0.0)
        self.tape.push_back(n)
        return Number[Self.width, Self.type](res, self.tape, False)

    @always_inline("nodebug")
    def sqrt(self) -> Number[Self.width, Self.type]:
        comptime assert Self.type.is_floating_point(), "sqrt requires floating-point type"
        var res = sqrt(self.value)
        var n = Node[Self.width, Self.type](1, self.idx, 0, 0.5 / res, 0.0)
        self.tape.push_back(n)
        return Number[Self.width, Self.type](res, self.tape, False)

    @always_inline("nodebug")
    def __neg__(self) -> Number[Self.width, Self.type]:
        return Number[Self.width, Self.type](SIMD[Self.type, Self.width](0.0), self.tape) - self

    @staticmethod
    @always_inline("nodebug")
    def from_partials(
        value: SIMD[Self.type, Self.width],
        a: Self, da: SIMD[Self.type, Self.width],
        b: Self, db: SIMD[Self.type, Self.width],
        c: Self, dc: SIMD[Self.type, Self.width],
    ) -> Self:
        """Record one node with caller-supplied local partials (a fused/custom op).

        Use for hot expressions whose Jacobian is known in closed form: one tape
        node instead of one per primitive op.
        """
        var idx = Array[Int, MAX_NODE_INPUTS](fill=0)
        var der = Array[SIMD[Self.type, Self.width], MAX_NODE_INPUTS](fill=0)
        idx[0] = a.idx
        idx[1] = b.idx
        idx[2] = c.idx
        der[0] = da
        der[1] = db
        der[2] = dc
        a.tape.push_fused(idx, der, 3)
        return Self(value, a.tape, push=False)

    @staticmethod
    @always_inline("nodebug")
    def unary(value: SIMD[Self.type, Self.width], a: Self, da: SIMD[Self.type, Self.width]) -> Self:
        """Record one node with a single input and caller-supplied partial."""
        var n = Node[Self.width, Self.type](1, a.idx, 0, da, 0.0)
        a.tape.push_back(n)
        return Self(value, a.tape, push=False)

    def grad(self) -> SIMD[Self.type, Self.width]:
        return self.tape.adjoint(self.idx)

    def mean_grad(self) -> SIMD[Self.type, 1]:
        return self.tape.adjoint(self.idx).reduce_add() / Scalar[Self.type](Self.width)

    def mean(self) -> SIMD[Self.type, 1]:
        return self.value.reduce_add() / Scalar[Self.type](Self.width)

    @always_inline("nodebug")
    def propagate_adjoints(self, starting_grad: SIMD[Self.type, Self.width] = 1.0):
        self.tape.propagate_adjoints(self.idx, starting_grad)

    def write_to(self, mut writer: Some[Writer]):
        writer.write(
            "<Number[",
            Self.width,
            ", ",
            Self.type,
            "]: value=",
            self.value,
            ">",
        )
