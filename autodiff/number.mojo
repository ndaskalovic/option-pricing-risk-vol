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

from math import log, sqrt, exp
from .node import Node
from .tape import Tape


@register_passable("trivial")
struct Number[width: Int = 1, type: DType = DType.float32](Stringable):
    var value: SIMD[type, width]
    var idx: Int
    var tape: Tape[width, type]
    alias null_tape = UnsafePointer[Tape[width, type]].alloc(1)[0]

    @always_inline("nodebug")
    fn __init__(
        inout self,
        x: SIMD[type, width],
        tape: Tape[width, type],
        push: Bool = True,
    ):
        # @parameter
        if push:
            var n = Node[width, type](0, 0, 0, 0.0, 0.0)
            tape.push_back(n)
        self.value = x
        self.idx = len(tape) - 1
        self.tape = tape

    # @always_inline("nodebug")
    # fn __init__[push: Bool = True](x: FloatLiteral, tape: Tape[width, type]) -> Self:
    #     @parameter
    #     if push:
    #         var n = Node[width, type](0, 0, 0, 0.0, 0.0)
    #         tape.push_back(n)
    #     return Self {value: SIMD[type, width](x), idx: len(tape) - 1, tape: tape}

    @always_inline("nodebug")
    fn __getitem__(self, idx: Int) -> SIMD[type, 1]:
        if idx >= width:
            print(
                "Performed an out of bound access on SIMD type. Width=",
                width,
                ", index=",
                idx,
                ", on",
                self.__str__(),
            )
            return SIMD[type, 1](0.0)
        return self.value[idx]

    # @always_inline("nodebug")
    # fn __init__(x: FloatLiteral) -> Self:
    #     var tape = Tape[width, type](0)
    #     return Self {value: x, idx: len(tape) - 1, tape: tape}

    # @always_inline("nodebug")
    # fn __init__(x: Float64) -> Self:
    #     var tape = Tape[width, type](0)
    #     return Self {value: x.cast[type](), idx: len(tape) - 1, tape: tape}

    @always_inline("nodebug")
    fn __add__(self, rhs: Number[width, type]) -> Number[width, type]:
        var n = Node[width, type](2, self.idx, rhs.idx, 1.0, 1.0)
        self.tape.push_back(n)
        return Number[width, type](
            self.value + rhs.value, self.tape, push=False
        )

    @always_inline("nodebug")
    fn __add__(self, rhs: FloatLiteral) -> Number[width, type]:
        var n = Node[width, type](1, self.idx, 0, 1.0, 0.0)
        self.tape.push_back(n)
        return Number[width, type](self.value + rhs, self.tape, False)

    @always_inline("nodebug")
    fn __gt__(self, rhs: Number[width, type]) -> SIMD[DType.bool, width]:
        return self.value > rhs.value

    @always_inline("nodebug")
    fn __lt__(self, rhs: Number[width, type]) -> SIMD[DType.bool, width]:
        return self.value < rhs.value

    @always_inline("nodebug")
    fn __sub__(self, rhs: Number[width, type]) -> Number[width, type]:
        var n = Node[width, type](2, self.idx, rhs.idx, 1.0, -1.0)
        self.tape.push_back(n)
        return Number[width, type](self.value - rhs.value, self.tape, False)

    @always_inline("nodebug")
    fn __mul__(self, rhs: Number[width, type]) -> Number[width, type]:
        var n = Node[width, type](2, self.idx, rhs.idx, rhs.value, self.value)
        self.tape.push_back(n)
        return Number[width, type](self.value * rhs.value, self.tape, False)

    @always_inline("nodebug")
    fn __mul__(self, rhs: Float32) -> Number[width, type]:
        var n = Node[width, type](1, self.idx, 0, rhs.cast[type](), 0.0)
        self.tape.push_back(n)
        return Number[width, type](self.value * rhs.cast[type](), self.tape, False)

    @always_inline("nodebug")
    fn __mul__(self, rhs: SIMD[type, 1]) -> Number[width, type]:
        var n = Node[width, type](1, self.idx, 0, rhs, 0.0)
        self.tape.push_back(n)
        return Number[width, type](self.value * rhs, self.tape, False)

    @always_inline("nodebug")
    fn __mul__(self, rhs: SIMD[type, width]) -> Number[width, type]:
        var n = Node[width, type](1, self.idx, 0, rhs, 0.0)
        self.tape.push_back(n)
        return Number[width, type](self.value * rhs, self.tape, False)

    @always_inline("nodebug")
    fn __truediv__(self, rhs: Number[width, type]) -> Number[width, type]:
        var n = Node[width, type](
            2,
            self.idx,
            rhs.idx,
            1.0 / rhs.value,
            -self.value / (rhs.value**2),
        )
        self.tape.push_back(n)
        return Number[width, type](self.value / rhs.value, self.tape, False)

    @always_inline("nodebug")
    fn __truediv__(self, rhs: FloatLiteral) -> Number[width, type]:
        var n = Node[width, type](1, self.idx, 0, 1.0 / rhs, 0.0)
        self.tape.push_back(n)
        return Number[width, type](self.value / rhs, self.tape, False)

    @always_inline("nodebug")
    fn __truediv__(self, rhs: Int) -> Number[width, type]:
        var n = Node[width, type](1, self.idx, 0, (1.0 / rhs).cast[type](), 0.0)
        self.tape.push_back(n)
        return Number[width, type](self.value / rhs, self.tape, False)

    @always_inline("nodebug")
    fn __pow__(self, rhs: FloatLiteral) -> Number[width, type]:
        var n = Node[width, type](
            1, self.idx, 0, rhs * (self.value ** (rhs - 1)), 0.0
        )
        self.tape.push_back(n)
        return Number[width, type](self.value**rhs, self.tape, False)

    @always_inline("nodebug")
    fn log(self) -> Number[width, type]:
        var res = log(self.value)
        var n = Node[width, type](1, self.idx, 0, 1.0 / self.value, 0.0)
        self.tape.push_back(n)
        return Number[width, type](res, self.tape, False)

    @always_inline("nodebug")
    fn expon(self) -> Number[width, type]:
        var res = exp(self.value)
        var n = Node[width, type](1, self.idx, 0, res, 0.0)
        self.tape.push_back(n)
        return Number[width, type](res, self.tape, False)

    @always_inline("nodebug")
    fn sqrt(self) -> Number[width, type]:
        var res = sqrt(self.value)
        var n = Node[width, type](1, self.idx, 0, 0.5 / res, 0.0)
        self.tape.push_back(n)
        return Number[width, type](res, self.tape, False)

    @always_inline("nodebug")
    fn __neg__(self) -> Number[width, type]:
        return Number[width, type](SIMD[type, width](0.0), self.tape) - self

    fn grad(self) -> SIMD[type, width]:
        return self.tape.adjoint(self.idx)

    fn mean_grad(self) -> SIMD[type, 1]:
        return self.tape.adjoint(self.idx).reduce_add[1]() / width

    fn mean(self) -> SIMD[type, 1]:
        return self.value.reduce_add[1]() / width

    @always_inline("nodebug")
    fn propagate_adjoints(self, starting_grad: SIMD[type, width] = 1.0) -> None:
        self.tape.propagate_adjoints(self.idx, starting_grad)

    fn __str__(self) -> String:
        return (
            "<Number["
            + str(width)
            + ", "
            + str(type)
            + "]: value="
            + str(self.value)
            + ">"
        )
