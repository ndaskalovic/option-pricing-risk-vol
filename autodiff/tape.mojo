# Copyright (c) 2024 Nicholas Daskalovic
# All rights reserved.
#
# This code is proprietary and confidential. Permission is granted to view
# this source code for personal, educational purposes only. Any other use,
# including copying, modification, distribution, or use of any portion of
# this source code, in whole or in part, without express written permission
# is strictly prohibited.

'''
This file contains the Tape structure used to store the computation graph
of a program. It enables automatic differentiation through the backpropagation
algorithm.
'''

from .node import Node
from std.memory import memcpy, memset_zero


struct Tape[width: Int = 1, type: DType = DType.float32](TrivialRegisterPassable, Sized):
    var _data: UnsafePointer[Node[Self.width, Self.type], MutExternalOrigin]
    var _adjoints: UnsafePointer[SIMD[Self.type, Self.width], MutExternalOrigin]
    var _size: UnsafePointer[Int, MutExternalOrigin]
    var _tail: UnsafePointer[Int, MutExternalOrigin]
    var _mark: UnsafePointer[Int, MutExternalOrigin]
    var _frozen: Bool

    @always_inline("nodebug")
    def __init__(out self, size: Int = 100_000):
        self._data = alloc[Node[Self.width, Self.type]](size)
        self._size = alloc[Int](1)
        self._tail = alloc[Int](1)
        self._mark = alloc[Int](1)
        self._adjoints = alloc[SIMD[Self.type, Self.width]](size)
        memset_zero(self._adjoints, size)
        self._size[] = size
        self._tail[] = -1
        self._mark[] = 0
        self._frozen = False

    @always_inline("nodebug")
    def __setitem__(self, idx: Int, node: Node[Self.width, Self.type]):
        self._data[idx] = node

    @always_inline("nodebug")
    def __getitem__(self, idx: Int) -> Node[Self.width, Self.type]:
        return self._data[idx]

    @always_inline("nodebug")
    def get_tail(self) -> Int:
        return self._tail[]

    @always_inline("nodebug")
    def get_size(self) -> Int:
        return self._size[]

    @always_inline("nodebug")
    def rewind_to_mark(self):
        self._tail[] = self._mark[]

    @always_inline("nodebug")
    def set_mark(self):
        self._mark[] = self._tail[]

    @always_inline("nodebug")
    def push_back(self, node: Node[Self.width, Self.type]):
        self._tail[] += 1
        self[self._tail[]] = node

    @always_inline("nodebug")
    def __len__(self) -> Int:
        return self._tail[] + 1

    @always_inline("nodebug")
    def free(self):
        self._data.free()
        self._size.free()
        self._tail.free()
        self._mark.free()
        self._adjoints.free()

    def print(self):
        print(" Position | Node Info ")
        print("----------|-----------")
        for i in range(self._tail[] + 1):
            print(String.write(i) + " | ", String.write(self._data[i]))


    @always_inline("nodebug")
    def adjoint(self, idx: Int) -> SIMD[Self.type, Self.width]:
        return self._adjoints[idx]


    @always_inline("nodebug")
    def propagate_adjoints(self, idx: Int, start_grad: SIMD[Self.type, Self.width] = 1.0):
        memset_zero(
            self._adjoints + self._mark[] + 1, len(self) - self._mark[]
        )
        var N = idx
        self._adjoints[N] = start_grad
        for i in range(N, 0, -1):
            var node = self[i]
            if node.numArg > 0:
                self._adjoints[node.idx1] = self._adjoints[node.idx1] + (self._adjoints[i] * node.der1)
                if node.numArg > 1:
                    self._adjoints[node.idx2] = self._adjoints[node.idx2] + (self._adjoints[i] * node.der2)
