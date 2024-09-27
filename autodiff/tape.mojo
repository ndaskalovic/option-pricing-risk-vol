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
from memory import memcpy, memset_zero


@register_passable("trivial")
struct Tape[width: Int = 1, type: DType = DType.float32](Sized):
    var _data: UnsafePointer[Node[width, type]]
    var _adjoints: UnsafePointer[SIMD[type, width]]
    var _size: UnsafePointer[Int]
    var _tail: UnsafePointer[Int]
    var _mark: UnsafePointer[Int]
    var _frozen: Bool

    @always_inline("nodebug")
    fn __init__(inout self, size: Int = 100_000):
        self._data = UnsafePointer[Node[width, type]].alloc(size)
        self._size = UnsafePointer[Int].alloc(1)
        self._tail = UnsafePointer[Int].alloc(1)
        self._mark = UnsafePointer[Int].alloc(1)
        self._adjoints = UnsafePointer[SIMD[type, width]].alloc(size)
        memset_zero(self._adjoints, size)
        self._size[] = size
        self._tail[] = -1
        self._mark[] = 0
        self._frozen = False

    @always_inline("nodebug")
    fn __setitem__(self, idx: Int, node: Node[width, type]) -> None:
        self._data[idx] = node

    @always_inline("nodebug")
    fn __getitem__(self, idx: Int) -> Node[width, type]:
        return self._data[idx]

    @always_inline("nodebug")
    fn get_tail(self) -> Int:
        return self._tail[]

    # @always_inline("nodebug")
    # fn freeze(inout self) -> None:
    #     self._frozen = True

    # @always_inline("nodebug")
    # fn unfreeze(inout self) -> None:
    #     self._frozen = False

    # @always_inline("nodebug")
    # fn set_tail(self, tail: Int) -> None:
    #     self._tail.store(tail)

    @always_inline("nodebug")
    fn get_size(self) -> Int:
        return self._size[]

    # @always_inline("nodebug")
    # fn set_size(self, size: Int) -> None:
    #     self._size.store(size)

    # @always_inline("nodebug")
    # fn rewind(self, idx: Int) -> None:
    #     self._tail.store(idx)

    @always_inline("nodebug")
    fn rewind_to_mark(self) -> None:
        self._tail[] = self._mark[]

    @always_inline("nodebug")
    fn set_mark(self) -> None:
        self._mark[] = self._tail[]

    @always_inline("nodebug")
    fn push_back(self, node: Node[width, type]) -> None:
        # if self._frozen:
        #     return
        self._tail[] += 1
        self[self._tail[]] = node

    @always_inline("nodebug")
    fn __len__(self) -> Int:
        return self._tail[] + 1

    # @always_inline("nodebug")
    # fn free(self):
    #     self._data.free()
    #     self._size.free()
    #     self._tail.free()
    #     self._adjoints.free()

    fn print(self):
        print(" Position | Node Info ")
        print("----------|-----------")
        for i in range(self._tail[] + 1):
            print(str(i) + " | ", str(self._data[i]))

    
    @always_inline("nodebug")
    fn adjoint(self, idx: Int) -> SIMD[type, width]:
        return self._adjoints[idx]
            

    @always_inline("nodebug")
    fn propagate_adjoints(self, idx: Int, start_grad: SIMD[type, width] = 1.0):
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
