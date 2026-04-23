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
from std.sys.info import size_of


struct Tape[width: Int = 1, type: DType = DType.float32](TrivialRegisterPassable, Sized):
    # Topology arrays (frozen after compile)
    var _num_args: UnsafePointer[Int, MutExternalOrigin]
    var _idx1: UnsafePointer[Int, MutExternalOrigin]
    var _idx2: UnsafePointer[Int, MutExternalOrigin]
    # Derivative arrays (mutable per path)
    var _der1: UnsafePointer[SIMD[Self.type, Self.width], MutExternalOrigin]
    var _der2: UnsafePointer[SIMD[Self.type, Self.width], MutExternalOrigin]
    var _adjoints: UnsafePointer[SIMD[Self.type, Self.width], MutExternalOrigin]
    var _size: UnsafePointer[Int, MutExternalOrigin]
    var _tail: UnsafePointer[Int, MutExternalOrigin]
    var _mark: UnsafePointer[Int, MutExternalOrigin]
    var _frozen: Bool
    var _peak_nodes: UnsafePointer[Int, MutExternalOrigin]
    var _total_pushes: UnsafePointer[Int, MutExternalOrigin]
    var _compiled: UnsafePointer[Bool, MutExternalOrigin]
    var _replay_idx: UnsafePointer[Int, MutExternalOrigin]

    @always_inline("nodebug")
    def __init__(out self, size: Int = 100_000):
        self._num_args = alloc[Int](size)
        self._idx1 = alloc[Int](size)
        self._idx2 = alloc[Int](size)
        self._der1 = alloc[SIMD[Self.type, Self.width]](size)
        self._der2 = alloc[SIMD[Self.type, Self.width]](size)
        self._size = alloc[Int](1)
        self._tail = alloc[Int](1)
        self._mark = alloc[Int](1)
        self._adjoints = alloc[SIMD[Self.type, Self.width]](size)
        memset_zero(self._adjoints, size)
        self._size[] = size
        self._tail[] = -1
        self._mark[] = 0
        self._frozen = False
        self._peak_nodes = alloc[Int](1)
        self._total_pushes = alloc[Int](1)
        self._peak_nodes[] = 0
        self._total_pushes[] = 0
        self._compiled = alloc[Bool](1)
        self._replay_idx = alloc[Int](1)
        self._compiled[] = False
        self._replay_idx[] = 0

    @always_inline("nodebug")
    def __setitem__(self, idx: Int, node: Node[Self.width, Self.type]):
        self._num_args[idx] = node.numArg
        self._idx1[idx] = node.idx1
        self._idx2[idx] = node.idx2
        self._der1[idx] = node.der1
        self._der2[idx] = node.der2

    @always_inline("nodebug")
    def __getitem__(self, idx: Int) -> Node[Self.width, Self.type]:
        return Node[Self.width, Self.type](
            self._num_args[idx], self._idx1[idx], self._idx2[idx],
            self._der1[idx], self._der2[idx]
        )

    @always_inline("nodebug")
    def get_tail(self) -> Int:
        return self._tail[]

    @always_inline("nodebug")
    def get_size(self) -> Int:
        return self._size[]

    @always_inline("nodebug")
    def rewind_to_mark(self):
        self._tail[] = self._mark[]
        if self._compiled[]:
            self._replay_idx[] = self._mark[]

    @always_inline("nodebug")
    def set_mark(self):
        self._mark[] = self._tail[]

    @always_inline("nodebug")
    def compile(self):
        """Freeze tape topology after first path."""
        self._compiled[] = True

    @always_inline("nodebug")
    def is_compiled(self) -> Bool:
        return self._compiled[]

    @always_inline("nodebug")
    def start_replay(self):
        """Reset replay counter to mark position for next path."""
        self._replay_idx[] = self._mark[]

    @always_inline("nodebug")
    def push_back(self, node: Node[Self.width, Self.type]):
        if self._compiled[]:
            # Replay mode: SoA — only write derivatives, topology is frozen
            self._replay_idx[] += 1
            var idx = self._replay_idx[]
            self._der1[idx] = node.der1
            self._der2[idx] = node.der2
            self._tail[] = idx
            self._total_pushes[] += 1
            return
        # Recording mode: write everything
        self._tail[] += 1
        self[self._tail[]] = node
        self._total_pushes[] += 1
        if self._tail[] + 1 > self._peak_nodes[]:
            self._peak_nodes[] = self._tail[] + 1

    @always_inline("nodebug")
    def __len__(self) -> Int:
        return self._tail[] + 1

    @always_inline("nodebug")
    def free(self):
        self._num_args.free()
        self._idx1.free()
        self._idx2.free()
        self._der1.free()
        self._der2.free()
        self._size.free()
        self._tail.free()
        self._mark.free()
        self._adjoints.free()
        self._peak_nodes.free()
        self._total_pushes.free()
        self._compiled.free()
        self._replay_idx.free()

    @always_inline("nodebug")
    def peak_nodes(self) -> Int:
        return self._peak_nodes[]

    @always_inline("nodebug")
    def total_pushes(self) -> Int:
        return self._total_pushes[]

    @always_inline("nodebug")
    def peak_adjoint_bytes(self) -> Int:
        return self._peak_nodes[] * size_of[SIMD[Self.type, Self.width]]()

    @always_inline("nodebug")
    def reset_metrics(self):
        self._peak_nodes[] = 0
        self._total_pushes[] = 0

    def print(self):
        print(" Position | Node Info ")
        print("----------|-----------")
        for i in range(self._tail[] + 1):
            print(String.write(i) + " | ", String.write(self[i]))


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
            # Skip nodes with zero adjoint across all SIMD lanes
            var adj = self._adjoints[i]
            if adj == 0:
                continue
            # SoA: read directly from separate arrays
            var na = self._num_args[i]
            if na > 0:
                var i1 = self._idx1[i]
                self._adjoints[i1] = self._adjoints[i1] + (adj * self._der1[i])
                if na > 1:
                    var i2 = self._idx2[i]
                    self._adjoints[i2] = self._adjoints[i2] + (adj * self._der2[i])
