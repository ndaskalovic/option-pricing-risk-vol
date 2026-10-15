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

from std.memory.alloc import unsafe_alloc
from .node import Node
from std.memory import unsafe_memcpy, unsafe_memset_zero
from std.sys.info import size_of
from std.os import abort

# Maximum inputs per node (supports fused expression templates up to 4 inputs).
comptime MAX_NODE_INPUTS = 4


struct Tape[width: Int = 1, type: DType = DType.float32](TrivialRegisterPassable, Sized):
    # Topology arrays (frozen after compile)
    var _num_args: Pointer[Int, MutUntrackedOrigin]
    var _idx1: Pointer[Int, MutUntrackedOrigin]
    var _idx2: Pointer[Int, MutUntrackedOrigin]
    var _idx3: Pointer[Int, MutUntrackedOrigin]
    var _idx4: Pointer[Int, MutUntrackedOrigin]
    # Derivative arrays (mutable per path)
    var _der1: Pointer[SIMD[Self.type, Self.width], MutUntrackedOrigin]
    var _der2: Pointer[SIMD[Self.type, Self.width], MutUntrackedOrigin]
    var _der3: Pointer[SIMD[Self.type, Self.width], MutUntrackedOrigin]
    var _der4: Pointer[SIMD[Self.type, Self.width], MutUntrackedOrigin]
    var _adjoints: Pointer[SIMD[Self.type, Self.width], MutUntrackedOrigin]
    var _size: Pointer[Int, MutUntrackedOrigin]
    var _tail: Pointer[Int, MutUntrackedOrigin]
    var _mark: Pointer[Int, MutUntrackedOrigin]
    var _frozen: Bool
    var _peak_nodes: Pointer[Int, MutUntrackedOrigin]
    var _total_pushes: Pointer[Int, MutUntrackedOrigin]
    var _leaves_above_mark: Pointer[Int, MutUntrackedOrigin]
    var _compiled: Pointer[Bool, MutUntrackedOrigin]

    @always_inline("nodebug")
    def __init__(out self, size: Int = 100_000):
        # Single contiguous block for all 5 topology Int arrays
        var topo = unsafe_alloc[Int](5 * size)
        self._num_args = topo
        self._idx1 = topo.unsafe_offset(size)
        self._idx2 = topo.unsafe_offset(2 * size)
        self._idx3 = topo.unsafe_offset(3 * size)
        self._idx4 = topo.unsafe_offset(4 * size)
        # Single contiguous block for all 5 SIMD arrays (4 derivs + adjoints)
        var simd = unsafe_alloc[SIMD[Self.type, Self.width]](5 * size)
        self._der1 = simd
        self._der2 = simd.unsafe_offset(size)
        self._der3 = simd.unsafe_offset(2 * size)
        self._der4 = simd.unsafe_offset(3 * size)
        self._adjoints = simd.unsafe_offset(4 * size)
        unsafe_memset_zero(self._adjoints, size)
        # Control variables (shared mutable state via heap pointers)
        self._size = unsafe_alloc[Int](1)
        self._tail = unsafe_alloc[Int](1)
        self._mark = unsafe_alloc[Int](1)
        self._size[] = size
        self._tail[] = -1
        self._mark[] = 0
        self._frozen = False
        self._peak_nodes = unsafe_alloc[Int](1)
        self._total_pushes = unsafe_alloc[Int](1)
        self._leaves_above_mark = unsafe_alloc[Int](1)
        self._leaves_above_mark[] = 0
        self._peak_nodes[] = 0
        self._total_pushes[] = 0
        self._compiled = unsafe_alloc[Bool](1)
        self._compiled[] = False

    @always_inline("nodebug")
    def __setitem__(self, idx: Int, node: Node[Self.width, Self.type]):
        self._num_args[unsafe_offset=idx] = node.numArg
        self._idx1[unsafe_offset=idx] = node.idx1
        self._idx2[unsafe_offset=idx] = node.idx2
        self._der1[unsafe_offset=idx] = node.der1
        self._der2[unsafe_offset=idx] = node.der2

    @always_inline("nodebug")
    def __getitem__(self, idx: Int) -> Node[Self.width, Self.type]:
        return Node[Self.width, Self.type](
            self._num_args[unsafe_offset=idx], self._idx1[unsafe_offset=idx], self._idx2[unsafe_offset=idx],
            self._der1[unsafe_offset=idx], self._der2[unsafe_offset=idx]
        )

    @always_inline("nodebug")
    def get_tail(self) -> Int:
        return self._tail[]

    @always_inline("nodebug")
    def get_size(self) -> Int:
        return self._size[]

    @always_inline("nodebug")
    def rewind_to_mark(self):
        # Pushes are counted here rather than per push to keep the hot path lean
        self._total_pushes[] += self._tail[] - self._mark[]
        # Leaf adjoints are not cleared by the backward sweep (so their gradients stay
        # readable); zero the region before reuse only if leaves were pushed above the mark.
        if self._leaves_above_mark[] > 0:
            unsafe_memset_zero(self._adjoints.unsafe_offset(self._mark[] + 1), self._tail[] - self._mark[])
            self._leaves_above_mark[] = 0
        self._tail[] = self._mark[]

    @always_inline("nodebug")
    def set_mark(self):
        self._mark[] = self._tail[]

    @always_inline("nodebug")
    def compile(self):
        """Freeze tape topology after first path. Replay reuses `_tail` as the cursor,
        so subsequent paths (each starting from rewind_to_mark) overwrite only derivatives."""
        self._compiled[] = True

    @always_inline("nodebug")
    def is_compiled(self) -> Bool:
        return self._compiled[]

    @always_inline("nodebug")
    def push_fused(
        self,
        indices: Array[Int, MAX_NODE_INPUTS],
        derivs: Array[SIMD[Self.type, Self.width], MAX_NODE_INPUTS],
        count: Int,
    ):
        """Record a fused multi-input node (up to 4 inputs) using SoA arrays."""
        if self._compiled[]:
            self._tail[] += 1
            var slot = self._tail[]
            # Replay: only write derivatives, topology is frozen
            self._der1[unsafe_offset=slot] = derivs[0]
            if count > 1:
                self._der2[unsafe_offset=slot] = derivs[1]
            if count > 2:
                self._der3[unsafe_offset=slot] = derivs[2]
            if count > 3:
                self._der4[unsafe_offset=slot] = derivs[3]
            return
        # Recording mode: write topology + derivatives
        self._check_capacity()
        self._tail[] += 1
        var slot = self._tail[]
        self._num_args[unsafe_offset=slot] = count
        self._idx1[unsafe_offset=slot] = indices[0]
        self._der1[unsafe_offset=slot] = derivs[0]
        if count > 1:
            self._idx2[unsafe_offset=slot] = indices[1]
            self._der2[unsafe_offset=slot] = derivs[1]
        if count > 2:
            self._idx3[unsafe_offset=slot] = indices[2]
            self._der3[unsafe_offset=slot] = derivs[2]
        if count > 3:
            self._idx4[unsafe_offset=slot] = indices[3]
            self._der4[unsafe_offset=slot] = derivs[3]
        if self._tail[] + 1 > self._peak_nodes[]:
            self._peak_nodes[] = self._tail[] + 1

    @always_inline("nodebug")
    def push_back(self, node: Node[Self.width, Self.type]):
        if self._compiled[]:
            # Replay mode: SoA — only write derivatives, topology is frozen
            self._tail[] += 1
            var idx = self._tail[]
            self._der1[unsafe_offset=idx] = node.der1
            self._der2[unsafe_offset=idx] = node.der2
            if node.numArg == 0:
                self._leaves_above_mark[] += 1
            return
        # Recording mode: write everything
        self._check_capacity()
        self._tail[] += 1
        self[self._tail[]] = node
        if node.numArg == 0 and self._tail[] > self._mark[]:
            self._leaves_above_mark[] += 1
        if self._tail[] + 1 > self._peak_nodes[]:
            self._peak_nodes[] = self._tail[] + 1

    @always_inline("nodebug")
    def _check_capacity(self):
        if self._tail[] + 1 >= self._size[]:
            abort("Tape capacity exceeded; construct Tape with a larger size")

    @always_inline("nodebug")
    def __len__(self) -> Int:
        return self._tail[] + 1

    @always_inline("nodebug")
    def free(self):
        # Free the two contiguous blocks (base pointers only)
        self._num_args.unsafe_free()   # frees entire topo block (idx1-4 are offsets into it)
        self._der1.unsafe_free()       # frees entire SIMD block (der2-4, adjoints are offsets)
        # Free control variables
        self._size.unsafe_free()
        self._tail.unsafe_free()
        self._mark.unsafe_free()
        self._peak_nodes.unsafe_free()
        self._total_pushes.unsafe_free()
        self._leaves_above_mark.unsafe_free()
        self._compiled.unsafe_free()

    @always_inline("nodebug")
    def peak_nodes(self) -> Int:
        return self._peak_nodes[]

    @always_inline("nodebug")
    def total_pushes(self) -> Int:
        """Nodes pushed above the mark, summed over completed rewind_to_mark() calls."""
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
        return self._adjoints[unsafe_offset=idx]


    @always_inline("nodebug")
    def propagate_adjoints(self, idx: Int, start_grad: SIMD[Self.type, Self.width] = 1.0):
        """Backward pass from `idx` down to (excluding) the mark.

        Adjoints of nodes at or below the mark (inputs and path-invariant
        setup) accumulate across calls; finish with propagate_mark_to_start().
        """
        # Adjoints above the mark are kept zero between calls: _backward[clear=True]
        # zeroes each one as it is consumed, so no memset of the region is needed.
        self._adjoints[unsafe_offset=idx] = start_grad
        self._backward[clear=True](idx, self._mark[])

    @always_inline("nodebug")
    def propagate_mark_to_start(self):
        """Propagate accumulated adjoints through nodes at or below the mark (once, after all paths)."""
        self._backward[clear=False](self._mark[], 0)

    @always_inline("nodebug")
    def _backward[clear: Bool](self, start: Int, stop: Int):
        for i in range(start, stop, -1):
            # No all-lanes-zero skip: with W paths per lane group it almost never fires,
            # and the W-wide compare+reduce costs ~25 instructions per node (-7% at W=32).
            var adj = self._adjoints[unsafe_offset=i]
            var na = self._num_args[unsafe_offset=i]
            comptime if clear:
                if na > 0:  # leaves keep their adjoint (gradient readable after the sweep)
                    self._adjoints[unsafe_offset=i] = 0
            # Unrolled: no loops, direct SoA reads for each input slot
            if na > 0:
                var i1 = self._idx1[unsafe_offset=i]
                self._adjoints[unsafe_offset=i1] = self._adjoints[unsafe_offset=i1] + (adj * self._der1[unsafe_offset=i])
            if na > 1:
                var i2 = self._idx2[unsafe_offset=i]
                self._adjoints[unsafe_offset=i2] = self._adjoints[unsafe_offset=i2] + (adj * self._der2[unsafe_offset=i])
            if na > 2:
                var i3 = self._idx3[unsafe_offset=i]
                self._adjoints[unsafe_offset=i3] = self._adjoints[unsafe_offset=i3] + (adj * self._der3[unsafe_offset=i])
            if na > 3:
                var i4 = self._idx4[unsafe_offset=i]
                self._adjoints[unsafe_offset=i4] = self._adjoints[unsafe_offset=i4] + (adj * self._der4[unsafe_offset=i])
