# Future AAD Optimizations

Remaining improvements identified during the modernization effort, ranked by estimated impact.

## Expression Templates / Operation Fusion

Savine's flagship optimization (AADExpr.h, chapters 14-15). Instead of recording every +, *, / as a separate tape node, build a compile-time expression tree that collapses an entire expression into a single multi-input tape node. The inner MC loop expression `path * (drift + vsdt * randoms).expon()` currently creates ~5 nodes per time step. With fusion: 1-2 nodes. Over 52 steps x 200k paths = ~52M recordings reduced to ~10-20M.

Mojo's parametric types and `comptime` features can implement CRTP-equivalent patterns more cleanly than C++. This is the highest-impact optimization but the largest architectural change.

Reference: Savine's CompFinance repo, AADExpr.h

## Block Arena Allocator

Replace fixed pre-allocation with a linked list of fixed-size blocks (16k nodes per block). No reallocation needed, no size guessing, cache-friendly contiguous access within blocks. Grows on demand.

Reference: Savine's blocklist.h

## Constant Folding / Activity Analysis

Inspired by Enzyme. Skip taping variables that don't need gradients. If dP/dK is not needed, K can be plain SIMD instead of Number. Review which variables truly require gradient tracking for each pricing scenario.

## Unary Node Optimization

For log, exp, sqrt: idx2/der2 storage is wasted. Options:
- Tagged node variants (UnaryNode vs BinaryNode)
- Carry scalar multipliers lazily for scalar-times-Number operations
- Skip tape recording entirely for constant-factor multiplications

## Parallel Path Processing

Thread-local tapes with reduction. Each thread runs its own MC paths and accumulates into local adjoint buffers, reduced at the end. Natural parallelism since paths are independent.

Reference: Savine's threadPool.h

## Checkpointing (Griewank's Revolve Algorithm)

Trade recomputation for memory: O(sqrt(N)) tape memory instead of O(N). Not critical at 52 timesteps but essential for daily steps (252), exotic path-dependent products, or LSM American options where tape length per path can be 10,000+.

## Tapeless Forward Mode

Dual numbers with no tape for single-Greek scenarios (e.g., just delta). O(1) memory overhead, naturally SIMD-friendly. Reverse mode still wins when all Greeks are needed simultaneously, but for production scenarios needing only one Greek, forward mode is significantly faster.
