# Mojo vs Rust (build.rs codegen, Enzyme): GBM call, log space, price + 5 Greeks

Same inputs everywhere: `mojo_ref.mojo` dumps the normals (seed 7, W=32 layout, 200k × 52) to `normals_w32.f32` (41.6 MB, not committed).

```bash
../../.venv/bin/mojo run -I ../.. mojo_ref.mojo                       # dump normals + Mojo reference results
cargo +stable build --release --bin codegen && ./target/release/codegen --bench
rustup component add enzyme --toolchain nightly
RUSTFLAGS="-Zautodiff=Enable -C target-cpu=native" cargo +nightly build --release --features enzyme --bin enzyme --target-dir target-enzyme
./target-enzyme/release/enzyme --bench
../../.venv/bin/mojo build -I ../.. mojo_bench.mojo -o mojo_bench && ./mojo_bench
```

- `build.rs`: traces `src/model.rs` (generic over `AdNum`) and emits straight-line fwd/bwd into `OUT_DIR/kernels.rs`. Same analysis as the Mojo `Compiled`; works on stable Rust.
- `enzyme`: plain Rust pricer with `#[autodiff_reverse(d_price, Duplicated, Const, Active)]`.

Results (2026-10-15, Apple M5, f32, single core, idle machine, best of 31, three alternating rounds):

| Variant | Price + 5 Greeks (ms) | vs Mojo | Pricing only (ms) |
|---|---|---|---|
| **Mojo** compiled engine (compile-time graph) | **0.926–0.951** | 1.00× | 0.550–0.563 |
| **Rust build.rs**, `std::simd` lanes + explicit FMA (nightly) | **0.979–0.993** | **1.06×** | 0.578–0.587 |
| Rust build.rs, `[f32; 32]` lanes + FMA (stable) | 1.357–1.376 | 1.47× | 0.596–0.605 |
| Rust Enzyme, `std::simd` + FMA | 1.792–1.826 | 1.94× | (same code as above) |
| Rust Enzyme, plain lane loops | 2.783–2.870 | 3.0× | 1.49–1.84 |
| Mojo legacy runtime tape (reference) | 3.30–3.36 | 3.6× | |

Tuning steps that mattered for Rust (each checked in the disassembly):

1. `std::simd::Simd<f32, 32>` instead of `[f32; 32]`.
2. Explicit `mul_add` in generated code. Rust never contracts `a*b + c`; Mojo/LLVM does.
3. Products fused straight into the caller's adjoint accumulators.
4. No per-step copies or bounds-checked slices.

The inner forward and backward loops end at 8 fmla + 8 fadd + 4 ldp per step per 8 registers.

All variants agree on price and delta. Enzyme's dK differs by ≤2e-4 relative (f32 adjoint accumulation).
