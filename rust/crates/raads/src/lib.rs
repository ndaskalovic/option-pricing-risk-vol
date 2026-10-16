//! Compile-time adjoint algorithmic differentiation for option pricing.
//!
//! Write a kernel once, generic over [`AdNum`]; `#[kernel]` emits straight-line forward and
//! reverse code generic over [`Real`]. Engines own the runtime loops (paths, time steps,
//! grids) and return the price and every first-order Greek in one run; running them on
//! [`Dual`] values adds Hessian-vector products.
//!
//! # Build flags
//!
//! The SIMD backend is chosen at compile time from the enabled target features (no runtime
//! dispatch). aarch64 always uses NEON. **On x86_64, build with `-C target-cpu=x86-64-v3`
//! (AVX2 + FMA), or `native` / `x86-64-v4` for AVX-512**; default x86_64 builds use the
//! portable fallback (no SIMD `exp`, no FMA), which is much slower. For example in
//! `.cargo/config.toml`:
//!
//! ```toml
//! [target.'cfg(target_arch = "x86_64")']
//! rustflags = ["-C", "target-cpu=x86-64-v3"]
//! ```
//!
//! # Monte Carlo
//!
//! A Black-Scholes call in log space, priced with every Greek in one pass. Lanes of the
//! value type are paths; normals are supplied by the caller (`f32`, laid out per
//! [`mc_greeks`]).
//!
//! ```
//! use raads::prelude::*;
//!
//! pub struct GbmCall;
//!
//! // params: s0, r, sigma, T, K
//! #[mc_model]
//! impl GbmCall {
//!     pub fn setup<N: AdNum>(p: [N; 5], #[data] n_steps: N) -> [N; 2] {
//!         let dt = p[3] / n_steps;
//!         [(p[1] - 0.5 * p[2] * p[2]) * dt, p[2] * dt.sqrt()]
//!     }
//!     pub fn init<N: AdNum>(p: [N; 5]) -> [N; 1] {
//!         [p[0].ln()]
//!     }
//!     pub fn step<N: AdNum>(s: [N; 1], c: [N; 2], #[data] z: [N; 1]) -> [N; 1] {
//!         [s[0] + c[0] + c[1] * z[0]]
//!     }
//!     pub fn payoff<N: AdNum>(s: [N; 1], p: [N; 5], _c: [N; 2]) -> N {
//!         (s[0].exp() - p[4]).relu() * (-p[1] * p[3]).exp()
//!     }
//! }
//!
//! # fn normals(n: usize) -> Vec<f32> {
//! #     // LCG + Box-Muller: deterministic, no dependencies
//! #     let mut state = 0x2545_f491_4f6c_dd1d_u64;
//! #     let mut uniform = move || {
//! #         state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
//! #         ((state >> 11) as f64 + 0.5) / (1u64 << 53) as f64
//! #     };
//! #     (0..n.div_ceil(2))
//! #         .flat_map(|_| {
//! #             let (r, t) = ((-2.0 * uniform().ln()).sqrt(), std::f64::consts::TAU * uniform());
//! #             [(r * t.cos()) as f32, (r * t.sin()) as f32]
//! #         })
//! #         .take(n)
//! #         .collect()
//! # }
//! let (n_paths, n_steps) = (65_536, 4);
//! let z = normals(n_paths * n_steps);
//! let params = [100.0, 0.01, 0.2, 1.0, 100.0].map(F64x4::from);
//! let r = mc_greeks::<GbmCall, F64x4>(&params, n_steps, n_paths, &z);
//! // Black-Scholes: price 8.433, delta 0.560
//! assert!((r.price - 8.433).abs() < 4.0 * r.std_err);
//! assert!((r.grad[0] - 0.560).abs() < 0.02);
//! ```
//!
//! # Finite differences
//!
//! An American put on a Crank–Nicolson grid in `x = log S`. Lanes of the value type are
//! independent problems (here one, `f64`).
//!
//! ```
//! use raads::prelude::*;
//!
//! pub struct AmericanPut;
//!
//! // params: s0, r, sigma, T, K
//! #[fd_model]
//! impl AmericanPut {
//!     const AMERICAN: bool = true;
//!     pub fn coeffs<N: AdNum>(#[data] _x: N, p: [N; 5]) -> [N; 3] {
//!         let (r, sigma) = (p[1], p[2]);
//!         [0.5 * sigma * sigma, r - 0.5 * sigma * sigma, r]
//!     }
//!     pub fn payoff<N: AdNum>(#[data] x: N, p: [N; 5]) -> N {
//!         (p[4] - x.exp()).relu()
//!     }
//!     pub fn spot<N: AdNum>(p: [N; 5]) -> N {
//!         p[0].ln()
//!     }
//!     pub fn maturity<N: AdNum>(p: [N; 5]) -> N {
//!         p[3]
//!     }
//! }
//!
//! let grid = FdGrid::new(100f64.ln() - 1.2, 100f64.ln() + 1.2, 201, 200);
//! let r = fd_greeks::<AmericanPut, f64>(&[100.0, 0.01, 0.2, 1.0, 100.0], &grid);
//! assert!((r.price - 7.51).abs() < 0.02);
//! assert!(r.grad[0] < 0.0); // delta
//! ```

#![warn(missing_docs)]

extern crate self as raads;

mod adnum;
mod arr;
pub mod fd;
pub mod lanes;
pub mod mc;
mod real;
pub mod surface;
mod trace;

pub use adnum::AdNum;
pub use arr::Arr;
pub use fd::{FdGrid, FdModel, FdResult, fd_greeks, fd_price};
pub use lanes::Lanes;
pub use mc::{McModel, McPrice, McResult, mc_greeks, mc_price};
pub use raads_macros::{fd_model, kernel, mc_model};
pub use real::{Dual, Real};
pub use surface::{FdBook, FdSurface, SurfaceGrid, SurfaceOption, SurfacePoint, fd_surface, fd_surface_book};

/// Everything a model or a pricing call needs: `use raads::prelude::*;`.
pub mod prelude {
    pub use crate::lanes::{F32x4, F32x8, F32x16, F32x32, F64x2, F64x4, F64x8, F64x16};
    pub use crate::{AdNum, Arr, Dual, Lanes, Real, fd_model, kernel, mc_model};
    pub use crate::{FdBook, FdSurface, SurfaceGrid, SurfaceOption, SurfacePoint, fd_surface, fd_surface_book};
    pub use crate::{FdGrid, FdModel, FdResult, fd_greeks, fd_price};
    pub use crate::{McModel, McPrice, McResult, mc_greeks, mc_price};
}

/// Used by macro-generated code and tests; not public API.
#[doc(hidden)]
pub mod __private {
    pub use crate::trace::{Tracer, trace};
    pub use raads_codegen as codegen;
}

/// Compiles and runs the README example.
#[cfg(doctest)]
#[doc = include_str!("../../../README.md")]
struct ReadmeDoctest;

mod sealed {
    /// Supertrait of [`AdNum`](crate::AdNum) (and so of [`Real`](crate::Real) and
    /// [`Lanes`](crate::Lanes)) and of [`Arr`](crate::Arr): only this crate implements them.
    pub trait Sealed {}
}
