//! Compile-time adjoint algorithmic differentiation for option pricing.
//!
//! Write a kernel once, generic over [`AdNum`]; `#[kernel]` emits straight-line forward and
//! reverse code generic over [`Real`]. Engines own the runtime loops (paths, time steps,
//! grids) and return the price and every first-order Greek in one run; running them on
//! [`Dual`] values adds Hessian-vector products.

extern crate self as ctad;

mod adnum;
mod arr;
pub mod lanes;
pub mod mc;
mod real;
pub mod trace;

pub use adnum::AdNum;
pub use arr::Arr;
pub use ctad_codegen as codegen;
pub use lanes::Lanes;
pub use real::{Dual, Real};
pub use ctad_macros::{kernel, mc_model};
pub use mc::{McModel, McResult, mc_greeks};
