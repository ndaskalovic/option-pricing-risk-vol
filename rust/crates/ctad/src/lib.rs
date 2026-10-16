//! Compile-time adjoint algorithmic differentiation for option pricing.
//!
//! Write a kernel once, generic over [`AdNum`]; `#[kernel]` emits straight-line forward and
//! reverse code generic over [`Real`]. Engines own the runtime loops (paths, time steps,
//! grids) and return the price and every first-order Greek in one run; running them on
//! [`Dual`] values adds Hessian-vector products.

extern crate self as ctad;

mod adnum;
mod arr;
pub mod fd;
pub mod lanes;
pub mod mc;
mod real;
pub mod surface;
pub mod trace;

pub use adnum::AdNum;
pub use arr::Arr;
pub use ctad_codegen as codegen;
pub use lanes::Lanes;
pub use real::{Dual, Real};
pub use ctad_macros::{fd_model, kernel, mc_model};
pub use fd::{FdModel, FdResult, Grid, fd_greeks, fd_price};
pub use surface::{FdBook, FdSurface, SurfaceGrid, SurfacePoint, fd_surface, fd_surface_book};
pub use mc::{McModel, McResult, mc_greeks};
