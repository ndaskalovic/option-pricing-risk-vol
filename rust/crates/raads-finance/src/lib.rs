//! Option pricing on [`raads_engines`]: Black-Scholes and SVI models for the path and 1-D PDE
//! engines (price = the engine's value, Greeks = its gradient, second-order Greeks with
//! [`Dual`](raads::Dual) parameters) and option surfaces from one multi-maturity PDE solve.

pub mod bs_fd;
pub mod gbm;
pub mod surface;
pub mod svi;

pub use surface::{SurfaceOption, option};
