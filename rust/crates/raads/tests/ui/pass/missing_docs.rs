//! Generated items are documented.
#![deny(missing_docs)]

use raads::{AdNum, kernel, mc_model};

/// A kernel.
#[kernel]
pub fn payoff<N: AdNum>(s: [N; 1], #[data] k: N) -> N {
    (s[0].exp() - k).relu()
}

/// A model.
pub struct Model;

#[mc_model]
impl Model {
    /// Coefficients.
    pub fn setup<N: AdNum>(p: [N; 5], #[data] n_steps: N) -> [N; 2] {
        let dt = p[3] / n_steps;
        [p[1] * dt, p[2] * dt.sqrt()]
    }

    /// Initial state.
    pub fn init<N: AdNum>(p: [N; 5]) -> [N; 1] {
        [p[0].ln()]
    }

    /// One step.
    pub fn step<N: AdNum>(s: [N; 1], c: [N; 2], #[data] z: [N; 1]) -> [N; 1] {
        [s[0] + c[0] + c[1] * z[0]]
    }

    /// Payoff.
    pub fn payoff<N: AdNum>(s: [N; 1], p: [N; 5], _c: [N; 2]) -> N {
        (s[0].exp() - p[4]).relu()
    }
}

fn main() {}
