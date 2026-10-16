//! Black-Scholes European call and American puts for the FD engine.

use raads::{AdNum, fd_model};

/// params: s0, r, sigma, T, K.
pub struct BsEuroCall;

#[fd_model]
impl BsEuroCall {
    pub fn coeffs<N: AdNum>(#[data] _x: N, p: [N; 5]) -> [N; 3] {
        let r = p[1];
        let sigma = p[2];
        [0.5 * sigma * sigma, r - 0.5 * sigma * sigma, r]
    }

    pub fn payoff<N: AdNum>(#[data] x: N, p: [N; 5]) -> N {
        (x.exp() - p[4]).relu()
    }

    pub fn spot<N: AdNum>(p: [N; 5]) -> N {
        p[0].ln()
    }

    pub fn maturity<N: AdNum>(p: [N; 5]) -> N {
        p[3]
    }
}

/// American put. params: s0, r, sigma, T, K.
pub struct BsAmericanPut;

#[fd_model]
impl BsAmericanPut {
    const AMERICAN: bool = true;

    pub fn coeffs<N: AdNum>(#[data] _x: N, p: [N; 5]) -> [N; 3] {
        let r = p[1];
        let sigma = p[2];
        [0.5 * sigma * sigma, r - 0.5 * sigma * sigma, r]
    }

    pub fn payoff<N: AdNum>(#[data] x: N, p: [N; 5]) -> N {
        (p[4] - x.exp()).relu()
    }

    pub fn spot<N: AdNum>(p: [N; 5]) -> N {
        p[0].ln()
    }

    pub fn maturity<N: AdNum>(p: [N; 5]) -> N {
        p[3]
    }
}

/// American put with a softplus-smoothed exercise decision (width 0.002), so second-order
/// Greeks w.r.t. boundary-moving parameters are well defined.
pub struct BsAmericanPutSmooth;

#[fd_model]
impl BsAmericanPutSmooth {
    const AMERICAN: bool = true;
    const EXERCISE_EPS: f64 = 0.002;

    pub fn coeffs<N: AdNum>(#[data] _x: N, p: [N; 5]) -> [N; 3] {
        let r = p[1];
        let sigma = p[2];
        [0.5 * sigma * sigma, r - 0.5 * sigma * sigma, r]
    }

    pub fn payoff<N: AdNum>(#[data] x: N, p: [N; 5]) -> N {
        (p[4] - x.exp()).relu()
    }

    pub fn spot<N: AdNum>(p: [N; 5]) -> N {
        p[0].ln()
    }

    pub fn maturity<N: AdNum>(p: [N; 5]) -> N {
        p[3]
    }
}

/// Unit-strike put on x = log(S/K) for the surface engine: V(S, K, T) = K u(log(S/K), T), so
/// one solve prices every strike. params: r, q (continuous dividend yield), sigma.
pub struct BsPutMoneyness<const AMERICAN: bool>;

#[fd_model]
impl<const A: bool> BsPutMoneyness<A> {
    const AMERICAN: bool = A;

    pub fn coeffs<N: AdNum>(#[data] _x: N, p: [N; 3]) -> [N; 3] {
        let r = p[0];
        let q = p[1];
        let sigma = p[2];
        [0.5 * sigma * sigma, r - q - 0.5 * sigma * sigma, r]
    }

    pub fn payoff<N: AdNum>(#[data] x: N, _p: [N; 3]) -> N {
        x.exp().rsub(1.0).relu()
    }

    /// Unused by the surface engine.
    pub fn spot<N: AdNum>(_p: [N; 3]) -> N {
        N::from(0.0)
    }

    /// Unused: maturities are the surface's time grid.
    pub fn maturity<N: AdNum>(_p: [N; 3]) -> N {
        N::from(0.0)
    }
}
