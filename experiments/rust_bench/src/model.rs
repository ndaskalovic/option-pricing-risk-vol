//! GBM European call, log-space exact scheme -- written once, generic over AdNum.
//! params p: s0, r, sigma, T, K.  coeffs c: drift, vol*sqrt(dt).  state s: log S.
use crate::adnum::AdNum;

pub fn setup<N: AdNum>(p: &[N; 5], n_steps: N) -> [N; 2] {
    let (r, sigma, t) = (p[1], p[2], p[3]);
    let dt = t / n_steps;
    [(r - sigma * sigma * 0.5) * dt, sigma * dt.sqrt()]
}

pub fn init<N: AdNum>(p: &[N; 5]) -> [N; 1] {
    [p[0].ln()]
}

pub fn step<N: AdNum>(s: &[N; 1], c: &[N; 2], z: &[N; 1]) -> [N; 1] {
    [s[0] + c[0] + c[1] * z[0]]
}

pub fn payoff<N: AdNum>(s: &[N; 1], p: &[N; 5], _c: &[N; 2]) -> N {
    let (r, t, k) = (p[1], p[3], p[4]);
    (s[0].exp() - k).relu() * (-(r * t)).exp()
}
