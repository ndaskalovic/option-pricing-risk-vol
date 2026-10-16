# raads

Compile-time adjoint algorithmic differentiation for option pricing, in Rust.

You write a model once as a few small functions. A macro differentiates them at compile time into straight-line SIMD code, and the Monte Carlo and finite-difference engines return the price and every Greek in one run.

## Example

```rust
use raads::prelude::*;

pub struct GbmCall; // params: s0, r, sigma, T, K

#[mc_model]
impl GbmCall {
    pub fn setup<N: AdNum>(p: [N; 5], #[data] n_steps: N) -> [N; 2] {
        let dt = p[3] / n_steps;
        [(p[1] - 0.5 * p[2] * p[2]) * dt, p[2] * dt.sqrt()]
    }
    pub fn init<N: AdNum>(p: [N; 5]) -> [N; 1] {
        [p[0].ln()]
    }
    pub fn step<N: AdNum>(s: [N; 1], c: [N; 2], #[data] z: [N; 1]) -> [N; 1] {
        [s[0] + c[0] + c[1] * z[0]]
    }
    pub fn payoff<N: AdNum>(s: [N; 1], p: [N; 5], _c: [N; 2]) -> N {
        (s[0].exp() - p[4]).relu() * (-p[1] * p[3]).exp()
    }
}

let params = [100.0, 0.01, 0.2, 1.0, 100.0].map(F32x32::from);
let r = mc_greeks::<GbmCall, F32x32>(&params, n_steps, n_paths, &normals);
println!("{} {} {}", r.price, r.grad[0], r.grad[2]); // price, delta, vega
```

## License

MIT or Apache-2.0, at your option.
