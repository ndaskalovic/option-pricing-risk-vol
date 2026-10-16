# raads

Compile-time adjoint algorithmic differentiation for option pricing, in Rust.

You write a model once as a few small functions. A macro differentiates them at compile time into straight-line SIMD code, and the Monte Carlo and finite-difference engines return the price and every Greek in one run.

## Monte Carlo: a European call

```rust
use raads::prelude::*;
use rand::{SeedableRng, rngs::StdRng};
use rand_distr::{Distribution, StandardNormal};

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

// n_paths * n_steps standard normal draws (f32); n_paths a multiple of the lanes (32)
let (n_steps, n_paths) = (52, 100_000);
let normals: Vec<f32> = StandardNormal.sample_iter(StdRng::seed_from_u64(7)).take(n_paths * n_steps).collect();

let params = [100.0, 0.01, 0.2, 1.0, 100.0].map(F32x32::from);
let r = mc_greeks::<GbmCall, F32x32>(&params, n_steps, n_paths, &normals);
println!("{} ± {} delta {} vega {}", r.price, r.std_err, r.grad[0], r.grad[2]);
```

## Finite differences: one American put

```rust
use raads::prelude::*;

pub struct AmericanPut; // params: s0, r, sigma, T, K

#[fd_model]
impl AmericanPut {
    const AMERICAN: bool = true;

    // PDE in x = log S:  V_t = a V_xx + b V_x - c V
    pub fn coeffs<N: AdNum>(#[data] _x: N, p: [N; 5]) -> [N; 3] {
        [0.5 * p[2] * p[2], p[1] - 0.5 * p[2] * p[2], p[1]]
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

let grid = FdGrid::new(100f64.ln() - 1.5, 100f64.ln() + 1.5, 401, 400);
let r = fd_greeks::<AmericanPut, f64>(&[100.0, 0.05, 0.2, 1.0, 100.0], &grid);
println!("{} delta {} vega {}", r.price, r.grad[0], r.grad[2]); // 6.0883 -0.4110 37.4945
```

## Any function

`#[kernel]` keeps the function and adds a module with compiled forward and reverse code. Parameters without `#[data]` get gradients.

```rust
use raads::prelude::*;

/// f(x, y; z) = exp(x y) + sqrt(x^2 + z) / y
#[kernel]
pub fn f<N: AdNum>(v: [N; 2], #[data] z: N) -> N {
    let (x, y) = (v[0], v[1]);
    (x * y).exp() + (x * x + z).sqrt() / y
}

let value = f([0.5, 2.0], 1.0); // plain call

let mut grad = [0.0; 2]; // value and [df/dx, df/dy] in one reverse pass
let value = f::vjp(&[0.5, 2.0], 1.0, 1.0, &mut grad);

let mut h = [Dual::from(0.0); 2]; // second order: Hessian column for x
f::vjp(&[Dual::new(0.5, 1.0), Dual::from(2.0)], Dual::from(1.0), Dual::from(1.0), &mut h);
println!("{value} {grad:?} d2f/dx2 {} d2f/dydx {}", h[0].t, h[1].t);
```

On x86_64, build with `RUSTFLAGS="-C target-cpu=x86-64-v3"` (AVX2 + FMA) or `native`; default builds use the portable fallback (no SIMD `exp`, no FMA), which is much slower.

## License

MIT or Apache-2.0, at your option.
