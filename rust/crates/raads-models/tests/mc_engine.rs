//! MC engine, f64 (port of `tests/test_mc_engine.mojo`, same tolerances):
//! 1. dot-product test: price tangent with Dual params (forward mode through the whole engine)
//!    == grad · v from the reverse sweep
//! 2. gradient == central differences of the price with common random numbers (SVI's T is
//!    detached inside the vol by design, so it is excluded)
//! 3. GBM call vs Black-Scholes within about 4 standard errors
//!
//! Also: f32 lanes == f64 lanes on the same normals (no accumulation bias), a two-factor model
//! (NZ = 2, saved residuals) vs a scalar reference, and the argument checks.

use raads::lanes::{F32x8, F64x4, F64x8};
use raads::{AdNum, Dual, McModel, mc_greeks, mc_model};
use raads_models::gbm::{GbmAsianCall, GbmCall};
use raads_models::svi::SviCall;

mod common;
use common::{Checks, normals};

type V = F64x8;
type D = Dual<V>;
const W: usize = 8;
const N_STEPS: usize = 52;
const N_PATHS: usize = 200_000 / W;

fn vec_params<M: McModel>(p: &[f64]) -> M::P<V> {
    let mut r: M::P<V> = raads::Arr::splat(V::from(0.0));
    for (x, &v) in r.as_mut().iter_mut().zip(p) {
        *x = V::from(v);
    }
    r
}

/// `detached`: a parameter the model passes through `detach` (stop-gradient); its AD
/// derivative intentionally differs from a finite difference.
fn verify<M: McModel>(c: &mut Checks, label: &str, p: &[f64], n_paths: usize, z: &[f32], detached: Option<usize>) {
    println!("{label}");
    let r = mc_greeks::<M, V>(&vec_params::<M>(p), N_STEPS, n_paths, z);
    println!("    price {} grad {:?}", r.price, r.grad.as_ref());
    // 1. dot test
    let mut pd: M::P<D> = raads::Arr::splat(D::from(0.0));
    let mut gv = 0.0;
    for (j, x) in pd.as_mut().iter_mut().enumerate() {
        let v = 0.3 + 0.1 * j as f64;
        *x = D::new(V::from(p[j]), V::from(v));
        gv += r.grad.as_ref()[j] * v;
    }
    let d = mc_greeks::<M, D>(&pd, N_STEPS, n_paths, z);
    c.le(&format!("{label}: dot test (Dual price tangent vs grad . v)"), (d.price_tangent - gv).abs() / gv.abs(), 1e-12);
    // 2. central differences (same normals)
    let mut e = 0.0f64;
    for j in 0..p.len() {
        let h = 1e-5 * p[j].abs().max(1e-2);
        let (mut pp, mut pm) = (p.to_vec(), p.to_vec());
        pp[j] += h;
        pm[j] -= h;
        let fd = (mc_greeks::<M, V>(&vec_params::<M>(&pp), N_STEPS, n_paths, z).price
            - mc_greeks::<M, V>(&vec_params::<M>(&pm), N_STEPS, n_paths, z).price)
            / (2.0 * h);
        if Some(j) != detached {
            e = e.max((r.grad.as_ref()[j] - fd).abs() / fd.abs().max(1e-2));
        }
    }
    c.le(&format!("{label}: gradient vs central differences"), e, 1e-4);
}

#[test]
fn mc_engine() {
    let z = normals(N_PATHS * N_STEPS * W, 42);
    let mut c = Checks::default();
    let p = [100.0, 0.01, 0.2, 1.0, 100.0];
    verify::<GbmCall>(&mut c, "GBM call (s0, r, sigma, T, K)", &p, N_PATHS, &z, None);
    verify::<GbmAsianCall>(&mut c, "GBM Asian call", &p, N_PATHS / 4, &z, None);
    let sp = [227.7899932861328, 0.04, 0.7287671233, 220.0, 5.73431376e-02, 6.59987208e-04, -1.23157254, 2.22873847e+02, 9.99943135e-02];
    verify::<SviCall>(&mut c, "SVI call (s0, r, T, K, a, b, rho, m, sigma)", &sp, N_PATHS, &z, Some(2));

    // GBM call vs Black-Scholes, within 4 standard errors
    let call = mc_greeks::<GbmCall, V>(&vec_params::<GbmCall>(&p), N_STEPS, N_PATHS, &z);
    let bs = [8.433318690109608, 0.5596176923702425, 47.52845054691464, 39.447933090788894, 4.420077814548036, -0.4752845054691464];
    let tol = [0.12, 0.0053, 0.44, 0.66, 0.2, 0.0044];
    let ours = [call.price, call.grad[0], call.grad[1], call.grad[2], call.grad[3], call.grad[4]];
    println!("GBM call vs Black-Scholes (4 SE)");
    for i in 0..6 {
        c.abs(&format!("price / Greek {i}"), ours[i], bs[i], tol[i]);
    }
    c.assert_ok();
}

/// f32 lanes over 400k paths: the engine's sums over paths must not drift (they used to: rho
/// biased by -1.5e-4 relative here, -2.5e-3 at 1.6M paths). Reference: the same f32 path
/// arithmetic in batches of 8 path blocks, averaged in f64. Against f64 lanes the difference is
/// the model's own f32 rounding (the constant drift added to log S every step: ~5e-5).
#[test]
fn mc_f32_accumulation() {
    let n_paths = 400_000 / W;
    let z = normals(n_paths * N_STEPS * W, 1);
    let p = [100.0, 0.01, 0.2, 1.0, 100.0];
    let p32 = p.map(F32x8::from);
    let r32 = mc_greeks::<GbmCall, F32x8>(&p32, N_STEPS, n_paths, &z);
    let batch = 8;
    let (mut price, mut grad) = (0.0, [0.0; 5]);
    for zb in z.chunks(batch * N_STEPS * W) {
        let r = mc_greeks::<GbmCall, F32x8>(&p32, N_STEPS, batch, zb);
        price += r.price;
        for (g, rg) in grad.iter_mut().zip(r.grad) {
            *g += rg;
        }
    }
    let nb = (n_paths / batch) as f64;
    let r64 = mc_greeks::<GbmCall, V>(&p.map(V::from), N_STEPS, n_paths, &z);
    let mut c = Checks::default();
    c.rel("price vs f32 batches", r32.price, price / nb, 1e-6);
    c.rel("price vs f64", r32.price, r64.price, 1e-4);
    for (j, g) in grad.iter().enumerate() {
        c.rel(&format!("grad {j} vs f32 batches"), r32.grad[j], g / nb, 1e-6);
        c.rel(&format!("grad {j} vs f64"), r32.grad[j], r64.grad[j], 1e-4);
    }
    c.assert_ok();
}

/// Two normals per step and a step that saves residuals (`exp` of a state-dependent argument).
pub struct TwoFactor;

#[mc_model]
impl TwoFactor {
    pub fn setup<N: AdNum>(p: [N; 4], #[data] n_steps: N) -> [N; 3] {
        let dt = p[3] / n_steps;
        [p[1] * dt.sqrt(), p[2] * dt.sqrt(), dt]
    }
    pub fn init<N: AdNum>(p: [N; 4]) -> [N; 2] {
        [p[0], N::from(0.0)]
    }
    pub fn step<N: AdNum>(s: [N; 2], c: [N; 3], #[data] z: [N; 2]) -> [N; 2] {
        let y = s[1] * 0.9 + c[1] * z[1];
        [s[0] * (c[0] * z[0] + y * c[2]).exp(), y]
    }
    pub fn payoff<N: AdNum>(s: [N; 2], p: [N; 4], _c: [N; 3]) -> N {
        (s[0] - 1.0).relu() + s[1] * s[1] * p[0]
    }
}

/// `TwoFactor` path by path in plain f64, on the engine's normals layout.
fn two_factor_scalar(p: [f64; 4], n_steps: usize, z: &[f32], w: usize, n_paths: usize) -> f64 {
    let dt = p[3] / n_steps as f64;
    let (c0, c1) = (p[1] * dt.sqrt(), p[2] * dt.sqrt());
    let mut sum = 0.0;
    for path in z.chunks_exact(n_steps * 2 * w).take(n_paths) {
        for lane in 0..w {
            let (mut x, mut y) = (p[0], 0.0);
            for zk in path.chunks_exact(2 * w) {
                y = y * 0.9 + c1 * zk[w + lane] as f64;
                x *= (c0 * zk[lane] as f64 + y * dt).exp();
            }
            sum += (x - 1.0).max(0.0) + y * y * p[0];
        }
    }
    sum / (n_paths * w) as f64
}

#[test]
fn mc_two_factor() {
    type V4 = F64x4;
    let (n_steps, n_paths, w) = (12, 500, 4);
    let z = normals(n_paths * n_steps * 2 * w, 5);
    let p = [1.0, 0.3, 0.5, 2.0];
    const { assert!(<TwoFactor as McModel>::STEP_NRES > 0) };
    let r = mc_greeks::<TwoFactor, V4>(&p.map(V4::from), n_steps, n_paths, &z);
    let mut c = Checks::default();
    c.rel("price vs scalar", r.price, two_factor_scalar(p, n_steps, &z, w, n_paths), 1e-12);
    for j in 0..4 {
        let h = 1e-6 * p[j];
        let (mut a, mut b) = (p, p);
        a[j] += h;
        b[j] -= h;
        let fd = (two_factor_scalar(a, n_steps, &z, w, n_paths) - two_factor_scalar(b, n_steps, &z, w, n_paths)) / (2.0 * h);
        c.rel(&format!("grad {j} vs scalar central differences"), r.grad[j], fd, 1e-6);
    }
    let v = |i: usize| 0.3 + 0.1 * i as f64;
    let pd: [Dual<V4>; 4] = std::array::from_fn(|i| Dual::new(V4::from(p[i]), V4::from(v(i))));
    let d = mc_greeks::<TwoFactor, Dual<V4>>(&pd, n_steps, n_paths, &z);
    c.rel("dot test", d.price_tangent, (0..4).map(|i| r.grad[i] * v(i)).sum(), 1e-12);
    c.assert_ok();
}

#[test]
#[should_panic(expected = "need n_steps, n_paths and NZ >= 1")]
fn mc_zero_steps() {
    mc_greeks::<GbmCall, V>(&[100.0, 0.01, 0.2, 1.0, 100.0].map(V::from), 0, 1, &[]);
}

#[test]
#[should_panic(expected = "need n_steps, n_paths and NZ >= 1")]
fn mc_zero_paths() {
    mc_greeks::<GbmCall, V>(&[100.0, 0.01, 0.2, 1.0, 100.0].map(V::from), 52, 0, &[]);
}

#[test]
#[should_panic(expected = "need n_paths * n_steps * NZ * W normals")]
fn mc_short_normals() {
    mc_greeks::<GbmCall, V>(&[100.0, 0.01, 0.2, 1.0, 100.0].map(V::from), 52, 2, &[0.0; 52 * W]);
}
