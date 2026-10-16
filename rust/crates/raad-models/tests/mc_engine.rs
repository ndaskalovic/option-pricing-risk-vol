//! MC engine, f64 (port of `tests/test_mc_engine.mojo`, same tolerances):
//! 1. dot-product test: price tangent with Dual params (forward mode through the whole engine)
//!    == grad · v from the reverse sweep
//! 2. gradient == central differences of the price with common random numbers (SVI's T is
//!    detached inside the vol by design, so it is excluded)
//! 3. GBM call vs Black-Scholes within about 4 standard errors

use raad::lanes::F64x8;
use raad::{Dual, McModel, mc_greeks};
use raad_models::gbm::{GbmAsianCall, GbmCall};
use raad_models::svi::SviCall;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, StandardNormal};

type V = F64x8;
type D = Dual<V>;
const W: usize = 8;
const N_STEPS: usize = 52;
const N_PATHS: usize = 200_000 / W;

fn normals(n: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n).map(|_| StandardNormal.sample(&mut rng)).collect()
}

fn vec_params<M: McModel>(p: &[f64]) -> M::P<V> {
    let mut r: M::P<V> = raad::Arr::splat(V::from(0.0));
    for (x, &v) in r.as_mut().iter_mut().zip(p) {
        *x = V::from(v);
    }
    r
}

fn check(fails: &mut Vec<String>, label: &str, err: f64, tol: f64) {
    println!("    {label}: max rel err {err:e} (tol {tol:e})");
    if err > tol {
        fails.push(format!("{label}: {err:e} > {tol:e}"));
    }
}

/// `detached`: a parameter the model passes through `detach` (stop-gradient); its AD
/// derivative intentionally differs from a finite difference.
fn verify<M: McModel>(fails: &mut Vec<String>, label: &str, p: &[f64], n_paths: usize, z: &[f32], detached: Option<usize>) {
    println!("{label}");
    let r = mc_greeks::<M, V>(&vec_params::<M>(p), N_STEPS, n_paths, z);
    println!("    price {} grad {:?}", r.price, r.grad.as_ref());
    // 1. dot test
    let mut pd: M::P<D> = raad::Arr::splat(D::from(0.0));
    let mut gv = 0.0;
    for (j, x) in pd.as_mut().iter_mut().enumerate() {
        let v = 0.3 + 0.1 * j as f64;
        *x = D::new(V::from(p[j]), V::from(v));
        gv += r.grad.as_ref()[j] * v;
    }
    let d = mc_greeks::<M, D>(&pd, N_STEPS, n_paths, z);
    check(fails, &format!("{label}: dot test (Dual price tangent vs grad . v)"), (d.price_tangent - gv).abs() / gv.abs(), 1e-12);
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
    check(fails, &format!("{label}: gradient vs central differences"), e, 1e-4);
}

#[test]
fn mc_engine() {
    let z = normals(N_PATHS * N_STEPS * W, 42);
    let mut fails = Vec::new();
    let p = [100.0, 0.01, 0.2, 1.0, 100.0];
    verify::<GbmCall>(&mut fails, "GBM call (s0, r, sigma, T, K)", &p, N_PATHS, &z, None);
    verify::<GbmAsianCall>(&mut fails, "GBM Asian call", &p, N_PATHS / 4, &z, None);
    let sp = [227.7899932861328, 0.04, 0.7287671233, 220.0, 5.73431376e-02, 6.59987208e-04, -1.23157254, 2.22873847e+02, 9.99943135e-02];
    verify::<SviCall>(&mut fails, "SVI call (s0, r, T, K, a, b, rho, m, sigma)", &sp, N_PATHS, &z, Some(2));

    // GBM call vs Black-Scholes, within 4 standard errors
    let c = mc_greeks::<GbmCall, V>(&vec_params::<GbmCall>(&p), N_STEPS, N_PATHS, &z);
    let bs = [8.433318690109608, 0.5596176923702425, 47.52845054691464, 39.447933090788894, 4.420077814548036, -0.4752845054691464];
    let tol = [0.12, 0.0053, 0.44, 0.66, 0.2, 0.0044];
    let ours = [c.price, c.grad[0], c.grad[1], c.grad[2], c.grad[3], c.grad[4]];
    for i in 0..6 {
        if (ours[i] - bs[i]).abs() > tol[i] {
            fails.push(format!("Black-Scholes check {i}: {} vs {}", ours[i], bs[i]));
        }
    }
    println!("GBM call vs Black-Scholes (4 SE): {:?} vs {:?}", ours, bs);
    assert!(fails.is_empty(), "{fails:#?}");
}
