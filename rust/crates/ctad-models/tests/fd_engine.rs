//! FD engine, f64 (port of `tests/test_fd_engine.mojo`, same tolerances): forward == reverse,
//! gradient == finite differences, European call vs Black-Scholes, American put vs a CRR
//! binomial tree, lanes == single solves.

use ctad::lanes::F64x4;
use ctad::{Dual, FdModel, Grid, Lanes, fd_greeks, fd_price};
use ctad_models::bs_fd::{BsAmericanPut, BsEuroCall};

type D = Dual<f64>;

fn crr_american_put(s: f64, k: f64, r: f64, sigma: f64, t: f64, n: usize) -> f64 {
    let dt = t / n as f64;
    let up = (sigma * dt.sqrt()).exp();
    let dn = 1.0 / up;
    let q = ((r * dt).exp() - dn) / (up - dn);
    let disc = (-r * dt).exp();
    let node = |i: usize, step: usize| s * up.powi(i as i32) * dn.powi((step - i) as i32);
    let mut v: Vec<f64> = (0..=n).map(|i| (k - node(i, n)).max(0.0)).collect();
    for step in (0..n).rev() {
        for i in 0..=step {
            v[i] = (disc * (q * v[i + 1] + (1.0 - q) * v[i])).max(k - node(i, step));
        }
    }
    v[0]
}

fn check(fails: &mut Vec<String>, label: &str, err: f64, tol: f64) {
    println!("    {label}: max err {err:e} (tol {tol:e})");
    if err > tol {
        fails.push(format!("{label}: {err:e} > {tol:e}"));
    }
}

fn params<M: FdModel>(p: &[f64]) -> M::P<f64> {
    let mut r: M::P<f64> = ctad::Arr::splat(0.0);
    r.as_mut().copy_from_slice(p);
    r
}

fn verify<M: FdModel>(fails: &mut Vec<String>, label: &str, p: &[f64], g: &Grid, fd_tol: f64) {
    println!("{label} {}x{}", g.nx, g.nt);
    let r = fd_greeks::<M, f64>(&params::<M>(p), g);
    // dot test: forward tangent of the price through the scheme vs the reverse sweep
    let mut pd: M::P<D> = ctad::Arr::splat(D::from(0.0));
    let mut gv = 0.0;
    for (j, x) in pd.as_mut().iter_mut().enumerate() {
        let v = 0.3 + 0.1 * j as f64;
        *x = D::new(p[j], v);
        gv += r.grad.as_ref()[j] * v;
    }
    let d = fd_greeks::<M, D>(&pd, g);
    check(fails, &format!("{label}: dot test (rel)"), (d.price.t - gv).abs() / gv.abs(), 1e-12);
    // gradient vs central differences of the price-only solve
    let mut e = 0.0f64;
    for j in 0..p.len() {
        let h = 1e-6 * p[j].abs().max(1e-2);
        let (mut pp, mut pm) = (p.to_vec(), p.to_vec());
        pp[j] += h;
        pm[j] -= h;
        let fd = (fd_price::<M, f64>(&params::<M>(&pp), g) - fd_price::<M, f64>(&params::<M>(&pm), g)) / (2.0 * h);
        e = e.max((r.grad.as_ref()[j] - fd).abs() / fd.abs().max(1e-3));
    }
    check(fails, &format!("{label}: gradient vs central differences (rel)"), e, fd_tol);
}

#[test]
fn fd_engine() {
    let mut fails = Vec::new();
    let p = [100.0, 0.01, 0.2, 1.0, 100.0];
    let (lo, hi) = (100f64.ln() - 6.0 * 0.2, 100f64.ln() + 6.0 * 0.2);
    let g201 = Grid::new(lo, hi, 201, 200);
    verify::<BsEuroCall>(&mut fails, "European call", &p, &g201, 1e-6);
    verify::<BsAmericanPut>(&mut fails, "American put", &p, &g201, 1e-6);

    let g801 = Grid::new(lo, hi, 801, 800);
    let e = fd_greeks::<BsEuroCall, f64>(&p, &g801);
    let bs = [8.433318690109608, 0.5596176923702425, 47.52845054691464, 39.447933090788894, 4.420077814548036, -0.4752845054691464];
    let mut worst = (e.price - bs[0]).abs() / bs[0];
    for j in 0..5 {
        worst = worst.max((e.grad[j] - bs[j + 1]).abs() / bs[j + 1].abs());
    }
    println!("European call 801x800 vs Black-Scholes");
    check(&mut fails, "price and 5 Greeks (rel)", worst, 2e-5);
    let a = fd_price::<BsAmericanPut, f64>(&p, &g801);
    let crr = crr_american_put(100.0, 100.0, 0.01, 0.2, 1.0, 4000);
    println!("American put 801x800 vs CRR(4000): {a} {crr}");
    check(&mut fails, "price (rel)", (a - crr).abs() / crr, 5e-5);

    // 4 strikes in 4 lanes == 4 single solves; fd_price == fd_greeks price
    let strikes = [90.0, 100.0, 110.0, 120.0];
    let mut pv = p.map(F64x4::from);
    pv[4] = F64x4::from_fn(|l| strikes[l]);
    let bt = fd_greeks::<BsAmericanPut, F64x4>(&pv, &g201);
    let po = fd_price::<BsAmericanPut, F64x4>(&pv, &g201);
    let mut lane_err = 0.0f64;
    for (k, &kk) in strikes.iter().enumerate() {
        let mut q = p;
        q[4] = kk;
        let sgl = fd_greeks::<BsAmericanPut, f64>(&q, &g201);
        lane_err = lane_err.max((bt.price.lane(k) - sgl.price).abs() / sgl.price).max((po.lane(k) - sgl.price).abs() / sgl.price);
        for j in 0..5 {
            lane_err = lane_err.max((bt.grad[j].lane(k) - sgl.grad[j]).abs() / (sgl.grad[j].abs() + 1e-12));
        }
    }
    println!("4 strikes in lanes vs single solves");
    check(&mut fails, "price and gradient (rel)", lane_err, 1e-12);
    assert!(fails.is_empty(), "{fails:#?}");
}
