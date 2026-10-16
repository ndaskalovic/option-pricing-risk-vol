//! FD surface engine (port of `tests/test_fd_surface.mojo`, same tolerances):
//! 1. European surface (price, delta, gamma, theta, dV/dK; Dual vega, rho) vs Black-Scholes
//! 2. Dual tangents (vega, rho) == central bumps of the surface (European exact, American)
//! 3. American book: reverse-sweep gradient == sum of position-weighted forward tangents
//! 4. American surface prices vs the single-option engine (fd_price, BsAmericanPut)
//!
//! Also: book gradient == forward tangents for W = 2 lanes with different points, every
//! Rannacher count and one or several maturities; the argument checks.

use raad::lanes::{F64x2, F64x4};
use raad::{Dual, FdSurface, Grid, Lanes, SurfaceGrid, SurfacePoint, fd_price, fd_surface, fd_surface_book};
use raad_models::bs_fd::{BsAmericanPut, BsPutMoneyness};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

mod common;
use common::{Checks, rel};

const S0: f64 = 100.0;
const R_: f64 = 0.03;
const Q_: f64 = 0.02;
const SIG: f64 = 0.25;
const T: [f64; 6] = [1.0 / 52.0, 1.0 / 12.0, 0.25, 0.5, 1.0, 2.0];

type D2 = Dual<F64x2>;
type D4 = Dual<F64x4>;

fn strikes() -> Vec<f64> {
    (0..13).map(|i| 70.0 + 5.0 * i as f64).collect()
}

fn grid(h: f64, dt: f64) -> SurfaceGrid {
    let z = 5.0 * SIG * 2f64.sqrt();
    let (lo, hi) = ((S0 / 130.0).ln() - z, (S0 / 70.0).ln() + z);
    SurfaceGrid::new(lo, hi, ((hi - lo) / h) as usize + 1, dt)
}

fn ncdf(x: f64) -> f64 {
    0.5 * (1.0 + libm::erf(x / 2f64.sqrt()))
}

fn npdf(x: f64) -> f64 {
    (-0.5 * x * x).exp() / (2.0 * std::f64::consts::PI).sqrt()
}

/// Price, delta, gamma, theta (calendar), dV/dK, vega, rho.
fn bs_put(k: f64, t: f64) -> [f64; 7] {
    let st = SIG * t.sqrt();
    let d1 = ((S0 / k).ln() + (R_ - Q_ + 0.5 * SIG * SIG) * t) / st;
    let d2 = d1 - st;
    let (dq, dr) = ((-Q_ * t).exp(), (-R_ * t).exp());
    [
        k * dr * ncdf(-d2) - S0 * dq * ncdf(-d1),
        -dq * ncdf(-d1),
        dq * npdf(d1) / (S0 * st),
        -S0 * dq * npdf(d1) * SIG / (2.0 * t.sqrt()) + R_ * k * dr * ncdf(-d2) - Q_ * S0 * dq * ncdf(-d1),
        dr * ncdf(-d2),
        S0 * dq * npdf(d1) * t.sqrt(),
        -k * t * dr * ncdf(-d2),
    ]
}

/// Lane 0 tangent = d/dsigma, lane 1 = d/dr.
fn dual_params() -> [D2; 3] {
    [D2::new(F64x2::from(R_), F64x2::from_fn(|l| l as f64)), D2::from(Q_), D2::new(F64x2::from(SIG), F64x2::from_fn(|l| 1.0 - l as f64))]
}

fn tangent_vs_bump<const AM: bool>(c: &mut Checks, g: &SurfaceGrid, rel: f64) {
    let sa = fd_surface::<BsPutMoneyness<AM>, D2>(&dual_params(), g, &T);
    let eps = 1e-6;
    let bumped: Vec<FdSurface<f64>> = [(0.0, eps), (0.0, -eps), (eps, 0.0), (-eps, 0.0)]
        .iter()
        .map(|&(dr, ds)| fd_surface::<BsPutMoneyness<AM>, f64>(&[R_ + dr, Q_, SIG + ds], g, &T))
        .collect();
    let (mut ev, mut er, mut vscale, mut rscale) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for m in 0..T.len() {
        for k in strikes() {
            let o = sa.option(m, F64x2::from(S0), F64x2::from(k));
            let pr: Vec<f64> = bumped.iter().map(|b| b.option(m, S0, k)[0]).collect();
            let (vb, rb) = ((pr[0] - pr[1]) / (2.0 * eps), (pr[2] - pr[3]) / (2.0 * eps));
            ev = ev.max((o[0].t.lane(0) - vb).abs());
            er = er.max((o[0].t.lane(1) - rb).abs());
            vscale = vscale.max(vb.abs());
            rscale = rscale.max(rb.abs());
        }
    }
    let lab = if AM { "American" } else { "European" };
    c.le(&format!("{lab} vega Dual vs bump (rel to max |vega|)"), ev / vscale, rel);
    c.le(&format!("{lab} rho Dual vs bump (rel to max |rho|)"), er / rscale, rel);
}

#[test]
fn fd_surface_engine() {
    let mut c = Checks::default();
    let ks = strikes();

    // 1. European vs closed form
    let g1 = grid(0.0025, 1.0 / 1000.0);
    println!("1. European surface vs Black-Scholes (h 0.0025, dt 1/1000, nx {})", g1.nx);
    let se = fd_surface::<BsPutMoneyness<false>, D2>(&dual_params(), &g1, &T);
    let mut emax = [0.0f64; 7];
    for (m, &t) in T.iter().enumerate() {
        for &k in &ks {
            let o = se.option(m, F64x2::from(S0), F64x2::from(k));
            let b = bs_put(k, t);
            let got = [o[0].v.lane(0), o[1].v.lane(0), o[2].v.lane(0), o[3].v.lane(0), o[4].v.lane(0), o[0].t.lane(0), o[0].t.lane(1)];
            for i in 0..7 {
                emax[i] = emax[i].max((got[i] - b[i]).abs());
            }
        }
    }
    let names = ["price", "delta", "gamma", "theta", "dV/dK", "vega (Dual)", "rho (Dual)"];
    let tols = [2e-3, 2e-4, 2e-4, 5e-2, 2e-4, 2e-2, 2e-3];
    for i in 0..7 {
        c.le(names[i], emax[i], tols[i]);
    }

    // 2. Dual tangents == central bumps of the same scheme (European: exact; American: the
    //    tangent holds the exercise set fixed, a bump of 1e-6 can flip a node)
    println!("2. Dual vega/rho == central bumps (h 0.005, dt 1/500)");
    let g2 = grid(0.005, 1.0 / 500.0);
    tangent_vs_bump::<false>(&mut c, &g2, 5e-7);
    tangent_vs_bump::<true>(&mut c, &g2, 1e-5);

    // 3. American book: one reverse sweep == position-weighted forward tangents
    println!("3. American book ({} options, random positions): reverse sweep vs forward tangents", T.len() * ks.len());
    let mut rng = StdRng::seed_from_u64(3);
    let pos: Vec<f64> = (0..T.len() * ks.len()).map(|_| rng.random_range(-10.0..10.0)).collect();
    let mut pts = Vec::new();
    for m in 0..T.len() {
        for (ki, &k) in ks.iter().enumerate() {
            pts.push(SurfacePoint { m, x: (S0 / k).ln(), w: pos[m * ks.len() + ki] * k });
        }
    }
    let book = fd_surface_book::<BsPutMoneyness<true>, f64>(&[R_, Q_, SIG], &g2, &T, &pts);
    // forward: lanes 0, 1, 2 carry d/dr, d/dq, d/dsigma
    let unit = |j: usize| F64x4::from_fn(|l| if l == j { 1.0 } else { 0.0 });
    let pf = [D4::new(F64x4::from(R_), unit(0)), D4::new(F64x4::from(Q_), unit(1)), D4::new(F64x4::from(SIG), unit(2))];
    let sf = fd_surface::<BsPutMoneyness<true>, D4>(&pf, &g2, &T);
    let (mut fwd, mut val) = ([0.0f64; 3], 0.0);
    for m in 0..T.len() {
        for (ki, &k) in ks.iter().enumerate() {
            let w = pos[m * ks.len() + ki] * k;
            let u = sf.read(m, F64x4::from((S0 / k).ln()))[0];
            for (j, f) in fwd.iter_mut().enumerate() {
                *f += w * u.t.lane(j);
            }
            val += w * u.v.lane(0);
        }
    }
    c.le("book value vs forward", (book.value - val).abs() / val.abs(), 1e-12);
    for (j, name) in ["dbook/dr", "dbook/dq", "dbook/dsigma"].iter().enumerate() {
        println!("      {name} reverse {} forward {}", book.grad[j], fwd[j]);
        c.le(&format!("{name} rel"), (book.grad[j] - fwd[j]).abs() / fwd[j].abs(), 1e-10);
    }

    // 4. American surface vs the single-option engine (q = 0)
    println!("4. American surface (q=0) vs fd_price[BsAmericanPut] 801x800 per option");
    let s0 = fd_surface::<BsPutMoneyness<true>, f64>(&[R_, 0.0, SIG], &g1, &T);
    let mut e4 = 0.0f64;
    for (m, &t) in T.iter().enumerate() {
        for k in [80.0, 100.0, 120.0] {
            let z = 5.0 * SIG * t.sqrt();
            let g = Grid::new(S0.ln() - 0.5 - z, S0.ln() + 0.5 + z, 801, 800);
            let single = fd_price::<BsAmericanPut, f64>(&[S0, R_, SIG, t, k], &g);
            e4 = e4.max((s0.option(m, S0, k)[0] - single).abs());
        }
    }
    c.le("price vs single-option engine (abs, S=100)", e4, 2e-3);
    c.assert_ok();
}

/// Book gradient (reverse) vs forward Dual tangents: W = 2 lanes with different reads and
/// weights, for a Rannacher count and maturity list. Max relative error.
fn book_vs_tangents<const AM: bool>(rannacher: usize, mats: &[f64]) -> f64 {
    let g = SurfaceGrid { x_lo: -1.5, x_hi: 1.5, nx: 241, dt: 1.0 / 200.0, rannacher };
    let p = [R_, Q_, SIG];
    let mut rng = StdRng::seed_from_u64(11);
    let pts: Vec<SurfacePoint<F64x2>> = (0..5 * mats.len())
        .map(|k| SurfacePoint {
            m: k % mats.len(),
            x: F64x2::from_fn(|_| rng.random_range(-0.4..0.4)),
            w: F64x2::from_fn(|_| rng.random_range(-5.0..5.0)),
        })
        .collect();
    let book = fd_surface_book::<BsPutMoneyness<AM>, F64x2>(&p.map(F64x2::from), &g, mats, &pts);
    let mut e = 0.0f64;
    for j in 0..3 {
        let pd: [D2; 3] = std::array::from_fn(|i| D2::new(F64x2::from(p[i]), F64x2::from(if i == j { 1.0 } else { 0.0 })));
        let s = fd_surface::<BsPutMoneyness<AM>, D2>(&pd, &g, mats);
        let fwd = pts.iter().fold(F64x2::from(0.0), |f, pt| f + pt.w * s.read(pt.m, pt.x)[0].t);
        for l in 0..2 {
            e = e.max(rel(book.grad[j].lane(l), fwd.lane(l)));
        }
    }
    e
}

#[test]
fn book_lanes_rannacher_maturities() {
    let mut c = Checks::default();
    for rannacher in [0, 1, 2] {
        for mats in [&[0.5][..], &[0.1, 0.3, 1.0][..]] {
            let label = format!("rannacher {rannacher}, maturities {mats:?}");
            c.le(&format!("European {label}"), book_vs_tangents::<false>(rannacher, mats), 1e-11);
            c.le(&format!("American {label}"), book_vs_tangents::<true>(rannacher, mats), 1e-11);
        }
    }
    c.assert_ok();
}

fn small_grid(nx: usize) -> SurfaceGrid {
    SurfaceGrid { x_lo: -1.0, x_hi: 1.0, nx, dt: 0.1, rannacher: 0 }
}

#[test]
fn surface_five_nodes() {
    let s = fd_surface::<BsPutMoneyness<false>, f64>(&[R_, Q_, SIG], &small_grid(5), &[0.5]);
    assert!(s.read(0, 0.0)[0].is_finite());
}

#[test]
#[should_panic(expected = "need at least 5 grid nodes")]
fn surface_four_nodes() {
    fd_surface::<BsPutMoneyness<false>, f64>(&[R_, Q_, SIG], &small_grid(4), &[0.5]);
}

#[test]
#[should_panic(expected = "need at least 1 maturity")]
fn surface_no_maturities() {
    fd_surface::<BsPutMoneyness<false>, f64>(&[R_, Q_, SIG], &grid(0.01, 0.01), &[]);
}

#[test]
#[should_panic(expected = "SurfacePoint::m out of range")]
fn book_maturity_out_of_range() {
    let pts = [SurfacePoint { m: 1, x: 0.0, w: 1.0 }];
    fd_surface_book::<BsPutMoneyness<false>, f64>(&[R_, Q_, SIG], &grid(0.01, 0.01), &[0.5], &pts);
}
