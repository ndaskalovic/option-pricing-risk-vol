//! MC engine: price + every Greek vs pricing only (port of `benchmarks/bench_engine.mojo`).
//! GBM call, log space, 200k scalar paths x 52 steps, single core, best of 31.
//! Every output is consumed.

use raads::lanes::{F32x16, F32x32};
use raads::{Arr, Lanes, McModel, Real, mc_greeks};
use raads_models::gbm::{GbmAsianCall, GbmCall};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, StandardNormal};
use std::hint::black_box;
use std::time::Instant;

const N_STEPS: usize = 52;
const REPS: usize = 31;

fn best_ms(mut f: impl FnMut() -> f64) -> (f64, f64) {
    let mut best = f64::INFINITY;
    let mut chk = 0.0;
    for _ in 0..REPS {
        let t = Instant::now();
        chk += black_box(f());
        best = best.min(t.elapsed().as_secs_f64() * 1e3);
    }
    (best, chk)
}

/// Pricing only, same kernels (no derivatives).
fn price<M: McModel, R: Real>(p: &M::P<R>, n_paths: usize, z: &[f32]) -> f64 {
    let w = R::Lanes::W;
    let nz = <M::Z<R> as Arr<R>>::LEN;
    let c = M::setup_eval(p, R::from(N_STEPS as f64));
    let s0 = M::init_eval(p);
    let mut acc = R::from(0.0);
    for path in z.chunks_exact(N_STEPS * nz * w).take(n_paths) {
        let mut s = s0;
        for zk in path.chunks_exact(nz * w) {
            let mut zz: M::Z<R> = Arr::splat(R::from(0.0));
            for (j, x) in zz.as_mut().iter_mut().enumerate() {
                *x = R::lift(R::Lanes::load_f32(&zk[j * w..]));
            }
            s = M::step_eval(&s, &c, &zz);
        }
        acc += M::payoff_eval(&s, p, &c);
    }
    acc.value().sum() / (n_paths * w) as f64
}

fn run<M: McModel, L: Lanes>(label: &str, n_paths_scalar: usize, z: &[f32]) {
    let w = L::W;
    let n_paths = n_paths_scalar / w;
    let mut p: M::P<L> = Arr::splat(L::from(0.0));
    for (x, v) in p.as_mut().iter_mut().zip([100.0, 0.01, 0.2, 1.0, 100.0]) {
        *x = L::from(v);
    }
    let (tp, _) = best_ms(|| price::<M, L>(black_box(&p), n_paths, z));
    let (tg, chk) = best_ms(|| {
        let r = mc_greeks::<M, L>(black_box(&p), N_STEPS, n_paths, z);
        r.price + r.grad.as_ref().iter().sum::<f64>()
    });
    println!("{label} W={w}: pricing {tp:.3} ms | price + 5 Greeks {tg:.3} ms ({:.2}x) | chk {chk:.4}", tg / tp);
}

fn main() {
    let n = 200_000 * N_STEPS;
    let mut rng = StdRng::seed_from_u64(7);
    let z: Vec<f32> = (0..n).map(|_| StandardNormal.sample(&mut rng)).collect();
    run::<GbmCall, F32x32>("GBM call f32", 200_000, &z);
    run::<GbmCall, F32x16>("GBM call f32", 200_000, &z);
    run::<GbmAsianCall, F32x32>("Asian call f32", 200_000, &z);
}
