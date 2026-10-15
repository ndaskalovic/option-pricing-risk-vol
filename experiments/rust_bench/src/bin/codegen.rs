//! build.rs variant: model traced at build time, fwd/bwd emitted as straight-line Rust.
#![cfg_attr(feature = "portable", feature(portable_simd))]
#[cfg(not(feature = "portable"))]
#[path = "../simd.rs"] mod simd;
#[cfg(feature = "portable")]
#[path = "../simd_portable.rs"] mod simd;
#[path = "../common.rs"] mod common;
mod kernels { include!(concat!(env!("OUT_DIR"), "/kernels.rs")); }
use common::*;
use kernels::*;
use simd::{V, W};

#[inline(never)]
fn mc(params: [f32; 5], z: &[f32]) -> (f64, [f64; 5]) {
    let n_paths = NPATHS / W;
    let p: [V; 5] = params.map(V::splat);
    let nd = [V::splat(NS as f32)];
    let c = setup_eval(&p, &nd);
    let s0 = init_eval(&p, &nd);
    let mut res = vec![V::splat(0.0); (NS * STEP_NRES).max(1)];
    let (mut a_c, mut a_s0, mut a_p) = ([V::splat(0.0); 2], [V::splat(0.0); 1], [V::splat(0.0); 5]);
    let mut acc = V::splat(0.0);
    for i in 0..n_paths {
        let zp = &z[i * NS * W..(i + 1) * NS * W];
        // unaligned load of one step's W draws, no copy through a temporary slice
        let zk = |k: usize| -> [V; 1] { unsafe { [V::load(zp.as_ptr().add(k * W))] } };
        // scan forward
        let mut x = [s0[0], c[0], c[1]];
        for k in 0..NS {
            let y = if STEP_NRES == 0 { step_eval(&x, &zk(k)) } else { step_fwd(&x, &zk(k), &mut res[k * STEP_NRES..]) };
            x[0] = y[0];
        }
        // payoff value + reverse (fused)
        let px = [x[0], p[0], p[1], p[2], p[3], p[4], c[0], c[1]];
        let mut a_px = [V::splat(-0.0); 8];
        let y = payoff_vjp(&px, &nd, &[V::splat(1.0)], &mut a_px);
        acc += y[0];
        for j in 0..5 { a_p[j] += a_px[1 + j]; }
        a_c[0] += a_px[6]; a_c[1] += a_px[7];
        // scan reverse
        let mut a_s = a_px[0];
        // accumulate straight into a_c (the kernel fuses its products into these with FMA)
        let mut a_x = [V::splat(-0.0), a_c[0], a_c[1]];
        for k in (0..NS).rev() {
            a_x[0] = V::splat(-0.0);
            step_bwd(&zk(k), if STEP_NRES == 0 { &[] } else { &res[k * STEP_NRES..] }, &[a_s], &mut a_x);
            a_s = a_x[0];
        }
        a_c[0] = a_x[1]; a_c[1] = a_x[2];
        a_s0[0] += a_s;
    }
    let _ = init_vjp(&p, &nd, &a_s0, &mut a_p);
    let _ = setup_vjp(&p, &nd, &a_c, &mut a_p);
    let scale = 1.0 / (n_paths * W) as f64;
    (acc.sum() * scale, [a_p[0].sum() * scale, a_p[1].sum() * scale, a_p[2].sum() * scale, a_p[3].sum() * scale, a_p[4].sum() * scale])
}

/// Pricing only (no derivatives) with the same V type: isolates the lane-type codegen.
#[inline(never)]
fn primal(params: [f32; 5], z: &[f32]) -> f64 {
    let n_paths = NPATHS / W;
    let p: [V; 5] = params.map(V::splat);
    let nd = [V::splat(NS as f32)];
    let c = setup_eval(&p, &nd);
    let s0 = init_eval(&p, &nd);
    let mut acc = V::splat(0.0);
    for i in 0..n_paths {
        let zp = &z[i * NS * W..(i + 1) * NS * W];
        let mut x = s0[0];
        for k in 0..NS {
            let zv = unsafe { V::load(zp.as_ptr().add(k * W)) };
            x = c[1].mul_add(zv, x + c[0]);
        }
        acc += payoff_eval(&[x, p[0], p[1], p[2], p[3], p[4], c[0], c[1]], &nd)[0];
    }
    acc.sum() / (n_paths * W) as f64
}

/// Same, but the lane loop written directly over plain f32 (what LLVM autovectorises best).
#[inline(never)]
fn primal_scalar(params: [f32; 5], z: &[f32]) -> f64 {
    let n_paths = NPATHS / W;
    let dt = params[3] / NS as f32;
    let drift = (params[1] - 0.5 * params[2] * params[2]) * dt;
    let vsdt = params[2] * dt.sqrt();
    let x0 = params[0].ln();
    let mut acc = [0f32; W];
    for i in 0..n_paths {
        let mut x = [x0; W];
        for k in 0..NS {
            let b = (i * NS + k) * W;
            for l in 0..W { x[l] = x[l] + drift + vsdt * z[b + l]; }
        }
        for l in 0..W { let v = simd::exp_f32(x[l]) - params[4]; acc[l] += if v > 0.0 { v } else { 0.0 }; }
    }
    acc.iter().map(|&v| v as f64).sum::<f64>() / (n_paths * W) as f64
}

fn main() {
    let z = load_normals();
    let params = [100.0, 0.01, 0.2, 1.0, 100.0];
    let (price, g) = mc(params, &z);
    report(if cfg!(feature = "portable") { "rust build.rs codegen (std::simd) f32" } else { "rust build.rs codegen ([f32;W]) f32" }, price, g);
    if std::env::args().any(|a| a == "--bench") {
        let (ms, chk) = best_ms(31, || { let (p, g) = mc(params, &z); p + g.iter().sum::<f64>() });
        println!("rust  build.rs codegen {:<16} best ms {ms:.3}  chk {chk:.6}", if cfg!(feature = "portable") { "std::simd" } else { "[f32;W]" });
        let (ms, _) = best_ms(31, || primal(params, &z));
        println!("rust  primal, V type {:<18} best ms {ms:.3}", if cfg!(feature = "portable") { "std::simd" } else { "[f32;W]" });
        let (ms, _) = best_ms(31, || primal_scalar(params, &z));
        println!("rust  primal, plain f32 lane loops      best ms {ms:.3}");
    }
}
