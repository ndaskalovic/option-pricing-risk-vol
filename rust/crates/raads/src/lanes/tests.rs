//! Every pack op lane-by-lane against plain scalar arithmetic.

use super::*;
use crate::{AdNum, Dual};

fn check<L: Lanes>(name: &str) {
    let w = L::W;
    let xs: Vec<f64> = (0..w).map(|i| -3.0 + 6.5 * (i as f64 + 0.37) / w as f64).collect();
    let ys: Vec<f64> = (0..w).map(|i| 0.4 + 2.1 * ((i * 7 % w) as f64 + 0.5) / w as f64).collect();
    let x = L::from_fn(|i| xs[i]);
    let y = L::from_fn(|i| ys[i]);
    let f32_ = std::mem::size_of::<L::Elem>() == 4;
    let round = |v: f64| if f32_ { v as f32 as f64 } else { v };
    let lane = |v: L, i: usize| v.lane(i);
    for i in 0..w {
        let (a, b) = (round(xs[i]), round(ys[i]));
        let exact = |got: f64, want: f64, op: &str| {
            assert_eq!(got, round(want), "{name} {op} lane {i}: {got} vs {want}");
        };
        exact(lane(x + y, i), a + b, "add");
        exact(lane(x - y, i), a - b, "sub");
        exact(lane(x * y, i), a * b, "mul");
        exact(lane(x / y, i), a / b, "div");
        exact(lane(-x, i), -a, "neg");
        exact(lane(x.relu(), i), a.max(0.0), "relu");
        exact(lane(y.sqrt(), i), b.sqrt(), "sqrt");
        exact(lane(L::select_gt(x, y, x, y), i), if a > b { a } else { b }, "select_gt");
        exact(lane(x.mul_add(y, x), i), if f32_ { (a as f32).mul_add(b as f32, a as f32) as f64 } else { a.mul_add(b, a) }, "mul_add");
        exact(lane(x.nmul_add(y, x), i), if f32_ { (-(a as f32)).mul_add(b as f32, a as f32) as f64 } else { (-a).mul_add(b, a) }, "nmul_add");
        let tol = if f32_ { 3e-7 } else { 1e-15 };
        let rel = |got: f64, want: f64, op: &str| {
            assert!(((got - want) / want).abs() <= tol, "{name} {op} lane {i}: {got} vs {want}");
        };
        rel(lane(x.exp(), i), a.exp(), "exp");
        rel(lane((x * 25.0).exp(), i), round(a * 25.0).exp(), "exp wide range");
        rel(lane(y.ln(), i), b.ln(), "ln");
        rel(lane(y.powf(1.7), i), b.powf(1.7), "powf");
    }
    let s: f64 = (0..w).map(|i| round(xs[i])).sum();
    assert!((x.sum() - s).abs() < 1e-12, "{name} sum");
    let mut buf = vec![L::Elem::default(); w];
    x.store(&mut buf);
    assert_eq!(L::load(&buf), x, "{name} load/store");
    let f32s: Vec<f32> = xs.iter().map(|&v| v as f32).collect();
    for i in 0..w {
        assert_eq!(L::load_f32(&f32s).lane(i), round(f32s[i] as f64), "{name} load_f32");
    }
}

/// `exp` at edge inputs (overflow and underflow thresholds, ±inf, NaN, -0.0) in every lane,
/// against `std` in the pack's precision; `relu` of NaN and ±inf.
fn edges<L: Lanes>(name: &str) {
    let f32_ = std::mem::size_of::<L::Elem>() == 4;
    let (max, min_pos, tiny, tol) =
        if f32_ { (f32::MAX as f64, f32::MIN_POSITIVE as f64, 1.5e-45, 3e-7) } else { (f64::MAX, f64::MIN_POSITIVE, 5e-324, 1e-15) };
    let reference = |x: f64| if f32_ { (x as f32).exp() as f64 } else { x.exp() };
    let xs = [
        0.0, -0.0, 1e-30, -1e-30, 0.346_573_59, -0.346_573_59, 10.0, -10.0, 80.0, -80.0, 86.9, -86.9, 87.0, -87.0, -87.3, -87.4,
        -87.5, -87.6, -88.0, 88.0, 88.5, 88.7, 88.72, 88.73, 89.0, 89.5, -100.0, -103.0, -104.0, -110.0, -200.0, 100.0, 700.0,
        708.0, 709.0, 709.5, 709.7, 709.78, 709.79, 710.0, 711.0, -708.0, -708.3, -708.5, -708.7, -709.0, -720.0, -745.0,
        -746.0, -750.0, 1e10, -1e10, f64::MAX, f64::MIN, f64::INFINITY, f64::NEG_INFINITY, f64::NAN,
    ];
    for (b, xb) in xs.chunks(L::W).enumerate() {
        let e = L::from_fn(|i| xb[i % xb.len()]).exp();
        for (i, &x) in xb.iter().enumerate() {
            let (got, want) = (e.lane(i), reference(x));
            let ok = if want.is_nan() {
                got.is_nan()
            } else if want.is_infinite() || got.is_infinite() {
                got == want || got.min(want) >= max * (1.0 - tol)
            } else if want < 1.5 * min_pos {
                // flushed to 0 by the exponent-bit backends, else (gradual) underflow
                got == 0.0 || (got - want).abs() <= tol * want + tiny
            } else {
                ((got - want) / want).abs() <= tol
            };
            assert!(ok, "{name} exp({x}) lane {}: {got:e} vs {want:e}", b * L::W + i);
            if x == f64::NEG_INFINITY || x < -760.0 {
                assert_eq!(got, 0.0, "{name} exp({x})");
            }
        }
    }
    let r = L::from_fn(|i| [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -2.0][i % 4]).relu();
    for i in 0..L::W {
        let got = r.lane(i);
        match i % 4 {
            0 => assert!(got.is_nan(), "{name} relu(NaN) = {got}"),
            1 => assert_eq!(got, f64::INFINITY, "{name} relu(inf)"),
            _ => assert_eq!(got, 0.0, "{name} relu(negative)"),
        }
    }
    let d = Dual::new(L::from(f64::NAN), L::from(1.0)).relu();
    assert!(d.v.lane(0).is_nan(), "{name} Dual relu(NaN)");
}

fn check_all<L: Lanes>(name: &str) {
    check::<L>(name);
    edges::<L>(name);
}

#[test]
fn scalar_f64() {
    check_all::<f64>("f64");
}

#[cfg(target_arch = "aarch64")]
#[test]
fn neon() {
    check_all::<F32x4>("F32x4");
    check_all::<F32x32>("F32x32");
    check_all::<F64x2>("F64x2");
    check_all::<F64x16>("F64x16");
}

#[test]
fn portable() {
    use super::portable::{F32, F64};
    check_all::<F32<1>>("portable F32<1>");
    check_all::<F32<8>>("portable F32<8>");
    check_all::<F64<1>>("portable F64<1>");
    check_all::<F64<4>>("portable F64<4>");
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma"))]
#[test]
fn avx2() {
    use super::avx2::{F32, F32h, F64, F64h};
    check_all::<F32h<1>>("F32h<1>");
    check_all::<F32h<2>>("F32h<2>");
    check_all::<F64h<1>>("F64h<1>");
    check_all::<F64h<2>>("F64h<2>");
    check_all::<F32<1>>("F32<1>");
    check_all::<F32<4>>("F32<4>");
    check_all::<F64<1>>("F64<1>");
    check_all::<F64<4>>("F64<4>");
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
#[test]
fn avx512() {
    use super::avx512::{F32, F64};
    check_all::<F32<1>>("F32<1>");
    check_all::<F32<2>>("F32<2>");
    check_all::<F64<1>>("F64<1>");
    check_all::<F64<2>>("F64<2>");
}

/// The width aliases of whichever backend this target selected.
#[test]
fn aliases() {
    check_all::<F32x4>("F32x4");
    check_all::<F32x8>("F32x8");
    check_all::<F32x16>("F32x16");
    check_all::<F32x32>("F32x32");
    check_all::<F64x2>("F64x2");
    check_all::<F64x4>("F64x4");
    check_all::<F64x8>("F64x8");
    check_all::<F64x16>("F64x16");
}
