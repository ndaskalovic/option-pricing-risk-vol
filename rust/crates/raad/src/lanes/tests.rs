//! Every pack op lane-by-lane against plain scalar arithmetic.

use super::*;

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

#[test]
fn scalar_f64() {
    check::<f64>("f64");
}

#[cfg(target_arch = "aarch64")]
#[test]
fn neon() {
    check::<F32x4>("F32x4");
    check::<F32x32>("F32x32");
    check::<F64x2>("F64x2");
    check::<F64x16>("F64x16");
}

#[test]
fn portable() {
    use super::portable::{F32, F64};
    check::<F32<1>>("portable F32<1>");
    check::<F32<8>>("portable F32<8>");
    check::<F64<1>>("portable F64<1>");
    check::<F64<4>>("portable F64<4>");
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma"))]
#[test]
fn avx2() {
    use super::avx2::{F32, F32h, F64, F64h};
    check::<F32h<1>>("F32h<1>");
    check::<F32h<2>>("F32h<2>");
    check::<F64h<1>>("F64h<1>");
    check::<F64h<2>>("F64h<2>");
    check::<F32<1>>("F32<1>");
    check::<F32<4>>("F32<4>");
    check::<F64<1>>("F64<1>");
    check::<F64<4>>("F64<4>");
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
#[test]
fn avx512() {
    use super::avx512::{F32, F64};
    check::<F32<1>>("F32<1>");
    check::<F32<2>>("F32<2>");
    check::<F64<1>>("F64<1>");
    check::<F64<2>>("F64<2>");
}

/// The width aliases of whichever backend this target selected.
#[test]
fn aliases() {
    check::<F32x4>("F32x4");
    check::<F32x8>("F32x8");
    check::<F32x16>("F32x16");
    check::<F32x32>("F32x32");
    check::<F64x2>("F64x2");
    check::<F64x4>("F64x4");
    check::<F64x8>("F64x8");
    check::<F64x16>("F64x16");
}
