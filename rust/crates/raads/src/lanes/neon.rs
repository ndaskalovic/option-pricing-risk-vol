//! NEON packs: `F32<R>` = R × `float32x4_t` (4R lanes), `F64<R>` = R × `float64x2_t` (2R lanes).

#![allow(unused_unsafe)]

use super::{Lanes, exp_poly};
use crate::{AdNum, Real};
use core::arch::aarch64::*;
use std::fmt;
use std::ops::{Add, AddAssign, Div, Mul, Neg, Sub, SubAssign};

/// [`exp_poly`] with exponent-bit scaling.
#[inline(always)]
fn exp_f32x4(x: float32x4_t) -> float32x4_t {
    use exp_poly::f32::bits::*;
    unsafe {
        let x = vminq_f32(vmaxq_f32(x, vdupq_n_f32(LO)), vdupq_n_f32(HI));
        let n = vrndnq_f32(vmulq_f32(x, vdupq_n_f32(LOG2E)));
        let r = vfmsq_f32(vfmsq_f32(x, n, vdupq_n_f32(LN2_HI)), n, vdupq_n_f32(LN2_LO));
        let mut p = vdupq_n_f32(P[0]);
        for &c in &P[1..] {
            p = vfmaq_f32(vdupq_n_f32(c), p, r);
        }
        let e = vshlq_n_s32::<23>(vaddq_s32(vcvtq_s32_f32(n), vdupq_n_s32(126)));
        vmulq_f32(p, vreinterpretq_f32_s32(e))
    }
}

#[inline(always)]
fn exp_f64x2(x: float64x2_t) -> float64x2_t {
    use exp_poly::f64::bits::*;
    unsafe {
        let x = vminq_f64(vmaxq_f64(x, vdupq_n_f64(LO)), vdupq_n_f64(HI));
        let n = vrndnq_f64(vmulq_f64(x, vdupq_n_f64(LOG2E)));
        let r = vfmsq_f64(vfmsq_f64(x, n, vdupq_n_f64(LN2_HI)), n, vdupq_n_f64(LN2_LO));
        let mut p = vdupq_n_f64(P[0]);
        for &c in &P[1..] {
            p = vfmaq_f64(vdupq_n_f64(c), p, r);
        }
        let e = vshlq_n_s64::<52>(vaddq_s64(vcvtq_s64_f64(n), vdupq_n_s64(1022)));
        vmulq_f64(p, vreinterpretq_f64_s64(e))
    }
}

#[inline(always)]
unsafe fn ld_f32_as_f32(p: *const f32) -> float32x4_t {
    unsafe { vld1q_f32(p) }
}

#[inline(always)]
unsafe fn ld_f32_as_f64(p: *const f32) -> float64x2_t {
    unsafe { vcvt_f64_f32(vld1_f32(p)) }
}

macro_rules! pack {
    (
        $(#[$doc:meta])* $T:ident, $e:ident, $reg:ty, $n:literal, $exp:ident, $ld_f32:ident,
        $dup:ident, $ld:ident, $st:ident, $add:ident, $sub:ident, $mul:ident, $div:ident, $neg:ident,
        $fma:ident, $fms:ident, $sqrt:ident, $max:ident, $cgt:ident, $bsl:ident
    ) => {
        $(#[$doc])*
        #[derive(Clone, Copy)]
        pub struct $T<const R: usize>(pub [$reg; R]);

        impl<const R: usize> $T<R> {
            #[inline(always)]
            fn dup(k: $e) -> $reg {
                unsafe { $dup(k) }
            }
            #[inline(always)]
            fn map(self, f: impl Fn($reg) -> $reg) -> Self {
                Self(std::array::from_fn(|i| f(self.0[i])))
            }
            #[inline(always)]
            fn zip(self, o: Self, f: impl Fn($reg, $reg) -> $reg) -> Self {
                Self(std::array::from_fn(|i| f(self.0[i], o.0[i])))
            }
            #[inline(always)]
            fn regs(self) -> [[$e; $n]; R] {
                self.0.map(|r| {
                    let mut a = [0.0; $n];
                    unsafe { $st(a.as_mut_ptr(), r) };
                    a
                })
            }
            #[inline(always)]
            fn per_lane(self, f: impl Fn($e) -> $e) -> Self {
                self.map(|r| {
                    let mut a = [0.0; $n];
                    unsafe { $st(a.as_mut_ptr(), r) };
                    let a = a.map(&f);
                    unsafe { $ld(a.as_ptr()) }
                })
            }
        }

        impl<const R: usize> fmt::Debug for $T<R> {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.debug_list().entries(self.regs().iter().flatten()).finish()
            }
        }
        impl<const R: usize> PartialEq for $T<R> {
            fn eq(&self, o: &Self) -> bool {
                self.regs() == o.regs()
            }
        }
        impl<const R: usize> From<f64> for $T<R> {
            #[inline(always)]
            fn from(k: f64) -> Self {
                Self([Self::dup(k as $e); R])
            }
        }

        impl<const R: usize> Add for $T<R> {
            type Output = Self;
            #[inline(always)]
            fn add(self, o: Self) -> Self {
                self.zip(o, |a, b| unsafe { $add(a, b) })
            }
        }
        impl<const R: usize> Sub for $T<R> {
            type Output = Self;
            #[inline(always)]
            fn sub(self, o: Self) -> Self {
                self.zip(o, |a, b| unsafe { $sub(a, b) })
            }
        }
        impl<const R: usize> Mul for $T<R> {
            type Output = Self;
            #[inline(always)]
            fn mul(self, o: Self) -> Self {
                self.zip(o, |a, b| unsafe { $mul(a, b) })
            }
        }
        impl<const R: usize> Div for $T<R> {
            type Output = Self;
            #[inline(always)]
            fn div(self, o: Self) -> Self {
                self.zip(o, |a, b| unsafe { $div(a, b) })
            }
        }
        impl<const R: usize> Neg for $T<R> {
            type Output = Self;
            #[inline(always)]
            fn neg(self) -> Self {
                self.map(|a| unsafe { $neg(a) })
            }
        }
        impl<const R: usize> Add<f64> for $T<R> {
            type Output = Self;
            #[inline(always)]
            fn add(self, k: f64) -> Self {
                self + Self::from(k)
            }
        }
        impl<const R: usize> Sub<f64> for $T<R> {
            type Output = Self;
            #[inline(always)]
            fn sub(self, k: f64) -> Self {
                self - Self::from(k)
            }
        }
        impl<const R: usize> Mul<f64> for $T<R> {
            type Output = Self;
            #[inline(always)]
            fn mul(self, k: f64) -> Self {
                self * Self::from(k)
            }
        }
        impl<const R: usize> Div<f64> for $T<R> {
            type Output = Self;
            #[inline(always)]
            fn div(self, k: f64) -> Self {
                self / Self::from(k)
            }
        }
        impl<const R: usize> AddAssign for $T<R> {
            #[inline(always)]
            fn add_assign(&mut self, o: Self) {
                *self = *self + o;
            }
        }
        impl<const R: usize> SubAssign for $T<R> {
            #[inline(always)]
            fn sub_assign(&mut self, o: Self) {
                *self = *self - o;
            }
        }

        impl<const R: usize> AdNum for $T<R> {
            #[inline(always)]
            fn exp(self) -> Self {
                self.map($exp)
            }
            #[inline(always)]
            fn ln(self) -> Self {
                self.per_lane($e::ln)
            }
            #[inline(always)]
            fn sqrt(self) -> Self {
                self.map(|a| unsafe { $sqrt(a) })
            }
            #[inline(always)]
            fn powf(self, k: f64) -> Self {
                self.per_lane(|x| x.powf(k as $e))
            }
            #[inline(always)]
            fn relu(self) -> Self {
                let z = Self::dup(0.0);
                self.map(|a| unsafe { $max(a, z) })
            }
            #[inline(always)]
            fn detach(self) -> Self {
                self
            }
            #[inline(always)]
            fn select_gt(a: Self, b: Self, x: Self, y: Self) -> Self {
                Self(std::array::from_fn(|i| unsafe { $bsl($cgt(a.0[i], b.0[i]), x.0[i], y.0[i]) }))
            }
            #[inline(always)]
            fn rsub(self, k: f64) -> Self {
                Self::from(k) - self
            }
            #[inline(always)]
            fn rdiv(self, k: f64) -> Self {
                Self::from(k) / self
            }
        }

        impl<const R: usize> Real for $T<R> {
            type Lanes = Self;
            #[inline(always)]
            fn lift(v: Self) -> Self {
                v
            }
            #[inline(always)]
            fn value(self) -> Self {
                self
            }
            #[inline(always)]
            fn tangent(self) -> Self {
                Self::from(0.0)
            }
            #[inline(always)]
            fn mul_add(self, b: Self, c: Self) -> Self {
                Self(std::array::from_fn(|i| unsafe { $fma(c.0[i], self.0[i], b.0[i]) }))
            }
            #[inline(always)]
            fn nmul_add(self, b: Self, c: Self) -> Self {
                Self(std::array::from_fn(|i| unsafe { $fms(c.0[i], self.0[i], b.0[i]) }))
            }
            #[inline(always)]
            fn neg_zero() -> Self {
                Self([Self::dup(-0.0); R])
            }
            #[inline]
            fn per_lane(f: impl Fn(usize) -> Self) -> Self {
                <Self as Lanes>::from_fn(|l| f(l).lane(l))
            }
        }

        impl<const R: usize> Lanes for $T<R> {
            type Elem = $e;
            const W: usize = $n * R;
            #[inline(always)]
            fn load(src: &[$e]) -> Self {
                assert!(src.len() >= Self::W);
                Self(std::array::from_fn(|i| unsafe { $ld(src.as_ptr().add($n * i)) }))
            }
            #[inline(always)]
            fn load_f32(src: &[f32]) -> Self {
                assert!(src.len() >= Self::W);
                Self(std::array::from_fn(|i| unsafe { $ld_f32(src.as_ptr().add($n * i)) }))
            }
            #[inline(always)]
            fn store(self, dst: &mut [$e]) {
                assert!(dst.len() >= Self::W);
                for (i, r) in self.0.iter().enumerate() {
                    unsafe { $st(dst.as_mut_ptr().add($n * i), *r) };
                }
            }
            #[inline(always)]
            fn lane(self, i: usize) -> f64 {
                self.regs()[i / $n][i % $n] as f64
            }
            #[inline(always)]
            fn from_fn(mut f: impl FnMut(usize) -> f64) -> Self {
                Self(std::array::from_fn(|r| {
                    let a: [$e; $n] = std::array::from_fn(|l| f($n * r + l) as $e);
                    unsafe { $ld(a.as_ptr()) }
                }))
            }
            #[inline(always)]
            fn sum(self) -> f64 {
                self.regs().iter().flatten().map(|&x| x as f64).sum()
            }
        }
    };
}

pack!(
    /// `4R` lanes of `f32`.
    F32, f32, float32x4_t, 4, exp_f32x4, ld_f32_as_f32,
    vdupq_n_f32, vld1q_f32, vst1q_f32, vaddq_f32, vsubq_f32, vmulq_f32, vdivq_f32, vnegq_f32,
    vfmaq_f32, vfmsq_f32, vsqrtq_f32, vmaxq_f32, vcgtq_f32, vbslq_f32
);
pack!(
    /// `2R` lanes of `f64`.
    F64, f64, float64x2_t, 2, exp_f64x2, ld_f32_as_f64,
    vdupq_n_f64, vld1q_f64, vst1q_f64, vaddq_f64, vsubq_f64, vmulq_f64, vdivq_f64, vnegq_f64,
    vfmaq_f64, vfmsq_f64, vsqrtq_f64, vmaxq_f64, vcgtq_f64, vbslq_f64
);
