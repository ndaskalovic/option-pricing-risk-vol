//! Portable packs: `F32<W>` = `[f32; W]`, `F64<W>` = `[f64; W]`, built on the std float methods.
//!
//! The fallback for targets without a SIMD backend (and x86_64 without AVX2 + FMA); LLVM
//! auto-vectorises the element loops where it can. `mul_add` is a fused multiply-add, which is a
//! libm call when the target has no FMA instruction.

use std::fmt;
use std::ops::{Add, AddAssign, Div, Mul, Neg, Sub, SubAssign};

use super::Lanes;
use crate::{AdNum, Real};

macro_rules! pack {
    ($(#[$doc:meta])* $T:ident, $e:ident) => {
        $(#[$doc])*
        #[derive(Clone, Copy, PartialEq)]
        pub struct $T<const W: usize>(pub [$e; W]);

        impl<const W: usize> $T<W> {
            #[inline(always)]
            fn map(self, f: impl Fn($e) -> $e) -> Self {
                Self(self.0.map(f))
            }
            #[inline(always)]
            fn zip(self, o: Self, f: impl Fn($e, $e) -> $e) -> Self {
                Self(std::array::from_fn(|i| f(self.0[i], o.0[i])))
            }
        }

        impl<const W: usize> fmt::Debug for $T<W> {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                self.0.fmt(f)
            }
        }
        impl<const W: usize> From<f64> for $T<W> {
            #[inline(always)]
            fn from(k: f64) -> Self {
                Self([k as $e; W])
            }
        }

        impl<const W: usize> Add for $T<W> {
            type Output = Self;
            #[inline(always)]
            fn add(self, o: Self) -> Self {
                self.zip(o, |a, b| a + b)
            }
        }
        impl<const W: usize> Sub for $T<W> {
            type Output = Self;
            #[inline(always)]
            fn sub(self, o: Self) -> Self {
                self.zip(o, |a, b| a - b)
            }
        }
        impl<const W: usize> Mul for $T<W> {
            type Output = Self;
            #[inline(always)]
            fn mul(self, o: Self) -> Self {
                self.zip(o, |a, b| a * b)
            }
        }
        impl<const W: usize> Div for $T<W> {
            type Output = Self;
            #[inline(always)]
            fn div(self, o: Self) -> Self {
                self.zip(o, |a, b| a / b)
            }
        }
        impl<const W: usize> Neg for $T<W> {
            type Output = Self;
            #[inline(always)]
            fn neg(self) -> Self {
                self.map(|a| -a)
            }
        }
        impl<const W: usize> Add<f64> for $T<W> {
            type Output = Self;
            #[inline(always)]
            fn add(self, k: f64) -> Self {
                self + Self::from(k)
            }
        }
        impl<const W: usize> Sub<f64> for $T<W> {
            type Output = Self;
            #[inline(always)]
            fn sub(self, k: f64) -> Self {
                self - Self::from(k)
            }
        }
        impl<const W: usize> Mul<f64> for $T<W> {
            type Output = Self;
            #[inline(always)]
            fn mul(self, k: f64) -> Self {
                self * Self::from(k)
            }
        }
        impl<const W: usize> Div<f64> for $T<W> {
            type Output = Self;
            #[inline(always)]
            fn div(self, k: f64) -> Self {
                self / Self::from(k)
            }
        }
        impl<const W: usize> AddAssign for $T<W> {
            #[inline(always)]
            fn add_assign(&mut self, o: Self) {
                *self = *self + o;
            }
        }
        impl<const W: usize> SubAssign for $T<W> {
            #[inline(always)]
            fn sub_assign(&mut self, o: Self) {
                *self = *self - o;
            }
        }

        impl<const W: usize> AdNum for $T<W> {
            #[inline(always)]
            fn exp(self) -> Self {
                self.map($e::exp)
            }
            #[inline(always)]
            fn ln(self) -> Self {
                self.map($e::ln)
            }
            #[inline(always)]
            fn sqrt(self) -> Self {
                self.map($e::sqrt)
            }
            #[inline(always)]
            fn powf(self, k: f64) -> Self {
                self.map(|x| x.powf(k as $e))
            }
            #[inline(always)]
            fn relu(self) -> Self {
                self.map(|a| if a < 0.0 { 0.0 } else { a })
            }
            #[inline(always)]
            fn detach(self) -> Self {
                self
            }
            #[inline(always)]
            fn select_gt(a: Self, b: Self, x: Self, y: Self) -> Self {
                Self(std::array::from_fn(|i| if a.0[i] > b.0[i] { x.0[i] } else { y.0[i] }))
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

        impl<const W: usize> Real for $T<W> {
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
                Self(std::array::from_fn(|i| self.0[i].mul_add(b.0[i], c.0[i])))
            }
            #[inline(always)]
            fn nmul_add(self, b: Self, c: Self) -> Self {
                Self(std::array::from_fn(|i| (-self.0[i]).mul_add(b.0[i], c.0[i])))
            }
            #[inline(always)]
            fn neg_zero() -> Self {
                Self([-0.0; W])
            }
            #[inline]
            fn per_lane(f: impl Fn(usize) -> Self) -> Self {
                Self(std::array::from_fn(|l| f(l).0[l]))
            }
        }

        impl<const W: usize> Lanes for $T<W> {
            type Elem = $e;
            const W: usize = W;
            #[inline(always)]
            fn load(src: &[$e]) -> Self {
                Self(src[..W].try_into().unwrap())
            }
            #[inline(always)]
            fn load_f32(src: &[f32]) -> Self {
                Self(std::array::from_fn(|i| src[..W][i] as $e))
            }
            #[inline(always)]
            fn store(self, dst: &mut [$e]) {
                dst[..W].copy_from_slice(&self.0);
            }
            #[inline(always)]
            fn lane(self, i: usize) -> f64 {
                self.0[i] as f64
            }
            #[inline(always)]
            fn from_fn(mut f: impl FnMut(usize) -> f64) -> Self {
                Self(std::array::from_fn(|i| f(i) as $e))
            }
        }
    };
}

pack!(
    /// `W` lanes of `f32`.
    F32, f32
);
pack!(
    /// `W` lanes of `f64`.
    F64, f64
);
