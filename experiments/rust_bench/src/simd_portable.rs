//! W-lane f32 vector on std::simd (nightly portable SIMD): the analogue of Mojo's SIMD[f32, W].
use std::ops::{Add, AddAssign, Div, Mul, Neg, Sub, SubAssign};
use std::simd::prelude::*;
use std::simd::StdFloat;
pub const W: usize = 32;
type S = Simd<f32, W>;

#[derive(Clone, Copy)]
pub struct V(pub S);

/// Scalar exp with the same polynomial (used by scalar code paths).
#[inline(always)]
pub fn exp_f32(x: f32) -> f32 { V::splat(x).exp().0[0] }

impl V {
    #[inline(always)] pub fn splat(v: f32) -> V { V(S::splat(v)) }
    #[inline(always)] pub unsafe fn load(p: *const f32) -> V { V(std::ptr::read_unaligned(p as *const S)) }
    #[inline(always)]
    pub fn exp(self) -> V {
        let x = self.0.simd_clamp(S::splat(-87.0), S::splat(88.0));
        let n = (x * S::splat(std::f32::consts::LOG2_E)).round();
        let r = x - n * S::splat(0.693_145_75) - n * S::splat(1.428_606_8e-6);
        let p = S::splat(1.0) + r * (S::splat(1.0) + r * (S::splat(0.5) + r * (S::splat(0.166_666_67)
            + r * (S::splat(0.041_666_668) + r * (S::splat(0.008_333_334) + r * S::splat(0.001_388_889))))));
        let bits = ((n.cast::<i32>() + Simd::splat(127)) << Simd::splat(23)).cast::<u32>();
        V(p * S::from_bits(bits))
    }
    #[inline(always)] pub fn mul_add(self, b: V, c: V) -> V { V(self.0.mul_add(b.0, c.0)) }
    #[inline(always)] pub fn ln(self) -> V { V(self.0.ln()) }
    #[inline(always)] pub fn sqrt(self) -> V { V(self.0.sqrt()) }
    #[inline(always)] pub fn relu(self) -> V { V(self.0.simd_max(S::splat(0.0))) }
    #[inline(always)] pub fn relu_mask(self, a: V) -> V { V(self.0.simd_gt(S::splat(0.0)).select(a.0, S::splat(0.0))) }
    #[inline(always)] pub fn sum(self) -> f64 { self.0.to_array().iter().map(|&v| v as f64).sum() }
}
impl Add for V { type Output = V; #[inline(always)] fn add(self, o: V) -> V { V(self.0 + o.0) } }
impl Sub for V { type Output = V; #[inline(always)] fn sub(self, o: V) -> V { V(self.0 - o.0) } }
impl Mul for V { type Output = V; #[inline(always)] fn mul(self, o: V) -> V { V(self.0 * o.0) } }
impl Div for V { type Output = V; #[inline(always)] fn div(self, o: V) -> V { V(self.0 / o.0) } }
impl Neg for V { type Output = V; #[inline(always)] fn neg(self) -> V { V(-self.0) } }
impl AddAssign for V { #[inline(always)] fn add_assign(&mut self, o: V) { self.0 += o.0; } }
impl SubAssign for V { #[inline(always)] fn sub_assign(&mut self, o: V) { self.0 -= o.0; } }
