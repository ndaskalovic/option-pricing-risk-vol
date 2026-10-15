//! W-lane f32 vector as a plain array: LLVM vectorises the lane loops (NEON).
use std::ops::{Add, AddAssign, Div, Mul, Neg, Sub, SubAssign};
pub const W: usize = 32;

#[derive(Clone, Copy)]
#[repr(C, align(64))]
pub struct V(pub [f32; W]);

macro_rules! lanes { ($a:expr, |$i:ident| $e:expr) => {{ let mut r = [0f32; W]; for $i in 0..W { r[$i] = $e; } V(r) }}; }

/// exp via range reduction 2^n * p(r): plain arithmetic, so it vectorises (libm expf does not).
#[inline(always)]
pub fn exp_f32(x: f32) -> f32 {
    let x = x.clamp(-87.0, 88.0);
    let n = (x * std::f32::consts::LOG2_E).round();
    let r = x - n * 0.693_145_75 - n * 1.428_606_8e-6;
    let p = 1.0 + r * (1.0 + r * (0.5 + r * (0.166_666_67 + r * (0.041_666_668 + r * (0.008_333_334 + r * 0.001_388_889)))));
    p * f32::from_bits(((n as i32 + 127) as u32) << 23)
}

impl V {
    #[inline(always)] pub fn splat(v: f32) -> V { V([v; W]) }
    #[inline(always)] pub unsafe fn load(p: *const f32) -> V { V(std::ptr::read_unaligned(p as *const [f32; W])) }
    #[inline(always)] pub fn exp(self) -> V { lanes!(self, |i| exp_f32(self.0[i])) }
    #[inline(always)] pub fn mul_add(self, b: V, c: V) -> V { lanes!(self, |i| self.0[i].mul_add(b.0[i], c.0[i])) }
    #[inline(always)] pub fn ln(self) -> V { lanes!(self, |i| self.0[i].ln()) }
    #[inline(always)] pub fn sqrt(self) -> V { lanes!(self, |i| self.0[i].sqrt()) }
    #[inline(always)] pub fn relu(self) -> V { lanes!(self, |i| if self.0[i] > 0.0 { self.0[i] } else { 0.0 }) }
    #[inline(always)] pub fn relu_mask(self, a: V) -> V { lanes!(self, |i| if self.0[i] > 0.0 { a.0[i] } else { 0.0 }) }
    #[inline(always)] pub fn sum(self) -> f64 { self.0.iter().map(|&v| v as f64).sum() }
}
impl Add for V { type Output = V; #[inline(always)] fn add(self, o: V) -> V { lanes!(self, |i| self.0[i] + o.0[i]) } }
impl Sub for V { type Output = V; #[inline(always)] fn sub(self, o: V) -> V { lanes!(self, |i| self.0[i] - o.0[i]) } }
impl Mul for V { type Output = V; #[inline(always)] fn mul(self, o: V) -> V { lanes!(self, |i| self.0[i] * o.0[i]) } }
impl Div for V { type Output = V; #[inline(always)] fn div(self, o: V) -> V { lanes!(self, |i| self.0[i] / o.0[i]) } }
impl Neg for V { type Output = V; #[inline(always)] fn neg(self) -> V { lanes!(self, |i| -self.0[i]) } }
impl AddAssign for V { #[inline(always)] fn add_assign(&mut self, o: V) { for i in 0..W { self.0[i] += o.0[i]; } } }
impl SubAssign for V { #[inline(always)] fn sub_assign(&mut self, o: V) { for i in 0..W { self.0[i] -= o.0[i]; } } }
