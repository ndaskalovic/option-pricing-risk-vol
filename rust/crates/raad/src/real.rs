//! Runtime value types for compiled kernels and engines.
//!
//! [`Real`] is what generated code and engines compute with. Every [`Lanes`] pack is a
//! `Real` (first order); [`Dual`] adds one tangent per lane, so running a reverse sweep
//! on `Dual` values gives forward-over-reverse: Hessian-vector products.

use crate::AdNum;
use crate::lanes::Lanes;
use std::ops::{Add, AddAssign, Div, Mul, Neg, Sub, SubAssign};

pub trait Real: AdNum + AddAssign + SubAssign + Send + Sync + 'static {
    /// The SIMD pack holding values (and tangents).
    type Lanes: Lanes;
    /// Constant (zero tangent).
    #[inline(always)]
    fn splat(k: f64) -> Self {
        Self::from(k)
    }
    /// Constant pack (zero tangent).
    fn lift(v: Self::Lanes) -> Self;
    fn value(self) -> Self::Lanes;
    fn tangent(self) -> Self::Lanes;
    /// `self * b + c`, fused.
    fn mul_add(self, b: Self, c: Self) -> Self;
    /// `c - self * b`, fused.
    fn nmul_add(self, b: Self, c: Self) -> Self;
    /// Additive identity with every component `-0.0`, so `acc + x` folds to `x`.
    fn neg_zero() -> Self;
    /// Lane `l` of the result is lane `l` of `f(l)` (per-lane gathers, e.g. interpolation).
    fn per_lane(f: impl Fn(usize) -> Self) -> Self;
}

/// Value and one tangent direction, lane-wise.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Dual<L> {
    pub v: L,
    pub t: L,
}

impl<L: Lanes> Dual<L> {
    #[inline(always)]
    pub fn new(v: L, t: L) -> Self {
        Dual { v, t }
    }
}

impl<L: Lanes> From<f64> for Dual<L> {
    #[inline(always)]
    fn from(k: f64) -> Self {
        Dual { v: L::from(k), t: L::from(0.0) }
    }
}

impl<L: Lanes> Add for Dual<L> {
    type Output = Self;
    #[inline(always)]
    fn add(self, r: Self) -> Self {
        Dual { v: self.v + r.v, t: self.t + r.t }
    }
}
impl<L: Lanes> Sub for Dual<L> {
    type Output = Self;
    #[inline(always)]
    fn sub(self, r: Self) -> Self {
        Dual { v: self.v - r.v, t: self.t - r.t }
    }
}
impl<L: Lanes> Mul for Dual<L> {
    type Output = Self;
    #[inline(always)]
    fn mul(self, r: Self) -> Self {
        Dual { v: self.v * r.v, t: self.v.mul_add(r.t, self.t * r.v) }
    }
}
impl<L: Lanes> Div for Dual<L> {
    type Output = Self;
    #[inline(always)]
    fn div(self, r: Self) -> Self {
        let q = self.v / r.v;
        Dual { v: q, t: q.nmul_add(r.t, self.t) / r.v }
    }
}
impl<L: Lanes> Neg for Dual<L> {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        Dual { v: -self.v, t: -self.t }
    }
}
impl<L: Lanes> Add<f64> for Dual<L> {
    type Output = Self;
    #[inline(always)]
    fn add(self, k: f64) -> Self {
        Dual { v: self.v + k, t: self.t }
    }
}
impl<L: Lanes> Sub<f64> for Dual<L> {
    type Output = Self;
    #[inline(always)]
    fn sub(self, k: f64) -> Self {
        Dual { v: self.v - k, t: self.t }
    }
}
impl<L: Lanes> Mul<f64> for Dual<L> {
    type Output = Self;
    #[inline(always)]
    fn mul(self, k: f64) -> Self {
        Dual { v: self.v * k, t: self.t * k }
    }
}
impl<L: Lanes> Div<f64> for Dual<L> {
    type Output = Self;
    #[inline(always)]
    fn div(self, k: f64) -> Self {
        Dual { v: self.v / k, t: self.t / k }
    }
}
impl<L: Lanes> AddAssign for Dual<L> {
    #[inline(always)]
    fn add_assign(&mut self, r: Self) {
        *self = *self + r;
    }
}
impl<L: Lanes> SubAssign for Dual<L> {
    #[inline(always)]
    fn sub_assign(&mut self, r: Self) {
        *self = *self - r;
    }
}

impl<L: Lanes> AdNum for Dual<L> {
    #[inline(always)]
    fn exp(self) -> Self {
        let e = self.v.exp();
        Dual { v: e, t: e * self.t }
    }
    #[inline(always)]
    fn ln(self) -> Self {
        Dual { v: self.v.ln(), t: self.t / self.v }
    }
    #[inline(always)]
    fn sqrt(self) -> Self {
        let s = self.v.sqrt();
        Dual { v: s, t: self.t * 0.5 / s }
    }
    #[inline(always)]
    fn powf(self, k: f64) -> Self {
        Dual { v: self.v.powf(k), t: self.v.powf(k - 1.0) * k * self.t }
    }
    #[inline(always)]
    fn relu(self) -> Self {
        let z = L::from(0.0);
        Dual { v: self.v.relu(), t: L::select_gt(self.v, z, self.t, z) }
    }
    #[inline(always)]
    fn detach(self) -> Self {
        Dual { v: self.v, t: L::from(0.0) }
    }
    #[inline(always)]
    fn select_gt(a: Self, b: Self, x: Self, y: Self) -> Self {
        Dual { v: L::select_gt(a.v, b.v, x.v, y.v), t: L::select_gt(a.v, b.v, x.t, y.t) }
    }
    #[inline(always)]
    fn rsub(self, k: f64) -> Self {
        Dual { v: self.v.rsub(k), t: -self.t }
    }
    #[inline(always)]
    fn rdiv(self, k: f64) -> Self {
        let q = self.v.rdiv(k);
        Dual { v: q, t: -(q * self.t) / self.v }
    }
}

impl<L: Lanes> Real for Dual<L> {
    type Lanes = L;
    #[inline(always)]
    fn lift(v: L) -> Self {
        Dual { v, t: L::from(0.0) }
    }
    #[inline(always)]
    fn value(self) -> L {
        self.v
    }
    #[inline(always)]
    fn tangent(self) -> L {
        self.t
    }
    #[inline(always)]
    fn mul_add(self, b: Self, c: Self) -> Self {
        Dual { v: self.v.mul_add(b.v, c.v), t: self.v.mul_add(b.t, self.t.mul_add(b.v, c.t)) }
    }
    #[inline(always)]
    fn nmul_add(self, b: Self, c: Self) -> Self {
        Dual { v: self.v.nmul_add(b.v, c.v), t: self.v.nmul_add(b.t, self.t.nmul_add(b.v, c.t)) }
    }
    #[inline(always)]
    fn neg_zero() -> Self {
        Dual { v: L::neg_zero(), t: L::neg_zero() }
    }
    #[inline]
    fn per_lane(f: impl Fn(usize) -> Self) -> Self {
        Dual { v: L::from_fn(|l| f(l).v.lane(l)), t: L::from_fn(|l| f(l).t.lane(l)) }
    }
}
