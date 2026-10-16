//! AVX-512 packs: `F32<R>` = R × `__m512` (16R lanes), `F64<R>` = R × `__m512d` (8R lanes).
//!
//! Compiled only when `avx512f` is enabled for the whole crate (e.g. `-C target-cpu=x86-64-v4`
//! or `-C target-feature=+avx512f`); `avx512f` implies `avx2` and `fma`, so the narrower
//! [`avx2`](super::avx2) packs are available too. Same pack generator as `avx2`: comparisons
//! go through mask registers, `exp` scales with `vscalefps` instead of exponent-bit arithmetic.

#![allow(unused_unsafe)]

use super::avx2::{NEAREST, x86_pack};

/// Register ops for `__m512` (16 × `f32`).
mod m512 {
    pub use crate::lanes::exp_poly::f32::scalef as exp;
    use core::arch::x86_64::*;
    pub type V = __m512;
    #[inline(always)]
    pub fn splat(k: f32) -> V {
        unsafe { _mm512_set1_ps(k) }
    }
    #[inline(always)]
    pub unsafe fn load(p: *const f32) -> V {
        unsafe { _mm512_loadu_ps(p) }
    }
    #[inline(always)]
    pub unsafe fn load_f32(p: *const f32) -> V {
        unsafe { _mm512_loadu_ps(p) }
    }
    #[inline(always)]
    pub unsafe fn store(p: *mut f32, a: V) {
        unsafe { _mm512_storeu_ps(p, a) }
    }
    #[inline(always)]
    pub fn add(a: V, b: V) -> V {
        unsafe { _mm512_add_ps(a, b) }
    }
    #[inline(always)]
    pub fn sub(a: V, b: V) -> V {
        unsafe { _mm512_sub_ps(a, b) }
    }
    #[inline(always)]
    pub fn mul(a: V, b: V) -> V {
        unsafe { _mm512_mul_ps(a, b) }
    }
    #[inline(always)]
    pub fn div(a: V, b: V) -> V {
        unsafe { _mm512_div_ps(a, b) }
    }
    #[inline(always)]
    pub fn neg(a: V) -> V {
        // `_mm512_xor_ps` needs AVX-512DQ; the integer xor is AVX-512F.
        unsafe { _mm512_castsi512_ps(_mm512_xor_epi32(_mm512_castps_si512(a), _mm512_set1_epi32(i32::MIN))) }
    }
    #[inline(always)]
    pub fn fma(a: V, b: V, c: V) -> V {
        unsafe { _mm512_fmadd_ps(a, b, c) }
    }
    #[inline(always)]
    pub fn fnma(a: V, b: V, c: V) -> V {
        unsafe { _mm512_fnmadd_ps(a, b, c) }
    }
    #[inline(always)]
    pub fn sqrt(a: V) -> V {
        unsafe { _mm512_sqrt_ps(a) }
    }
    #[inline(always)]
    pub fn min(a: V, b: V) -> V {
        unsafe { _mm512_min_ps(a, b) }
    }
    #[inline(always)]
    pub fn max(a: V, b: V) -> V {
        unsafe { _mm512_max_ps(a, b) }
    }
    #[inline(always)]
    pub fn sel_gt(a: V, b: V, x: V, y: V) -> V {
        unsafe { _mm512_mask_blend_ps(_mm512_cmp_ps_mask::<_CMP_GT_OQ>(a, b), y, x) }
    }
    #[inline(always)]
    pub fn round(a: V) -> V {
        unsafe { _mm512_roundscale_ps::<{ super::NEAREST }>(a) }
    }
    /// `p·2^n`.
    #[inline(always)]
    pub fn scale(p: V, n: V) -> V {
        unsafe { _mm512_scalef_ps(p, n) }
    }
}

/// Register ops for `__m512d` (8 × `f64`).
mod m512d {
    pub use crate::lanes::exp_poly::f64::scalef as exp;
    use core::arch::x86_64::*;
    pub type V = __m512d;
    #[inline(always)]
    pub fn splat(k: f64) -> V {
        unsafe { _mm512_set1_pd(k) }
    }
    #[inline(always)]
    pub unsafe fn load(p: *const f64) -> V {
        unsafe { _mm512_loadu_pd(p) }
    }
    #[inline(always)]
    pub unsafe fn load_f32(p: *const f32) -> V {
        unsafe { _mm512_cvtps_pd(_mm256_loadu_ps(p)) }
    }
    #[inline(always)]
    pub unsafe fn store(p: *mut f64, a: V) {
        unsafe { _mm512_storeu_pd(p, a) }
    }
    #[inline(always)]
    pub fn add(a: V, b: V) -> V {
        unsafe { _mm512_add_pd(a, b) }
    }
    #[inline(always)]
    pub fn sub(a: V, b: V) -> V {
        unsafe { _mm512_sub_pd(a, b) }
    }
    #[inline(always)]
    pub fn mul(a: V, b: V) -> V {
        unsafe { _mm512_mul_pd(a, b) }
    }
    #[inline(always)]
    pub fn div(a: V, b: V) -> V {
        unsafe { _mm512_div_pd(a, b) }
    }
    #[inline(always)]
    pub fn neg(a: V) -> V {
        // `_mm512_xor_pd` needs AVX-512DQ; the integer xor is AVX-512F.
        unsafe { _mm512_castsi512_pd(_mm512_xor_epi64(_mm512_castpd_si512(a), _mm512_set1_epi64(i64::MIN))) }
    }
    #[inline(always)]
    pub fn fma(a: V, b: V, c: V) -> V {
        unsafe { _mm512_fmadd_pd(a, b, c) }
    }
    #[inline(always)]
    pub fn fnma(a: V, b: V, c: V) -> V {
        unsafe { _mm512_fnmadd_pd(a, b, c) }
    }
    #[inline(always)]
    pub fn sqrt(a: V) -> V {
        unsafe { _mm512_sqrt_pd(a) }
    }
    #[inline(always)]
    pub fn min(a: V, b: V) -> V {
        unsafe { _mm512_min_pd(a, b) }
    }
    #[inline(always)]
    pub fn max(a: V, b: V) -> V {
        unsafe { _mm512_max_pd(a, b) }
    }
    #[inline(always)]
    pub fn sel_gt(a: V, b: V, x: V, y: V) -> V {
        unsafe { _mm512_mask_blend_pd(_mm512_cmp_pd_mask::<_CMP_GT_OQ>(a, b), y, x) }
    }
    #[inline(always)]
    pub fn round(a: V) -> V {
        unsafe { _mm512_roundscale_pd::<{ super::NEAREST }>(a) }
    }
    /// `p·2^n`.
    #[inline(always)]
    pub fn scale(p: V, n: V) -> V {
        unsafe { _mm512_scalef_pd(p, n) }
    }
}

x86_pack!(
    /// `16R` lanes of `f32`.
    F32, f32, 16, m512
);
x86_pack!(
    /// `8R` lanes of `f64`.
    F64, f64, 8, m512d
);
