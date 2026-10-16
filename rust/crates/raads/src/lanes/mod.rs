//! SIMD packs: W lanes of `f32` or `f64` (lanes = Monte Carlo paths or independent problems).
//!
//! Each instruction set has its own implementation behind [`Lanes`]: NEON on aarch64,
//! with AVX2/AVX-512 on x86_64 and plain `f64` for one lane. Engines and generated
//! kernels are written against [`Real`], so the same source compiles to each.
//! The packs are named by the aliases below (`F32x4` … `F64x16`) and `f64`.

use crate::Real;
use std::fmt::Debug;

mod scalar;

// the fallback backend, unused on targets with a SIMD backend except in tests
#[allow(dead_code)]
mod portable;

#[cfg(target_arch = "aarch64")]
mod neon;

#[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma"))]
mod avx2;

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
mod avx512;

macro_rules! seal {
    ($($m:ident::$T:ident),*) => {
        $(impl<const R: usize> crate::sealed::Sealed for $m::$T<R> {})*
    };
}
seal!(portable::F32, portable::F64);
#[cfg(target_arch = "aarch64")]
seal!(neon::F32, neon::F64);
#[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma"))]
seal!(avx2::F32, avx2::F64, avx2::F32h, avx2::F64h);
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
seal!(avx512::F32, avx512::F64);

/// A pack of `W` floats. Also a first-order [`Real`] (its own value, zero tangent).
/// Sealed through [`AdNum`](crate::AdNum).
pub trait Lanes: Real<Lanes = Self> + Debug + PartialEq {
    /// `f32` or `f64`.
    type Elem: Copy + Debug + Default + PartialEq + Into<f64>;
    /// Number of lanes.
    const W: usize;
    /// First `W` elements of `src`.
    fn load(src: &[Self::Elem]) -> Self;
    /// First `W` elements of `src`, converted (normals are stored as `f32` for every precision).
    fn load_f32(src: &[f32]) -> Self;
    /// Write the `W` lanes to `dst[..W]`.
    fn store(self, dst: &mut [Self::Elem]);
    /// Lane `i` as `f64`.
    fn lane(self, i: usize) -> f64;
    /// Pack with lane `i` = `f(i)`.
    fn from_fn(f: impl FnMut(usize) -> f64) -> Self;
    /// Sum of the lanes, accumulated in `f64`.
    #[inline]
    fn sum(self) -> f64 {
        (0..Self::W).map(|i| self.lane(i)).sum()
    }
}

/// Polynomial `exp` shared by the vector backends: `exp(x) = 2^n · p(r)`, `n = round(x / ln2)`,
/// `r = x - n·ln2` (two-constant Cody-Waite), `p` the Taylor polynomial in FMA Horner form.
/// Max relative error ≈ 2e-7 (`f32`, degree 6) and ≈ 6e-16 (`f64`, degree 12).
///
/// `x` is clamped to `[LO, HI]` first (NaN passes through). `HI` lies just above
/// `ln(MAX)`, so results are finite up to the true overflow threshold and `+inf` above it.
/// Two ways to apply `2^n`:
/// - `bits` (NEON, AVX2): exponent-bit arithmetic, with the coefficients doubled and the
///   scale `2^(n-1)` so that `n = 128` (`f64`: 1024) is representable. The lowest `n` gives an
///   exponent field of 0, i.e. 0: results below `2^-125.5` (`x < -86.99`; `f64`: `2^-1021.5`,
///   `x < -708.05`) flush to 0, subnormals and the bottom half binade of normals alike.
/// - `scalef` (AVX-512): `vscalef` applies `2^n` with correct rounding, overflow to `+inf`
///   and gradual underflow.
#[cfg(any(
    target_arch = "aarch64",
    all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma")
))]
pub(crate) mod exp_poly {
    pub mod f32 {
        pub const LOG2E: f32 = std::f32::consts::LOG2_E;
        pub const LN2_HI: f32 = 0.693_145_75;
        pub const LN2_LO: f32 = 1.428_606_8e-6;
        /// ln(f32::MAX) = 88.72; `n` stays ≤ 128.
        pub const HI: f32 = 89.0;
        pub mod bits {
            pub use super::{HI, LN2_HI, LN2_LO, LOG2E};
            /// `n = -126` here (exponent field 0 after the bias of 126).
            pub const LO: f32 = -87.6;
            /// `2·P`: `p(r) = 2e^r`.
            pub const P: [f32; 7] = [2.0 / 720.0, 2.0 / 120.0, 2.0 / 24.0, 2.0 / 6.0, 1.0, 2.0, 2.0];
        }
        #[cfg(target_feature = "avx512f")]
        pub mod scalef {
            pub use super::{HI, LN2_HI, LN2_LO, LOG2E};
            /// `e^LO < 2^-150` rounds to 0.
            pub const LO: f32 = -104.5;
            /// Taylor coefficients of `e^r`, highest degree first.
            pub const P: [f32; 7] = [1.0 / 720.0, 1.0 / 120.0, 1.0 / 24.0, 1.0 / 6.0, 0.5, 1.0, 1.0];
        }
    }
    pub mod f64 {
        pub const LOG2E: f64 = std::f64::consts::LOG2_E;
        pub const LN2_HI: f64 = 0.693_145_751_953_125;
        pub const LN2_LO: f64 = 1.428_606_820_309_417_3e-6;
        /// ln(f64::MAX) = 709.78; `n` stays ≤ 1024.
        pub const HI: f64 = 710.0;
        pub mod bits {
            pub use super::{HI, LN2_HI, LN2_LO, LOG2E};
            /// `n = -1022` here (exponent field 0 after the bias of 1022).
            pub const LO: f64 = -708.7;
            /// `2·P`: `p(r) = 2e^r`.
            pub const P: [f64; 13] = [
                2.0 / 479_001_600.0,
                2.0 / 39_916_800.0,
                2.0 / 3_628_800.0,
                2.0 / 362_880.0,
                2.0 / 40_320.0,
                2.0 / 5_040.0,
                2.0 / 720.0,
                2.0 / 120.0,
                2.0 / 24.0,
                2.0 / 6.0,
                1.0,
                2.0,
                2.0,
            ];
        }
        #[cfg(target_feature = "avx512f")]
        pub mod scalef {
            pub use super::{HI, LN2_HI, LN2_LO, LOG2E};
            /// `e^LO < 2^-1075` rounds to 0.
            pub const LO: f64 = -746.0;
            /// Taylor coefficients of `e^r`, highest degree first.
            pub const P: [f64; 13] = [
                1.0 / 479_001_600.0,
                1.0 / 39_916_800.0,
                1.0 / 3_628_800.0,
                1.0 / 362_880.0,
                1.0 / 40_320.0,
                1.0 / 5_040.0,
                1.0 / 720.0,
                1.0 / 120.0,
                1.0 / 24.0,
                1.0 / 6.0,
                0.5,
                1.0,
                1.0,
            ];
        }
    }
}

// Packs by precision and width for the target being compiled, chosen by the enabled target
// features (no runtime dispatch): NEON on aarch64; on x86_64 AVX-512 (`avx512f`), else AVX2
// (`avx2` + `fma`), else plain arrays (`portable`); `portable` on other targets. Packs narrower
// than a register use the 128-bit (AVX2) or 256-bit (AVX-512) packs, so every alias is one or
// more whole registers.
#[cfg(target_arch = "aarch64")]
mod aliases {
    use super::neon::{F32, F64};
    /// 4 lanes of `f32`.
    pub type F32x4 = F32<1>;
    /// 8 lanes of `f32`.
    pub type F32x8 = F32<2>;
    /// 16 lanes of `f32`.
    pub type F32x16 = F32<4>;
    /// 32 lanes of `f32`.
    pub type F32x32 = F32<8>;
    /// 2 lanes of `f64`.
    pub type F64x2 = F64<1>;
    /// 4 lanes of `f64`.
    pub type F64x4 = F64<2>;
    /// 8 lanes of `f64`.
    pub type F64x8 = F64<4>;
    /// 16 lanes of `f64`.
    pub type F64x16 = F64<8>;
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
mod aliases {
    use super::avx2::{F32, F32h, F64, F64h};
    use super::avx512::{F32 as F32z, F64 as F64z};
    /// 4 lanes of `f32`.
    pub type F32x4 = F32h<1>;
    /// 8 lanes of `f32`.
    pub type F32x8 = F32<1>;
    /// 16 lanes of `f32`.
    pub type F32x16 = F32z<1>;
    /// 32 lanes of `f32`.
    pub type F32x32 = F32z<2>;
    /// 2 lanes of `f64`.
    pub type F64x2 = F64h<1>;
    /// 4 lanes of `f64`.
    pub type F64x4 = F64<1>;
    /// 8 lanes of `f64`.
    pub type F64x8 = F64z<1>;
    /// 16 lanes of `f64`.
    pub type F64x16 = F64z<2>;
}

#[cfg(all(
    target_arch = "x86_64",
    not(target_feature = "avx512f"),
    target_feature = "avx2",
    target_feature = "fma"
))]
mod aliases {
    use super::avx2::{F32, F32h, F64, F64h};
    /// 4 lanes of `f32`.
    pub type F32x4 = F32h<1>;
    /// 8 lanes of `f32`.
    pub type F32x8 = F32<1>;
    /// 16 lanes of `f32`.
    pub type F32x16 = F32<2>;
    /// 32 lanes of `f32`.
    pub type F32x32 = F32<4>;
    /// 2 lanes of `f64`.
    pub type F64x2 = F64h<1>;
    /// 4 lanes of `f64`.
    pub type F64x4 = F64<1>;
    /// 8 lanes of `f64`.
    pub type F64x8 = F64<2>;
    /// 16 lanes of `f64`.
    pub type F64x16 = F64<4>;
}

#[cfg(not(any(
    target_arch = "aarch64",
    all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma"),
    all(target_arch = "x86_64", target_feature = "avx512f"),
)))]
mod aliases {
    use super::portable::{F32, F64};
    /// 4 lanes of `f32`.
    pub type F32x4 = F32<4>;
    /// 8 lanes of `f32`.
    pub type F32x8 = F32<8>;
    /// 16 lanes of `f32`.
    pub type F32x16 = F32<16>;
    /// 32 lanes of `f32`.
    pub type F32x32 = F32<32>;
    /// 2 lanes of `f64`.
    pub type F64x2 = F64<2>;
    /// 4 lanes of `f64`.
    pub type F64x4 = F64<4>;
    /// 8 lanes of `f64`.
    pub type F64x8 = F64<8>;
    /// 16 lanes of `f64`.
    pub type F64x16 = F64<16>;
}

pub use aliases::*;

#[cfg(test)]
mod tests;
