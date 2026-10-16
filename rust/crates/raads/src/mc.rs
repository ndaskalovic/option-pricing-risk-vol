//! Monte Carlo engine: price and gradient w.r.t. every model parameter (pathwise AAD).
//!
//! A model is four small kernels written once against [`AdNum`](crate::AdNum) inside an
//! `impl` block marked [`#[mc_model]`](crate::mc_model):
//!
//! ```text
//! setup(p, #[data] n_steps) -> c     once per run    (per-step coefficients)
//! init(p)                   -> s0    once per run    (initial state)
//! step(s, c, #[data] z)     -> s'    every step      (z: NZ normal draws)
//! payoff(s, p, c)           -> value every path
//! ```
//!
//! Per SIMD path the engine runs the steps forward (saving only the residuals the step's
//! reverse sweep reads), the payoff fused with its reverse sweep, and the steps backward;
//! the reverse of `init` and `setup` runs once per block of paths. Lanes are paths.

use crate::{Arr, Lanes, Real};

/// A model compiled by [`#[mc_model]`](crate::mc_model). The associated array types fix the
/// sizes: parameters `P`, per-step coefficients `C`, state `S`, normals per step `Z`.
///
/// Implement it with `#[mc_model]`, not by hand: the methods are generated (hidden from these
/// docs) and may change between minor versions.
pub trait McModel {
    /// Parameters.
    type P<T: Copy>: Arr<T>;
    /// Per-step coefficients.
    type C<T: Copy>: Arr<T>;
    /// Path state.
    type S<T: Copy>: Arr<T>;
    /// Normal draws per step.
    type Z<T: Copy>: Arr<T>;

    // Generated. `*_eval` compute a kernel; `*_vjp` add the adjoints of its inputs (seeded with
    // `a_y`) into the `a_*` arguments; `step_fwd` / `step_bwd` split the step's vector-Jacobian
    // product around the saved residuals `res`.
    /// Residuals `step_fwd` saves per step.
    #[doc(hidden)]
    const STEP_NRES: usize;

    /// `setup` kernel.
    #[doc(hidden)]
    fn setup_eval<R: Real>(p: &Self::P<R>, n_steps: R) -> Self::C<R>;
    /// `setup` reverse.
    #[doc(hidden)]
    fn setup_vjp<R: Real>(p: &Self::P<R>, n_steps: R, a_y: &Self::C<R>, a_p: &mut Self::P<R>) -> Self::C<R>;
    /// `init` kernel.
    #[doc(hidden)]
    fn init_eval<R: Real>(p: &Self::P<R>) -> Self::S<R>;
    /// `init` reverse.
    #[doc(hidden)]
    fn init_vjp<R: Real>(p: &Self::P<R>, a_y: &Self::S<R>, a_p: &mut Self::P<R>) -> Self::S<R>;
    /// `step` kernel.
    #[doc(hidden)]
    fn step_eval<R: Real>(s: &Self::S<R>, c: &Self::C<R>, z: &Self::Z<R>) -> Self::S<R>;
    /// `step` kernel, saving `STEP_NRES` residuals.
    #[doc(hidden)]
    fn step_fwd<R: Real>(s: &Self::S<R>, c: &Self::C<R>, z: &Self::Z<R>, res: &mut [R]) -> Self::S<R>;
    /// `step` reverse from the residuals.
    #[doc(hidden)]
    fn step_bwd<R: Real>(z: &Self::Z<R>, res: &[R], a_y: &Self::S<R>, a_s: &mut Self::S<R>, a_c: &mut Self::C<R>);
    /// `payoff` kernel.
    #[doc(hidden)]
    fn payoff_eval<R: Real>(s: &Self::S<R>, p: &Self::P<R>, c: &Self::C<R>) -> R;
    /// `payoff` and its reverse in one pass.
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    fn payoff_vjp<R: Real>(
        s: &Self::S<R>,
        p: &Self::P<R>,
        c: &Self::C<R>,
        a_y: R,
        a_s: &mut Self::S<R>,
        a_p: &mut Self::P<R>,
        a_c: &mut Self::C<R>,
    ) -> R;
}

/// Path means of the price and gradient, and their tangents: zero for a [`Lanes`] value
/// type; for [`Dual`](crate::Dual) parameters seeded with a direction v,
/// `price_tangent` = grad · v and `grad_tangent` = H v.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
#[must_use]
pub struct McResult<G> {
    /// Mean discounted payoff.
    pub price: f64,
    /// Standard error of `price`: sample standard deviation of the payoffs / sqrt(paths).
    pub std_err: f64,
    /// d price / d params.
    pub grad: G,
    /// d price / d params · v.
    pub price_tangent: f64,
    /// H v.
    pub grad_tangent: G,
}

/// Price and its standard error, from [`mc_price`].
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
#[must_use]
pub struct McPrice {
    /// Mean discounted payoff.
    pub price: f64,
    /// Standard error of `price`: sample standard deviation of the payoffs / sqrt(paths).
    pub std_err: f64,
}

/// Terms (path blocks × steps) summed in `R` before the sums are flushed into `f64`. Bounds the
/// accumulation error by the block (not by `n_paths`), which matters for `f32`.
const FLUSH_TERMS: usize = 4096;

/// Checks the arguments; returns the normals one path block reads (`n_steps * NZ * W`).
fn path_stride(n_steps: usize, n_paths: usize, nz: usize, w: usize, n_normals: usize) -> usize {
    assert!(n_steps >= 1 && n_paths >= 1 && nz >= 1, "need n_steps, n_paths and NZ >= 1");
    assert!(n_paths.is_multiple_of(w), "n_paths = {n_paths} must be a multiple of the lane count W = {w}");
    assert!(n_normals >= n_paths * n_steps * nz, "need n_paths * n_steps * NZ normals");
    n_steps * nz * w
}

/// `NZ` normals of one step, converted to `R`.
#[inline(always)]
fn load_z<M: McModel, R: Real>(zk: &[f32]) -> M::Z<R> {
    let w = R::Lanes::W;
    Arr::from_fn(|j| R::lift(R::Lanes::load_f32(&zk[j * w..])))
}

/// Mean payoff of the first path block, as a pack. Payoff sums are taken around it, so the
/// sum of squares does not cancel against the squared mean when the payoffs' spread is small
/// next to their mean (`f32`).
fn shift<M: McModel, R: Real>(params: &M::P<R>, c: &M::C<R>, s0: &M::S<R>, path: &[f32]) -> R::Lanes {
    let mut s = *s0;
    for zk in path.chunks_exact(<M::Z<R> as Arr<R>>::LEN * R::Lanes::W) {
        s = M::step_eval(&s, c, &load_z::<M, R>(zk));
    }
    R::Lanes::from(M::payoff_eval(&s, params, c).value().sum() / R::Lanes::W as f64)
}

/// Price and gradient (path means) in one pass.
///
/// `n_paths` counts scalar paths and must be a multiple of `W` (the lanes of `R`): lane `l`
/// of path block `i` is one path. `normals` holds `n_paths * n_steps * NZ` draws: block `i`
/// starts at `i * n_steps * NZ * W`, its step `k` at `+ k * NZ * W`, draw `j` at `+ j * W`.
///
/// Sums over paths are taken in `R` within groups of `ceil(4096 / n_steps)` path blocks, then
/// in `f64`. The reverse of `init` and `setup` is linear in its seeds, so it runs once per
/// group, on that group's sums.
///
/// # Panics
///
/// If `n_steps`, `n_paths` or `NZ` is zero, `n_paths` is not a multiple of `W`, or `normals`
/// is too short.
pub fn mc_greeks<M: McModel, R: Real>(params: &M::P<R>, n_steps: usize, n_paths: usize, normals: &[f32]) -> McResult<M::P<f64>> {
    let w = R::Lanes::W;
    let nz = <M::Z<R> as Arr<R>>::LEN;
    let nres = M::STEP_NRES;
    let stride = path_stride(n_steps, n_paths, nz, w, normals.len());

    // path-invariant parts: once per run
    let nd = R::from(n_steps as f64);
    let c = M::setup_eval(params, nd);
    let s0 = M::init_eval(params);
    let k = shift::<M, R>(params, &c, &s0, &normals[..stride]);
    let k_r = R::lift(k);

    let mut res = vec![R::from(0.0); n_steps * nres];
    let (mut sum, mut sum_sq, mut price_tangent) = (0.0, 0.0, 0.0);
    let mut grad: M::P<f64> = Arr::splat(0.0);
    let mut grad_tangent: M::P<f64> = Arr::splat(0.0);
    let flush = FLUSH_TERMS.div_ceil(n_steps);
    for block in normals[..n_paths * n_steps * nz].chunks(flush * stride) {
        let mut acc = R::neg_zero();
        let mut acc_sq = R::Lanes::neg_zero();
        let mut a_p: M::P<R> = Arr::splat(R::neg_zero());
        let mut a_c: M::C<R> = Arr::splat(R::neg_zero());
        let mut a_s0: M::S<R> = Arr::splat(R::neg_zero());
        for path in block.chunks_exact(stride) {
            let mut s = s0;
            for (k, zk) in path.chunks_exact(nz * w).enumerate() {
                let z = load_z::<M, R>(zk);
                s = if nres == 0 { M::step_eval(&s, &c, &z) } else { M::step_fwd(&s, &c, &z, &mut res[k * nres..(k + 1) * nres]) };
            }
            let mut a_s: M::S<R> = Arr::splat(R::neg_zero());
            let d = M::payoff_vjp(&s, params, &c, R::from(1.0), &mut a_s, &mut a_p, &mut a_c) - k_r;
            acc += d;
            acc_sq = d.value().mul_add(d.value(), acc_sq);
            for (k, zk) in path.chunks_exact(nz * w).enumerate().rev() {
                let z = load_z::<M, R>(zk);
                let mut a_prev: M::S<R> = Arr::splat(R::neg_zero());
                M::step_bwd(&z, &res[k * nres..(k + 1) * nres], &a_s, &mut a_prev, &mut a_c);
                a_s = a_prev;
            }
            for (a0, a) in a_s0.as_mut().iter_mut().zip(a_s.as_ref()) {
                *a0 += *a;
            }
        }

        M::init_vjp(params, &a_s0, &mut a_p);
        M::setup_vjp(params, nd, &a_c, &mut a_p);
        sum += acc.value().sum();
        sum_sq += acc_sq.sum();
        price_tangent += acc.tangent().sum();
        for (j, a) in a_p.as_ref().iter().enumerate() {
            grad.as_mut()[j] += a.value().sum();
            grad_tangent.as_mut()[j] += a.tangent().sum();
        }
    }

    let scale = 1.0 / n_paths as f64;
    for g in grad.as_mut().iter_mut().chain(grad_tangent.as_mut()) {
        *g *= scale;
    }
    let McPrice { price, std_err } = mean_and_err(k.lane(0), sum, sum_sq, n_paths);
    McResult { price, std_err, grad, price_tangent: price_tangent * scale, grad_tangent }
}

/// Mean and `sqrt(sample variance / n)` of `n` values from the sums of `v - k` and `(v - k)²`.
fn mean_and_err(k: f64, sum: f64, sum_sq: f64, n: usize) -> McPrice {
    let nf = n as f64;
    let std_err = if n < 2 { f64::NAN } else { ((sum_sq - sum * sum / nf).max(0.0) / ((nf - 1.0) * nf)).sqrt() };
    McPrice { price: k + sum / nf, std_err }
}

/// Price only (path mean of the payoff) and its standard error: no residuals saved, no
/// reverse sweep. Same arguments, normals layout and panics as [`mc_greeks`].
pub fn mc_price<M: McModel, R: Real>(params: &M::P<R>, n_steps: usize, n_paths: usize, normals: &[f32]) -> McPrice {
    let w = R::Lanes::W;
    let nz = <M::Z<R> as Arr<R>>::LEN;
    let stride = path_stride(n_steps, n_paths, nz, w, normals.len());
    let c = M::setup_eval(params, R::from(n_steps as f64));
    let s0 = M::init_eval(params);
    let k = shift::<M, R>(params, &c, &s0, &normals[..stride]);
    let (mut sum, mut sum_sq) = (0.0, 0.0);
    for block in normals[..n_paths * n_steps * nz].chunks(FLUSH_TERMS.div_ceil(n_steps) * stride) {
        let (mut acc, mut acc_sq) = (R::Lanes::neg_zero(), R::Lanes::neg_zero());
        for path in block.chunks_exact(stride) {
            let mut s = s0;
            for zk in path.chunks_exact(nz * w) {
                s = M::step_eval(&s, &c, &load_z::<M, R>(zk));
            }
            let d = M::payoff_eval(&s, params, &c).value() - k;
            acc += d;
            acc_sq = d.mul_add(d, acc_sq);
        }
        sum += acc.sum();
        sum_sq += acc_sq.sum();
    }
    mean_and_err(k.lane(0), sum, sum_sq, n_paths)
}
