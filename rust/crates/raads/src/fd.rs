//! 1-D finite-difference engine (theta scheme on x = log S): price and gradient.
//!
//! PDE in time to maturity τ: `V_τ = a(x) V_xx + b(x) V_x - c(x) V`, `V(x, 0) = payoff(x)`.
//! A model is small kernels written once against [`AdNum`](crate::AdNum) inside an `impl`
//! block marked [`#[fd_model]`](crate::fd_model):
//!
//! ```text
//! coeffs(#[data] x, p) -> [a, b, c]   per grid node
//! payoff(#[data] x, p) -> V           per grid node (also the exercise value if AMERICAN)
//! spot(p)              -> x0          today's log-spot (the price is interpolated there)
//! maturity(p)          -> T
//! ```
//!
//! The engine owns the grid, time stepping, tridiagonal solves, early exercise and a
//! hand-written reverse sweep (transposed solves); only the kernels' small vector-Jacobian
//! products run in the reverse sweep. Boundaries: `V_xx = 0` with one-sided `V_x`. The
//! initial condition is the payoff averaged over each cell (4-point Gauss), which smooths
//! the strike kink; early exercise compares with the pointwise payoff. The grid is fixed
//! (not differentiated); the price and spot sensitivity come from quadratic interpolation.
//! Lanes are independent problems (e.g. a strike ladder) on a shared grid.

use crate::{Arr, Lanes, Real};
use std::marker::PhantomData;

/// A model compiled by [`#[fd_model]`](crate::fd_model). `P` holds the parameters.
///
/// Generated `*_eval` functions compute a kernel; `*_vjp` add the adjoints of its inputs
/// (seeded with `a_y`) into `a_p` and return the adjoint of the data input `x` (if any).
pub trait FdModel {
    /// Parameters.
    type P<T: Copy>: Arr<T>;
    /// Early exercise: `V = max(V, payoff)` after every step.
    const AMERICAN: bool = false;
    /// Width of the smoothed exercise max (0 = hard max). The hard max makes the discrete
    /// price only piecewise smooth in parameters that move the exercise boundary, so
    /// second-order Greeks need `EXERCISE_EPS > 0`; first order does not.
    const EXERCISE_EPS: f64 = 0.0;

    /// `coeffs` kernel: `[a, b, c]` at log-spot `x`.
    fn coeffs_eval<R: Real>(x: R, p: &Self::P<R>) -> [R; 3];
    /// `coeffs` reverse.
    fn coeffs_vjp<R: Real>(x: R, p: &Self::P<R>, a_y: &[R; 3], a_p: &mut Self::P<R>) -> [R; 3];
    /// `payoff` kernel.
    fn payoff_eval<R: Real>(x: R, p: &Self::P<R>) -> R;
    /// `payoff` reverse.
    fn payoff_vjp<R: Real>(x: R, p: &Self::P<R>, a_y: R, a_p: &mut Self::P<R>) -> R;
    /// `spot` kernel (log-spot).
    fn spot_eval<R: Real>(p: &Self::P<R>) -> R;
    /// `spot` reverse.
    fn spot_vjp<R: Real>(p: &Self::P<R>, a_y: R, a_p: &mut Self::P<R>) -> R;
    /// `maturity` kernel.
    fn maturity_eval<R: Real>(p: &Self::P<R>) -> R;
    /// `maturity` reverse.
    fn maturity_vjp<R: Real>(p: &Self::P<R>, a_y: R, a_p: &mut Self::P<R>) -> R;
}

/// Uniform grid on `[x_lo, x_hi]` with `nx` nodes, `nt` time steps, theta scheme
/// (`theta = 0.5`: Crank–Nicolson).
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub struct FdGrid {
    /// Lowest log-spot.
    pub x_lo: f64,
    /// Highest log-spot.
    pub x_hi: f64,
    /// Space nodes.
    pub nx: usize,
    /// Time steps.
    pub nt: usize,
    /// Implicitness: 0.5 Crank–Nicolson, 1 fully implicit.
    pub theta: f64,
}

impl FdGrid {
    /// Crank–Nicolson grid.
    pub fn new(x_lo: f64, x_hi: f64, nx: usize, nt: usize) -> Self {
        FdGrid { x_lo, x_hi, nx, nt, theta: 0.5 }
    }
}

/// Price and gradient per lane.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
#[must_use]
pub struct FdResult<R, G> {
    /// Price.
    pub price: R,
    /// d price / d params.
    pub grad: G,
}

/// 4-point Gauss–Legendre on [-1/2, 1/2] (cell averaging).
const GAUSS_X: [f64; 4] = [-0.4305681557970263, -0.16999052179242813, 0.16999052179242813, 0.4305681557970263];
const GAUSS_W: [f64; 4] = [0.17392742256872692, 0.32607257743127307, 0.32607257743127307, 0.17392742256872692];

/// `max(y, g)`, or its softplus smoothing `g + eps·ln(1 + exp((y - g)/eps))` when `eps > 0`.
#[inline(always)]
pub(crate) fn exercise<R: Real>(y: R, g: R, eps: f64) -> R {
    if eps == 0.0 {
        R::select_gt(y, g, y, g)
    } else {
        let d = y - g;
        let ad = R::select_gt(d, R::from(0.0), d, -d);
        g + d.relu() + (-ad / eps).exp().radd(1.0).ln() * eps
    }
}

/// `d max(y, g) / dy` (the derivative w.r.t. `g` is one minus this).
#[inline(always)]
pub(crate) fn exercise_weight<R: Real>(y: R, g: R, eps: f64) -> R {
    if eps == 0.0 {
        R::select_gt(y, g, R::from(1.0), R::from(0.0))
    } else {
        let d = y - g;
        let ad = R::select_gt(d, R::from(0.0), d, -d);
        let e = (-ad / eps).exp();
        R::select_gt(d, R::from(0.0), (e + 1.0).rdiv(1.0), e / (e + 1.0))
    }
}

/// One theta-scheme solve on a fixed grid: operator `L = (l, dd, u)`, exercise values,
/// cell-averaged payoff, per-block `E` and LU factors of `M`, and the reverse-sweep buffers.
/// [`fd_greeks`] (one maturity) and the surface engine (many) drive the same steps.
///
/// Per step (level n → n+1): `rhs = E V`, `M Y = rhs` (unit-lower L, then U with one FMA on
/// the chain), `V = max(Y, payoff)` if American. Only Y is stored. Reverse per step:
/// `Uᵀ w = mask·aV`, `Lᵀ a_rhs = w` fused with the operator adjoints
/// `aL_ij += a_rhs_i (θ Y^{n+1}_j + (1-θ) V^n_j)` (without the dt factor; see `rescale`)
/// and `aV^n = Eᵀ a_rhs`.
pub(crate) struct Scheme<M, R> {
    pub nx: usize,
    pub x_lo: f64,
    pub h: f64,
    pub l: Vec<R>,
    pub dd: Vec<R>,
    pub u: Vec<R>,
    pub intr: Vec<R>,
    pub cur: Vec<R>,
    es: Vec<R>,
    ed: Vec<R>,
    eu: Vec<R>,
    inv_m: Vec<R>,
    cpr: Vec<R>,
    lmul: Vec<R>,
    q_t: Vec<R>,
    dp: Vec<R>,
    pub a_v: Vec<R>,
    al: Vec<R>,
    ad: Vec<R>,
    au: Vec<R>,
    aintr: Vec<R>,
    wv: Vec<R>,
    model: PhantomData<M>,
}

impl<M: FdModel, R: Real> Scheme<M, R> {
    pub fn new(p: &M::P<R>, x_lo: f64, x_hi: f64, nx: usize, grads: bool) -> Self {
        assert!(nx >= 3, "need at least 3 grid nodes");
        let h = (x_hi - x_lo) / (nx - 1) as f64;
        let zero = R::from(0.0);
        let buf = || vec![zero; nx];
        let abuf = || vec![zero; if grads { nx } else { 0 }];
        let mut s = Scheme {
            nx,
            x_lo,
            h,
            l: buf(),
            dd: buf(),
            u: buf(),
            intr: buf(),
            cur: buf(),
            es: buf(),
            ed: buf(),
            eu: buf(),
            inv_m: buf(),
            cpr: buf(),
            lmul: buf(),
            q_t: buf(),
            dp: buf(),
            a_v: abuf(),
            al: abuf(),
            ad: abuf(),
            au: abuf(),
            aintr: abuf(),
            wv: abuf(),
            model: PhantomData,
        };
        for i in 0..nx {
            let xi = x_lo + i as f64 * h;
            let [a, b, c] = M::coeffs_eval(R::from(xi), p);
            (s.l[i], s.dd[i], s.u[i]) = if i == 0 {
                (zero, -b / h - c, b / h)
            } else if i == nx - 1 {
                (-b / h, b / h - c, zero)
            } else {
                (a / (h * h) - b / (2.0 * h), a * (-2.0 / (h * h)) - c, a / (h * h) + b / (2.0 * h))
            };
            s.intr[i] = M::payoff_eval(R::from(xi), p);
            s.cur[i] = GAUSS_X.iter().zip(GAUSS_W).fold(zero, |acc, (gx, gw)| acc + M::payoff_eval(R::from(xi + gx * h), p) * gw);
        }
        s
    }

    /// `E = I + (1-θ) dt L` and the LU factors of `M = I - θ dt L` (no divisions in the steps).
    pub fn factor(&mut self, dt: R, th: f64) {
        let Scheme { l, dd, u, es, ed, eu, inv_m, cpr, lmul, q_t, .. } = self;
        let ew = dt * (1.0 - th);
        let iw = dt * th;
        for i in 0..l.len() {
            es[i] = ew * l[i];
            ed[i] = ew.mul_add(dd[i], R::from(1.0));
            eu[i] = ew * u[i];
        }
        inv_m[0] = iw.nmul_add(dd[0], R::from(1.0)).rdiv(1.0);
        cpr[0] = -(iw * u[0]) * inv_m[0];
        lmul[0] = R::from(0.0);
        q_t[0] = R::from(0.0);
        for i in 1..l.len() {
            let mprev = -(iw * u[i - 1]); // superdiagonal of M, row i-1
            lmul[i] = -(iw * l[i]) * inv_m[i - 1];
            let piv = lmul[i].nmul_add(mprev, iw.nmul_add(dd[i], R::from(1.0)));
            inv_m[i] = piv.rdiv(1.0);
            cpr[i] = -(iw * u[i]) * inv_m[i];
            q_t[i] = mprev * inv_m[i];
        }
    }

    /// One step; with `STORE`, Y (pre-exercise) goes to `ys[..nx]`.
    #[inline(always)]
    pub fn step<const STORE: bool>(&mut self, ys: &mut [R]) {
        let nx = self.nx;
        let Scheme { cur, dp, intr, es, ed, eu, inv_m, cpr, lmul, .. } = self;
        // equal-length slices: indices below nx need no bounds checks
        let (cur, dp, intr) = (&mut cur[..nx], &mut dp[..nx], &intr[..nx]);
        let (es, ed, eu, inv_m, cpr, lmul) = (&es[..nx], &ed[..nx], &eu[..nx], &inv_m[..nx], &cpr[..nx], &lmul[..nx]);
        let ys = if STORE { &mut ys[..nx] } else { ys };
        let zero = R::from(0.0);
        let (mut vprev, mut vcur, mut dprev) = (zero, cur[0], zero);
        for i in 0..nx {
            let vnext = if i < nx - 1 { cur[i + 1] } else { zero };
            let rhs = es[i].mul_add(vprev, ed[i].mul_add(vcur, eu[i] * vnext));
            let d = lmul[i].nmul_add(dprev, rhs);
            dp[i] = d * inv_m[i];
            dprev = d;
            vprev = vcur;
            vcur = vnext;
        }
        let mut ynext = zero;
        for i in (0..nx).rev() {
            let y = cpr[i].nmul_add(ynext, dp[i]);
            if STORE {
                ys[i] = y;
            }
            cur[i] = if M::AMERICAN { exercise(y, intr[i], M::EXERCISE_EPS) } else { y };
            ynext = y;
        }
    }

    /// Reverse of step `n` (level n → n+1): `aV` (level n+1) → `aV` (level n).
    /// `ys` holds the stored levels, `nx` values each.
    #[inline(always)]
    pub fn step_back(&mut self, ys: &[R], n: usize, th: f64) {
        let nx = self.nx;
        let Scheme { a_v, wv, intr, aintr, al, ad, au, es, ed, eu, inv_m, lmul, q_t, .. } = self;
        // equal-length slices: indices below nx need no bounds checks
        let (a_v, wv, aintr) = (&mut a_v[..nx], &mut wv[..nx], &mut aintr[..nx]);
        let (al, ad, au) = (&mut al[..nx], &mut ad[..nx], &mut au[..nx]);
        let (intr, es, ed, eu, inv_m, lmul, q_t) = (&intr[..nx], &es[..nx], &ed[..nx], &eu[..nx], &inv_m[..nx], &lmul[..nx], &q_t[..nx]);
        let zero = R::from(0.0);
        let y1 = &ys[(n + 1) * nx..(n + 2) * nx];
        let y0 = &ys[n * nx..(n + 1) * nx];
        // Uᵀ w = aY (forward), with the exercise mask: w_i = aY_i/m_i - qT_i w_{i-1}
        let mut wprev = zero;
        for i in 0..nx {
            let mut ay = a_v[i];
            if M::AMERICAN {
                let wy = exercise_weight(y1[i], intr[i], M::EXERCISE_EPS);
                aintr[i] = ay.mul_add(wy.rsub(1.0), aintr[i]);
                ay = ay * wy;
            }
            let w = q_t[i].nmul_add(wprev, ay * inv_m[i]);
            wv[i] = w;
            wprev = w;
        }
        let zval = |j: usize| -> R {
            // level 0 is the averaged payoff itself (no exercise applied)
            let vj = if M::AMERICAN && n > 0 { exercise(y0[j], intr[j], M::EXERCISE_EPS) } else { y0[j] };
            y1[j].mul_add(R::from(th), vj * (1.0 - th))
        };
        // Lᵀ a_rhs = w (backward): r_i = w_i - lmul_{i+1} r_{i+1}; fused operator adjoints and
        // aV_{i+1} = ed_{i+1} r_{i+1} + eu_i r_i + es_{i+2} r_{i+2} (lagged by one node)
        let (mut r1, mut r2, mut zp) = (zero, zero, zero);
        let mut zc = zval(nx - 1);
        for i in (0..nx).rev() {
            let lm = if i < nx - 1 { lmul[i + 1] } else { zero };
            let r = lm.nmul_add(r1, wv[i]);
            let zm = if i > 0 { zval(i - 1) } else { zero };
            al[i] = r.mul_add(zm, al[i]);
            ad[i] = r.mul_add(zc, ad[i]);
            au[i] = r.mul_add(zp, au[i]);
            zp = zc;
            zc = zm;
            if i < nx - 1 {
                let esn = if i < nx - 2 { es[i + 2] } else { zero };
                a_v[i + 1] = ed[i + 1].mul_add(r1, eu[i].mul_add(r, esn * r2));
            }
            r2 = r1;
            r1 = r;
        }
        a_v[0] = ed[0].mul_add(r1, es[1] * r2);
    }

    /// `Σ_i (al l + ad dd + au u)`: the dt adjoint per unit dt, before `rescale`.
    pub fn operator_dot(&self) -> R {
        (0..self.nx)
            .fold(R::from(0.0), |s, i| self.al[i].mul_add(self.l[i], self.ad[i].mul_add(self.dd[i], self.au[i].mul_add(self.u[i], s))))
    }

    /// Multiply the accumulated operator adjoints by `f`. One block: `f = dt` at the end.
    /// Several blocks (reverse order): `f = dt_b / dt_{b-1}` between blocks and `dt_first` at
    /// the end, so that `Σ_b dt_b A_b = dt_0 (A_0 + (dt_1/dt_0)(A_1 + …))`.
    pub fn rescale(&mut self, f: R) {
        for a in self.al.iter_mut().chain(self.ad.iter_mut()).chain(self.au.iter_mut()) {
            *a = *a * f;
        }
    }

    /// Payoff (initial and exercise values) and coefficient vjps into `a_p`.
    pub fn to_params(&self, p: &M::P<R>, a_p: &mut M::P<R>) {
        let h = self.h;
        for i in 0..self.nx {
            let xi = self.x_lo + i as f64 * h;
            for (gx, gw) in GAUSS_X.iter().zip(GAUSS_W) {
                M::payoff_vjp(R::from(xi + gx * h), p, self.a_v[i] * gw, a_p);
            }
            if M::AMERICAN {
                M::payoff_vjp(R::from(xi), p, self.aintr[i], a_p);
            }
            let (al, ad, au) = (self.al[i], self.ad[i], self.au[i]);
            let zero = R::from(0.0);
            let ak = if i == 0 {
                [zero, (au - ad) / h, -ad]
            } else if i == self.nx - 1 {
                [zero, (ad - al) / h, -ad]
            } else {
                [(al - ad * 2.0 + au) / (h * h), (au - al) / (2.0 * h), -ad]
            };
            M::coeffs_vjp(R::from(xi), p, &ak, a_p);
        }
    }
}

/// `nt` equal steps to the model's maturity, price interpolated at the model's spot.
fn fd_core<M: FdModel, R: Real, const GRADS: bool>(p: &M::P<R>, g: &FdGrid) -> FdResult<R, M::P<R>> {
    let (x_lo, nx, nt) = (g.x_lo, g.nx, g.nt);
    assert!(nt >= 1, "need at least 1 time step");
    let x0 = M::spot_eval(p);
    let dt = M::maturity_eval(p) / nt as f64;
    let mut sc = Scheme::<M, R>::new(p, x_lo, g.x_hi, nx, GRADS);
    let h = sc.h;
    sc.factor(dt, g.theta);
    let mut ys = vec![R::from(0.0); if GRADS { (nt + 1) * nx } else { 0 }];
    if GRADS {
        ys[..nx].copy_from_slice(&sc.cur);
    }
    for n in 0..nt {
        if GRADS {
            sc.step::<true>(&mut ys[(n + 1) * nx..(n + 2) * nx]);
        } else {
            sc.step::<false>(&mut []);
        }
    }

    // price: quadratic interpolation at x0 around the nearest node (chosen per lane by value)
    let w = R::Lanes::W;
    let x0v = x0.value();
    let jl: Vec<usize> = (0..w).map(|l| (((x0v.lane(l) - x_lo) / h + 0.5) as i64).clamp(1, nx as i64 - 2) as usize).collect();
    let xj = R::Lanes::from_fn(|l| x_lo + jl[l] as f64 * h);
    let cur = &sc.cur;
    let vm = R::per_lane(|l| cur[jl[l] - 1]);
    let vc = R::per_lane(|l| cur[jl[l]]);
    let vp = R::per_lane(|l| cur[jl[l] + 1]);
    let t = (x0 - R::lift(xj)) / h;
    let (w0, w1, w2) = (t * (t - 1.0) * 0.5, (t * t).rsub(1.0), t * (t + 1.0) * 0.5);
    let price = w0 * vm + w1 * vc + w2 * vp;

    let mut a_p: M::P<R> = Arr::splat(R::from(0.0));
    if GRADS {
        let a_x0 = ((t - 0.5) * vm - t * 2.0 * vc + (t + 0.5) * vp) / h;
        for (lane, &j) in jl.iter().enumerate() {
            for (k, wk) in [(j - 1, w0), (j, w1), (j + 1, w2)] {
                let q = sc.a_v[k];
                let upd = q + wk;
                sc.a_v[k] = R::per_lane(|m| if m == lane { upd } else { q });
            }
        }
        for n in (0..nt).rev() {
            sc.step_back(&ys, n, g.theta);
        }
        let a_dt = sc.operator_dot();
        sc.rescale(dt);
        sc.to_params(p, &mut a_p);
        M::spot_vjp(p, a_x0, &mut a_p);
        M::maturity_vjp(p, a_dt / nt as f64, &mut a_p);
    }
    FdResult { price, grad: a_p }
}

/// Price and gradient per lane, on a shared grid (lanes = independent parameter sets).
/// With [`Dual`](crate::Dual) parameters seeded with a direction v, the tangents are
/// grad · v and H v.
///
/// # Panics
///
/// If `grid.nx < 3` or `grid.nt == 0`.
pub fn fd_greeks<M: FdModel, R: Real>(params: &M::P<R>, grid: &FdGrid) -> FdResult<R, M::P<R>> {
    fd_core::<M, R, true>(params, grid)
}

/// Price only: no history stored, no reverse sweep.
///
/// # Panics
///
/// If `grid.nx < 3` or `grid.nt == 0`.
#[must_use]
pub fn fd_price<M: FdModel, R: Real>(params: &M::P<R>, grid: &FdGrid) -> R {
    fd_core::<M, R, false>(params, grid).price
}
