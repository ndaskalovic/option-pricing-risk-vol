//! FD surface engine: one solve gives the solution at every maturity and, for a unit-strike
//! model on x = log(S/K), every strike.
//!
//! Same PDE, operator and scheme as [`fd`](crate::fd) (theta scheme, LU factorised per time
//! block, cell-averaged payoff, exercise after each solve, `V_xx = 0` boundaries).
//! Differences:
//! - The time grid is a list of blocks so every maturity lands exactly on a step. The first
//!   block starts with Rannacher steps (implicit half-steps) that damp the payoff kink, which
//!   matters for short maturities.
//! - The solution u and its time derivative `u_τ = L u` (0 where exercised) are saved at
//!   every maturity instead of being interpolated once at the spot. The model's `spot` and
//!   `maturity` are unused; the time grid is data, not differentiated.
//!
//! With [`Dual`](crate::Dual) parameters one pass gives values and one forward tangent per
//! lane at every node (e.g. vega in one lane, rho in another). [`fd_surface_book`] adds a
//! reverse sweep for a weighted sum of surface values (a book): its gradient w.r.t. every
//! model parameter in one sweep. For a unit-strike model `V(S, K, T) = K u(log(S/K), T)`;
//! [`FdSurface::option`] maps a read to price and grid Greeks.

use crate::fd::{FdModel, Scheme};
use crate::{AdNum, Arr, Lanes, Real};

/// Spatial grid and time stepping of a surface solve.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub struct SurfaceGrid {
    /// Lowest grid coordinate.
    pub x_lo: f64,
    /// Highest grid coordinate.
    pub x_hi: f64,
    /// Space nodes.
    pub nx: usize,
    /// Largest time step; each maturity segment gets equal steps no longer than this.
    pub dt: f64,
    /// Crank–Nicolson steps replaced by two implicit half-steps each at the start.
    pub rannacher: usize,
}

impl SurfaceGrid {
    /// With two Rannacher steps.
    pub fn new(x_lo: f64, x_hi: f64, nx: usize, dt: f64) -> Self {
        SurfaceGrid { x_lo, x_hi, nx, dt, rannacher: 2 }
    }
}

#[derive(Clone, Copy, Debug)]
struct Block {
    dt: f64,
    th: f64,
    steps: usize,
    /// Global time level at the start of the block.
    level0: usize,
    /// Maturity index reached at the end of the block.
    mat: Option<usize>,
}

/// Each maturity segment gets `ceil(len / dt)` equal steps (at least `rannacher + 1`); the
/// first `rannacher` Crank–Nicolson steps become `2 * rannacher` implicit half-steps.
fn schedule(maturities: &[f64], dt_target: f64, rannacher: usize) -> Vec<Block> {
    let mut bl = Vec::new();
    let (mut t_prev, mut lev) = (0.0, 0);
    for (m, &t) in maturities.iter().enumerate() {
        let seg = t - t_prev;
        assert!(seg > 0.0, "maturities must be positive and strictly increasing");
        let ns = (rannacher + 1).max((seg / dt_target).ceil() as usize);
        let dt = seg / ns as f64;
        let mut start = 0;
        if m == 0 && rannacher > 0 {
            bl.push(Block { dt: dt * 0.5, th: 1.0, steps: 2 * rannacher, level0: lev, mat: None });
            lev += 2 * rannacher;
            start = rannacher;
        }
        bl.push(Block { dt, th: 0.5, steps: ns - start, level0: lev, mat: Some(m) });
        lev += ns - start;
        t_prev = t;
    }
    bl
}

/// A weighted read of the surface, per lane.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SurfacePoint<L> {
    /// Maturity index.
    pub maturity: usize,
    /// Grid coordinate.
    pub x: L,
    /// Weight in the book.
    pub weight: L,
}

/// `u(x, T_m)` and `u_τ(x, T_m)` on the grid for every maturity `m`.
#[derive(Clone, Debug)]
#[must_use]
pub struct FdSurface<R> {
    x_lo: f64,
    h: f64,
    nx: usize,
    slices: Vec<R>,
    /// `u_τ = a u_xx + b u_x - c u` (discrete `L u`); 0 on exercised nodes.
    taus: Vec<R>,
}

impl<R: Real> FdSurface<R> {
    /// Lowest grid coordinate.
    pub fn x_lo(&self) -> f64 {
        self.x_lo
    }

    /// Node spacing.
    pub fn h(&self) -> f64 {
        self.h
    }

    /// Space nodes.
    pub fn nx(&self) -> usize {
        self.nx
    }

    /// Nearest interior node per lane (at least two nodes from either end) and the offset
    /// `t = (x - x_j) / h`.
    fn weights(&self, x: R::Lanes) -> (Vec<usize>, R::Lanes) {
        let j: Vec<usize> = (0..R::Lanes::W)
            .map(|l| (((x.lane(l) - self.x_lo) / self.h + 0.5) as i64).clamp(2, self.nx as i64 - 3) as usize)
            .collect();
        let xj = R::Lanes::from_fn(|l| self.x_lo + j[l] as f64 * self.h);
        (j, (x - xj) / self.h)
    }

    /// u (quadratic through the nearest 3 nodes), u_x (its slope) and u_xx (second differences
    /// at the nearest node and its neighbour on x's side, linearly interpolated: O(h²)).
    fn read_buf(&self, buf: &[R], m: usize, x: R::Lanes) -> [R; 3] {
        let (j, ts) = self.weights(x);
        let t = R::lift(ts);
        let at = R::lift(R::Lanes::select_gt(ts, R::Lanes::from(0.0), ts, -ts));
        let s = &buf[m * self.nx..(m + 1) * self.nx];
        let side = |l: usize| if ts.lane(l) >= 0.0 { 1 } else { -1 };
        let at_off = |off: isize| R::per_lane(|l| s[(j[l] as isize + off) as usize]);
        let (vm, vc, vp) = (at_off(-1), at_off(0), at_off(1));
        let vi = R::per_lane(|l| s[(j[l] as isize + side(l)) as usize]); // inner neighbour on x's side
        let vo = R::per_lane(|l| s[(j[l] as isize + 2 * side(l)) as usize]); // outer node on x's side
        let h2 = self.h * self.h;
        let d2c = (vm - vc * 2.0 + vp) / h2;
        let d2n = (vc - vi * 2.0 + vo) / h2;
        [
            t * (t - 1.0) * 0.5 * vm + (t * t).rsub(1.0) * vc + t * (t + 1.0) * 0.5 * vp,
            ((t - 0.5) * vm - t * 2.0 * vc + (t + 0.5) * vp) / self.h,
            at.rsub(1.0) * d2c + at * d2n,
        ]
    }

    /// u, u_x, u_xx at maturity index `m`, per-lane coordinate `x`. Panics if `m` is out of range.
    pub fn read(&self, m: usize, x: R::Lanes) -> [R; 3] {
        self.read_buf(&self.slices, m, x)
    }

    /// Unit-strike model on x = log(S/K): price, delta, gamma, theta (dV/dt, calendar time),
    /// dV/dK. Theta comes from the PDE (`u_τ = L u` on continuation nodes): no dt error.
    /// Panics if `m` is out of range.
    pub fn option(&self, m: usize, spot: R::Lanes, strike: R::Lanes) -> [R; 5] {
        let x = (spot / strike).ln();
        let r = self.read_buf(&self.slices, m, x);
        let rt = self.read_buf(&self.taus, m, x);
        let (k, s) = (R::lift(strike), R::lift(spot));
        [k * r[0], k * r[1] / s, k * (r[2] - r[1]) / (s * s), -(k * rt[0]), r[0] - r[1]]
    }
}

/// Forward solve saving every maturity; with `GRADS`, also the reverse sweep of
/// `Σ_k weight_k u(x_k, T_{maturity_k})` into `a_p`.
fn surface_core<M: FdModel, R: Real, const GRADS: bool>(
    p: &M::P<R>,
    g: &SurfaceGrid,
    maturities: &[f64],
    points: &[SurfacePoint<R::Lanes>],
    a_p: &mut M::P<R>,
) -> FdSurface<R> {
    let nx = g.nx;
    assert!(nx >= 5, "need at least 5 grid nodes");
    assert!(!maturities.is_empty(), "need at least 1 maturity");
    assert!(points.iter().all(|pt| pt.maturity < maturities.len()), "SurfacePoint::maturity out of range");
    let mut sc = Scheme::<M, R>::new(p, g.x_lo, g.x_hi, nx, GRADS);
    let bl = schedule(maturities, g.dt, g.rannacher);
    let last = bl[bl.len() - 1];
    let mut ys = vec![R::from(0.0); if GRADS { (last.level0 + last.steps + 1) * nx } else { 0 }];
    let zero = R::from(0.0);
    let mut surf = FdSurface { x_lo: g.x_lo, h: sc.h, nx, slices: vec![zero; maturities.len() * nx], taus: vec![zero; maturities.len() * nx] };
    if GRADS {
        ys[..nx].copy_from_slice(&sc.cur);
    }

    // forward: per block factorise once, step, save u and u_τ at each maturity
    for b in &bl {
        sc.factor(R::from(b.dt), b.th);
        for s in 0..b.steps {
            if GRADS {
                let lev = b.level0 + s + 1;
                sc.step::<true>(&mut ys[lev * nx..(lev + 1) * nx]);
            } else {
                sc.step::<false>(&mut []);
            }
        }
        if let Some(m) = b.mat {
            let (cur, l, dd, u, intr) = (&sc.cur, &sc.l, &sc.dd, &sc.u, &sc.intr);
            surf.slices[m * nx..(m + 1) * nx].copy_from_slice(cur);
            for i in 0..nx {
                let c = cur[i];
                let mut lu = dd[i] * c;
                if i > 0 {
                    lu = l[i].mul_add(cur[i - 1], lu);
                }
                if i < nx - 1 {
                    lu = u[i].mul_add(cur[i + 1], lu);
                }
                surf.taus[m * nx + i] = if M::AMERICAN { R::select_gt(c, intr[i], lu, zero) } else { lu };
            }
        }
    }

    if GRADS {
        for bi in (0..bl.len()).rev() {
            let b = bl[bi];
            if bi + 1 < bl.len() {
                sc.rescale(R::from(bl[bi + 1].dt / b.dt)); // see `Scheme::rescale`
            }
            // seed the book's reads of this maturity (aV is the adjoint of V at the block's last level)
            if let Some(m) = b.mat {
                for pt in points.iter().filter(|pt| pt.maturity == m) {
                    let (j, t) = surf.weights(pt.x);
                    let w = [t * (t - 1.0) * 0.5 * pt.weight, (t * t).rsub(1.0) * pt.weight, t * (t + 1.0) * 0.5 * pt.weight];
                    for (lane, &jl) in j.iter().enumerate() {
                        for (k, wk) in w.iter().enumerate() {
                            let node = jl + k - 1;
                            let q = sc.a_v[node];
                            let upd = q + R::lift(R::Lanes::from(wk.lane(lane)));
                            sc.a_v[node] = R::per_lane(|l| if l == lane { upd } else { q });
                        }
                    }
                }
            }
            sc.factor(R::from(b.dt), b.th);
            for s in (0..b.steps).rev() {
                sc.step_back(&ys, b.level0 + s, b.th);
            }
        }
        sc.rescale(R::from(bl[0].dt));
        sc.to_params(p, a_p);
    }
    surf
}

/// Solution at every maturity in one solve.
///
/// # Panics
///
/// If `grid.nx < 5`, `maturities` is empty or not positive and strictly increasing.
pub fn fd_surface<M: FdModel, R: Real>(params: &M::P<R>, grid: &SurfaceGrid, maturities: &[f64]) -> FdSurface<R> {
    let mut a: M::P<R> = Arr::splat(R::from(0.0));
    surface_core::<M, R, false>(params, grid, maturities, &[], &mut a)
}

/// A book of surface reads and its gradient.
#[derive(Clone, Debug)]
#[non_exhaustive]
#[must_use]
pub struct FdBook<R, G> {
    /// `Σ_k weight_k u(x_k, T_{maturity_k})`, per lane.
    pub value: R,
    /// d value / d params, per lane.
    pub grad: G,
    /// The solution the book reads.
    pub surface: FdSurface<R>,
}

/// A book of surface reads `Σ_k weight_k u(x_k, T_{maturity_k})` and its gradient w.r.t. every model
/// parameter in one reverse sweep (per lane). For a unit-strike model `weight_k = position_k · K_k`.
/// With [`Dual`](crate::Dual) the gradient tangents are Hessian-vector products of the book.
///
/// # Panics
///
/// As [`fd_surface`], or if a point's maturity index is out of range.
pub fn fd_surface_book<M: FdModel, R: Real>(
    params: &M::P<R>,
    grid: &SurfaceGrid,
    maturities: &[f64],
    points: &[SurfacePoint<R::Lanes>],
) -> FdBook<R, M::P<R>> {
    let mut grad: M::P<R> = Arr::splat(R::from(0.0));
    let surface = surface_core::<M, R, true>(params, grid, maturities, points, &mut grad);
    let value = points.iter().fold(R::from(0.0), |v, pt| v + R::lift(pt.weight) * surface.read(pt.maturity, pt.x)[0]);
    FdBook { value, grad, surface }
}
