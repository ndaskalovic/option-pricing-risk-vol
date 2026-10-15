# Option Pricing, Risk and Volatility Modelling

Adjoint algorithmic differentiation (AAD) for option Greeks, written in [Mojo](https://www.modular.com/mojo). It started as part of my master's thesis on AAD for Monte Carlo Greeks under Black-Scholes [2, 3]. It has since grown into a small library with two ways to differentiate a pricer:

- **A runtime tape** (`autodiff/`) records every operation on `Number` values and runs the chain rule backwards. It handles any code, and uses SIMD lanes as Monte Carlo paths and pathwise adjoints with mark/rewind [2].
- **Compile-time AAD** (`autodiff/ct/`) is the main engine. You write small, branch-free model functions once, generic over a number type. The Mojo compiler traces each one into a graph *at compile time*, analyses it, and emits straight-line forward and reverse code: no tape, no runtime interpretation. Monte Carlo and finite-difference engines run these kernels.

Like Polars, the trade-off is a fixed set of supported operations in exchange for optimised execution.

## Writing a model

A Monte Carlo model is four small functions (`models/gbm.mojo`):

```mojo
struct GbmCall(McModel):
    """params: s0, r, sigma, T, K.   coeffs: drift, vol*sqrt(dt).   state: log S."""
    comptime NP = 5; comptime NC = 2; comptime NS = 1; comptime NZ = 1

    @staticmethod
    def setup[N: ADNum](p: Array[N, 5], n_steps: N) -> Array[N, 2]:      # once per run
        var dt = p[3] / n_steps
        var c = Array[N, 2](fill=dt)
        c[0] = (p[1] - 0.5 * p[2] * p[2]) * dt
        c[1] = p[2] * dt.sqrt()
        return c^

    @staticmethod
    def init[N: ADNum](p: Array[N, 5]) -> Array[N, 1]:                    # initial state
        return Array[N, 1](fill=p[0].log())

    @staticmethod
    def step[N: ADNum](s: Array[N, 1], c: Array[N, 2], z: Array[N, 1]) -> Array[N, 1]:   # each time step
        return Array[N, 1](fill=s[0] + c[0] + c[1] * z[0])

    @staticmethod
    def payoff[N: ADNum](s: Array[N, 1], p: Array[N, 5], c: Array[N, 2]) -> N:          # each path
        return (s[0].exp() - p[4]).relu() * (-p[1] * p[3]).exp()

var r = mc_compiled[GbmCall, 32, DType.float32](params, n_steps, n_paths, normals)
print(r.price, r.grad[0], r.grad[2])      # price, delta, vega; grad has one entry per parameter
```

A finite-difference model supplies the PDE coefficients, the payoff, the spot and the maturity (`models/bs_fd.mojo`). The engine owns the grid, the Crank–Nicolson stepping, the tridiagonal solves, early exercise and the reverse sweep:

```mojo
struct BsAmericanPut(FdModel):
    comptime NP = 5
    comptime AMERICAN = True
    @staticmethod
    def coeffs[N: ADNum](x: N, p: Array[N, 5]) -> Array[N, 3]: ...   # V_τ = a V_xx + b V_x − c V,  x = log S
    @staticmethod
    def payoff[N: ADNum](x: N, p: Array[N, 5]) -> N:
        return (p[4] - x.exp()).relu()
    ...

var res = fd_compiled[BsAmericanPut, 1, DType.float64](params, x_lo, x_hi, nx=201, nt=200)
var ladder = fd_batch[BsAmericanPut, 4, DType.float64](per_lane_params, x_lo, x_hi, 201, 200)  # 4 options, one solve
```

**Supported operations:**
- `+ - * /` (with other values or `Float64`), negation, `**k`, `exp`, `log`, `sqrt`, `relu`;
- `select_gt` / `maximum` / `minimum` / `absolute`, and `smooth_relu`;
- `detach` (stop-gradient) and `constant`.

**Rules:**
- No `if` on values: use `select_gt`.
- Loops inside a kernel need counts known at compile time. The engines own the runtime loops over paths, steps and the grid.

The same model source also runs on the runtime tape (`mc_tape`, `fd_tape`). That gives a flexible fallback, and an independent oracle for the tests.

## Performance

All numbers: Apple M5, single core, best of 11–31 runs, price plus every first-order Greek.

**Monte Carlo**, GBM European call, 200k paths × 52 steps, f32:

| | Time |
|---|---|
| Pricing only (log space) | 0.55 ms |
| **Compiled engine, price + 5 Greeks** | **0.93 ms** (1.7× pricing) |
| Runtime tape, price + 5 Greeks | 3.3 ms |

**Finite differences**, American put, f64:

| Grid | Price + 5 Greeks | Pricing only |
|---|---|---|
| 201×200, one option | 0.18 ms | 0.068 ms |
| 201×200, 4 options (`fd_batch`) | 0.23 ms | 0.081 ms |
| 401×400, one option | 0.69 ms | 0.27 ms |
| 801×800, one option | 2.76 ms | 1.04 ms |

- The same scheme on the runtime tape is about 35× slower.
- The tridiagonal solves are limited by the latency of their recurrences. Batching 4 (f64) or 8 (f32) options into SIMD lanes therefore costs about the same as one.

**How the engines get there:**
- **Compile-time differentiation:** activity and liveness analysis, so the reverse sweep saves only the forward values it reads; first-assign adjoints.
- **Pathwise MC:** the setup runs once below the "mark", and each step is a fused kernel.
- **FD:** the implicit operator is LU-factorised once, so the time loops have no divisions. Recurrences are carried in registers, and the reverse sweep is two fused passes per step.

`benchmarks/` reproduces these numbers. For profiling it has a line-level PC sampler (`pcsample.py`) and an Apple-Silicon hardware-counter harness (`pmu/`).

## Accuracy

| Check | Result |
|---|---|
| Compiled vs tape, same model source, f64 (MC GBM, Asian, SVI; FD European, American) | agree to ~1e-14 |
| Every operation: compiled reverse mode vs tape, random inputs | ≤ 1e-11 |
| FD European call 801×800 vs Black-Scholes (price and 5 Greeks) | ≤ 1e-5 relative |
| FD American put 801×800 vs CRR binomial (4000 steps) | 2e-5 relative |
| MC GBM call vs Black-Scholes | within 4 standard errors |

For Greeks on fine FD grids, use f64. At 801×800, f32 rounding (up to 6e-4) exceeds the discretisation error.

## Second-order Greeks

The engines are generic over the value type (`autodiff/ct/real.mojo`): `Vec` (plain SIMD) for first order, or `Dual` (value plus one tangent). Running an engine on `Dual` inputs with tangent v gives the price, the gradient and the Hessian-vector product H·v in one forward-over-reverse pass.

```mojo
var s = mc_hvp[GbmCallSmooth, 16, DType.float32](params, v, n_steps, n_paths, normals)   # s.grad, s.hv
var h = fd_hessian[BsAmericanPutSmooth[0.002], 1, DType.float64](params, x_lo, x_hi, 401, 400)
print(h.get(0, 0), h.get(0, 2))                                                           # gamma, vanna
```

- `mc_hvp`, `fd_hvp`: one pass. `mc_hessian`, `fd_hessian`: one pass per parameter.
- `mc_hessian_bumped`, `fd_hessian_bumped`: central differences of the exact AAD gradient (not AD), for entries AD cannot see through a kink.
- `smooth_relu(x, eps)` is a softplus payoff. An FD model with `EXERCISE_EPS > 0` smooths the early-exercise max the same way.

**What is verified** (f64; FD 401×400, MC 200k × 52; `tests/test_second_order.mojo`):
- Every AD Hessian equals central differences of the AAD gradient (≤ 5e-7 European and MC, ≤ 5e-5 American) and is symmetric (≤ 4e-12). The Black-Scholes reference uses mpmath at 40 digits.
- **FD European:** every entry within 2e-3 of Black-Scholes (gamma 1e-5), except ∂²V/∂K². K enters only the kinked payoff, so AD returns 0; `fd_hessian_bumped` is within 5e-3.
- **FD American, hard max:** only the S₀ row and column (gamma, vanna, …) are valid. Moving σ, r, T or K moves the exercise boundary, and the price of the discrete scheme is only piecewise smooth in them (AD volga −29 vs about −1).
- **FD American, smoothed (ε = 0.002):** every entry except ∂²V/∂K² within 5e-3 of a 1%-bump reference. Smoothing shifts the price (7.5163 vs 7.5129), so use the hard max for price and first-order Greeks.
- **MC:** a kinked payoff gives pathwise gamma = 0 (the Dirac term at the strike is missed); use a smoothed payoff or `mc_hessian_bumped`. With ε = 0.5, gamma and vanna are within 4 standard errors of Black-Scholes. Volga is dominated by the O(ε²) smoothing bias (−1.04 at ε = 0.5 vs −1.48), so it is not claimed.

**Cost** (`benchmarks/bench_second_order.mojo`; 5 parameters; ratios vs the gradient at the same settings):

| Workload | Gradient | H·v | AD Hessian | Bumped Hessian |
|---|---|---|---|---|
| MC smoothed call, f32, W=16 | 1.74 ms | 2.53 ms (1.45×) | 12.8 ms (7.4×) | 19.2 ms (11×) |
| MC smoothed call, f32, W=32 | 1.16 ms | 3.33 ms (2.9×) | 16.6 ms (14×) | 12.8 ms (11×) |
| FD American put, hard max, 401×400, f64 | 0.71 ms | 1.29 ms (1.8×) | 6.5 ms (9.1×) | 7.9 ms (11×) |
| FD American put, ε = 0.002, 401×400, f64 | 2.71 ms | 3.92 ms (1.45×) | 19.7 ms (7.3×) | 30.0 ms (11×) |

- A `Dual` doubles the registers per value, so at W=32 f32 the MC step spills. W=16 is the best width for H·v; at each method's best width, AD and bumped Hessians cost the same (12.8 ms) in MC.
- The AD Hessian has no bump size to choose and no bump bias. It is about 1.2–1.5× cheaper than bumping, not more, because each H·v pass costs 1.45–1.8× a gradient.
- Softplus exercise costs about 3.8× the hard max at every order (an exp and a log per node per step).

## SVI

[apple.ipynb](apple.ipynb) fits raw-SVI parameters to AAPL call quotes. [euro_call_svi.mojo](euro_call_svi.mojo) (tape) and `models/svi.mojo` (compiled) price a K = 220 call by Monte Carlo with the SVI volatility, and return the Greeks plus the sensitivities to every SVI parameter in the same backward pass.

| | Analytical | MC + AAD |
|---|---|---|
| Price | 29.5669 | 29.5396 |
| Δ | 0.6487 | 0.6493 |
| θ = −∂V/∂T | −19.1122 | −19.0754 |
| ρ | 86.1373 | 86.2563 |

| SVI parameter | Sensitivity |
|---|---|
| a | 169.727 |
| b | 1088.79 |
| ρ | −0.3219 |
| m | 0.2499 |
| σ | 0.003895 |

- Theta holds the SVI volatility fixed: T is `detach`ed inside the variance-to-vol conversion.
- The fit is in strike space and unconstrained. The fitted ρ = −1.23 lies outside |ρ| < 1, so treat this as a demonstration rather than a calibrated surface.

## Setup

[Mojo 1.1](https://docs.modular.com/mojo/) via [uv](https://docs.astral.sh/uv/):

```bash
uv sync                                    # Mojo 1.1.0 + MAX 26.6 into .venv
uv sync --group notebooks                  # numpy, scipy, pandas, matplotlib, yfinance

.venv/bin/mojo run -I . tests/test_ct_ops.mojo        # each op: compiled reverse mode == tape
.venv/bin/mojo run -I . tests/test_mc_engine.mojo     # MC engine vs tape oracle and Black-Scholes
.venv/bin/mojo run -I . tests/test_fd_engine.mojo     # FD engine vs tape oracle, Black-Scholes, CRR; batch == single
.venv/bin/mojo run -I . tests/test_greeks.mojo        # runtime-tape pricer vs Black-Scholes
.venv/bin/mojo run -I . tests/test_second_order.mojo  # Hessians vs FD of the gradient and Black-Scholes

.venv/bin/mojo run -I . benchmarks/bench_engine.mojo     # MC: compiled vs tape vs pricing only
.venv/bin/mojo run -I . benchmarks/bench_fd_engine.mojo  # FD: grids and SIMD widths
.venv/bin/mojo run -I . benchmarks/bench_second_order.mojo  # H.v and Hessians vs first order
```

## Layout

```
autodiff/          runtime tape: Number, Tape, Node
autodiff/ct/       compile-time AAD: ADNum, Real (Vec, Dual), Kernel, Compiled, Scan, MC engine (mc.mojo), FD engine (fd.mojo), TapeNum
models/            GBM call, smoothed call and Asian (MC), SVI call (MC), Black-Scholes European/American, smoothed American (FD)
tests/             correctness suites
benchmarks/        timing, PC sampler, hardware counters (pmu/)
vanilla_euro_call.mojo, euro_call_svi.mojo   runtime-tape pricers
```

## Earlier results

The SIMD-speedup figures in earlier versions of this README were measured with a bug: paths reused each other's random numbers, and the last paths read past the end of the buffer. They have been removed. The bug, a tape-replay off-by-one, and a missing tape bounds check are fixed and covered by the tests.

## References

- [1] Gatheral, J. and Jacquier, A. (2013). 'Arbitrage-free SVI volatility surfaces', Quantitative Finance, 14(1), pp. 59–71.
- [2] Giles, M.B., & Glasserman, P. (2005). Smoking Adjoints: fast evaluation of Greeks in Monte Carlo calculations.
- [3] Savine, Antoine. (2019). Modern Computational Finance: AAD and Parallel Simulations. Wiley.
- [4] Griewank, A., & Walther, A. (2008). Evaluating Derivatives: Principles and Techniques of Algorithmic Differentiation, 2nd edition. SIAM.
