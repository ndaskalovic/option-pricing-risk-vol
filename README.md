# Option Pricing, Risk and Volatility Modelling

Adjoint algorithmic differentiation (AAD) for option Greeks, written in [Mojo](https://www.modular.com/mojo). It started as part of my master's thesis on AAD for Monte Carlo Greeks under Black-Scholes.

You write a model once as a few small functions. The compiler differentiates them at compile time, and the Monte Carlo and finite-difference engines return the price and every Greek in one run.

## Example

```mojo
struct GbmCall(McModel):
    """params: s0, r, sigma, T, K."""
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
print(r.price, r.grad[0], r.grad[2])      # price, delta, vega
```

Finite-difference models (e.g. an American put) work the same way: `fd_compiled[BsAmericanPut, 1, DType.float64](params, x_lo, x_hi, 201, 200)`.
