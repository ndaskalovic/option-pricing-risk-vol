"""GBM European and Asian calls, written once against ADNum (log-space exact scheme)."""
from autodiff import ADNum
from engines import McModel


struct GbmCall(McModel):
    """params: s0, r, sigma, T, K.   coeffs: drift, vol*sqrt(dt).   state: log S."""
    comptime NP = 5
    comptime NC = 2
    comptime NS = 1
    comptime NZ = 1

    @staticmethod
    def setup[N: ADNum](p: Array[N, Self.NP], n_steps: N) -> Array[N, Self.NC]:
        var r = p[1]; var sigma = p[2]; var T = p[3]
        var dt = T / n_steps
        var c = Array[N, Self.NC](fill=dt)
        c[0] = (r - 0.5 * sigma * sigma) * dt
        c[1] = sigma * dt.sqrt()
        return c^

    @staticmethod
    def init[N: ADNum](p: Array[N, Self.NP]) -> Array[N, Self.NS]:
        return Array[N, Self.NS](fill=p[0].log())

    @staticmethod
    def step[N: ADNum](s: Array[N, Self.NS], c: Array[N, Self.NC], z: Array[N, Self.NZ]) -> Array[N, Self.NS]:
        return Array[N, Self.NS](fill=s[0] + c[0] + c[1] * z[0])

    @staticmethod
    def payoff[N: ADNum](s: Array[N, Self.NS], p: Array[N, Self.NP], c: Array[N, Self.NC]) -> N:
        var r = p[1]; var T = p[3]; var K = p[4]
        return (s[0].exp() - K).relu() * (-r * T).exp()


struct GbmAsianCall(McModel):
    """Arithmetic-average Asian call.  state: log S, running sum of S.  coeffs: drift, vol*sqrt(dt), 1/n."""
    comptime NP = 5
    comptime NC = 3
    comptime NS = 2
    comptime NZ = 1

    @staticmethod
    def setup[N: ADNum](p: Array[N, Self.NP], n_steps: N) -> Array[N, Self.NC]:
        var r = p[1]; var sigma = p[2]; var T = p[3]
        var dt = T / n_steps
        var c = Array[N, Self.NC](fill=dt)
        c[0] = (r - 0.5 * sigma * sigma) * dt
        c[1] = sigma * dt.sqrt()
        c[2] = 1.0 / n_steps
        return c^

    @staticmethod
    def init[N: ADNum](p: Array[N, Self.NP]) -> Array[N, Self.NS]:
        var s = Array[N, Self.NS](fill=p[0].log())
        s[1] = p[0].constant(0.0)
        return s^

    @staticmethod
    def step[N: ADNum](s: Array[N, Self.NS], c: Array[N, Self.NC], z: Array[N, Self.NZ]) -> Array[N, Self.NS]:
        var x = s[0] + c[0] + c[1] * z[0]
        var out = Array[N, Self.NS](fill=x)
        out[1] = s[1] + x.exp()
        return out^

    @staticmethod
    def payoff[N: ADNum](s: Array[N, Self.NS], p: Array[N, Self.NP], c: Array[N, Self.NC]) -> N:
        var r = p[1]; var T = p[3]; var K = p[4]
        return (s[1] * c[2] - K).relu() * (-r * T).exp()


@always_inline
def smooth_relu[N: ADNum](x: N, eps: Float64) -> N:
    """Softplus max(x, 0) of width eps, eps * log(1 + exp(x / eps)), in overflow-safe form.
    Makes pathwise second derivatives (gamma, vanna) consistent; price bias O(eps^2)."""
    var ax = N.select_gt(x, x.constant(0.0), x, -x)
    return x.relu() + eps * (1.0 + (-ax / eps).exp()).log()


struct GbmCallSmooth(McModel):
    """GBM call with a softplus-smoothed payoff (width EPS), for pathwise second-order Greeks."""
    comptime NP = 5
    comptime NC = 2
    comptime NS = 1
    comptime NZ = 1
    comptime EPS = 0.5

    @staticmethod
    def setup[N: ADNum](p: Array[N, Self.NP], n_steps: N) -> Array[N, Self.NC]:
        return GbmCall.setup[N](p, n_steps)

    @staticmethod
    def init[N: ADNum](p: Array[N, Self.NP]) -> Array[N, Self.NS]:
        return GbmCall.init[N](p)

    @staticmethod
    def step[N: ADNum](s: Array[N, Self.NS], c: Array[N, Self.NC], z: Array[N, Self.NZ]) -> Array[N, Self.NS]:
        return GbmCall.step[N](s, c, z)

    @staticmethod
    def payoff[N: ADNum](s: Array[N, Self.NS], p: Array[N, Self.NP], c: Array[N, Self.NC]) -> N:
        var r = p[1]; var T = p[3]; var K = p[4]
        return smooth_relu(s[0].exp() - K, Self.EPS) * (-r * T).exp()
