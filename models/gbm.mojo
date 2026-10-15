"""GBM European and Asian calls, written once against ADNum (log-space exact scheme)."""
from autodiff.ct import ADNum, McModel


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
