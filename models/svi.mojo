"""SVI-vol European call for the MC engine (log-space GBM with vol from a raw-SVI slice).

params: s0, r, T, K, a, b, rho, m, sigma.  vol = sqrt(w(K) / T)
with T detached, so Theta holds the SVI vol fixed.
"""
from autodiff import ADNum
from engines import McModel


struct SviCall(McModel):
    comptime NP = 9
    comptime NC = 2
    comptime NS = 1
    comptime NZ = 1

    @staticmethod
    def setup[N: ADNum](p: Array[N, Self.NP], n_steps: N) -> Array[N, Self.NC]:
        var r = p[1]; var T = p[2]; var K = p[3]
        var a = p[4]; var b = p[5]; var rho = p[6]; var m = p[7]; var sig = p[8]
        var km = K - m
        var w = a + b * (rho * km + (km * km + sig * sig) ** 0.5)
        var vol = (w / T.detach()).sqrt()
        var dt = T / n_steps
        var c = Array[N, Self.NC](fill=dt)
        c[0] = (r - 0.5 * vol * vol) * dt
        c[1] = vol * dt.sqrt()
        return c^

    @staticmethod
    def init[N: ADNum](p: Array[N, Self.NP]) -> Array[N, Self.NS]:
        return Array[N, Self.NS](fill=p[0].log())

    @staticmethod
    def step[N: ADNum](s: Array[N, Self.NS], c: Array[N, Self.NC], z: Array[N, Self.NZ]) -> Array[N, Self.NS]:
        return Array[N, Self.NS](fill=s[0] + c[0] + c[1] * z[0])

    @staticmethod
    def payoff[N: ADNum](s: Array[N, Self.NS], p: Array[N, Self.NP], c: Array[N, Self.NC]) -> N:
        var r = p[1]; var T = p[2]; var K = p[3]
        return (s[0].exp() - K).relu() * (-r * T).exp()
