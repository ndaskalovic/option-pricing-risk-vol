"""Black-Scholes European call / American put for the FD engine, written once against ADNum."""
from autodiff.ct import ADNum
from engines import FdModel


struct BsEuroCall(FdModel):
    """params: s0, r, sigma, T, K."""
    comptime NP = 5
    comptime AMERICAN = False
    comptime EXERCISE_EPS = 0.0

    @staticmethod
    def coeffs[N: ADNum](x: N, p: Array[N, Self.NP]) -> Array[N, 3]:
        var r = p[1]; var sigma = p[2]
        var k = Array[N, 3](fill=r)
        k[0] = 0.5 * sigma * sigma
        k[1] = r - 0.5 * sigma * sigma
        k[2] = r
        return k^

    @staticmethod
    def payoff[N: ADNum](x: N, p: Array[N, Self.NP]) -> N:
        return (x.exp() - p[4]).relu()

    @staticmethod
    def spot[N: ADNum](p: Array[N, Self.NP]) -> N:
        return p[0].log()

    @staticmethod
    def maturity[N: ADNum](p: Array[N, Self.NP]) -> N:
        return p[3]


struct BsAmericanPut(FdModel):
    comptime NP = 5
    comptime AMERICAN = True
    comptime EXERCISE_EPS = 0.0

    @staticmethod
    def coeffs[N: ADNum](x: N, p: Array[N, Self.NP]) -> Array[N, 3]:
        return BsEuroCall.coeffs[N](x, p)

    @staticmethod
    def payoff[N: ADNum](x: N, p: Array[N, Self.NP]) -> N:
        return (p[4] - x.exp()).relu()

    @staticmethod
    def spot[N: ADNum](p: Array[N, Self.NP]) -> N:
        return p[0].log()

    @staticmethod
    def maturity[N: ADNum](p: Array[N, Self.NP]) -> N:
        return p[3]


struct BsAmericanPutSmooth[eps: Float64 = 0.01](FdModel):
    """American put with a softplus-smoothed exercise decision of width `eps`, so that
    second-order Greeks w.r.t. boundary-moving parameters are well defined."""
    comptime NP = 5
    comptime AMERICAN = True
    comptime EXERCISE_EPS = Self.eps

    @staticmethod
    def coeffs[N: ADNum](x: N, p: Array[N, Self.NP]) -> Array[N, 3]:
        return BsEuroCall.coeffs[N](x, p)

    @staticmethod
    def payoff[N: ADNum](x: N, p: Array[N, Self.NP]) -> N:
        return BsAmericanPut.payoff[N](x, p)

    @staticmethod
    def spot[N: ADNum](p: Array[N, Self.NP]) -> N:
        return p[0].log()

    @staticmethod
    def maturity[N: ADNum](p: Array[N, Self.NP]) -> N:
        return p[3]


struct BsPutMoneyness[american: Bool = True](FdModel):
    """Unit-strike put on x = log(S/K) for the surface engine: V(S, K, T) = K u(log(S/K), T),
    so one solve prices every strike. params: r, q (continuous dividend yield), sigma."""
    comptime NP = 3
    comptime AMERICAN = Self.american
    comptime EXERCISE_EPS = 0.0

    @staticmethod
    def coeffs[N: ADNum](x: N, p: Array[N, Self.NP]) -> Array[N, 3]:
        var r = p[0]; var q = p[1]; var sigma = p[2]
        var k = Array[N, 3](fill=r)
        k[0] = 0.5 * sigma * sigma
        k[1] = r - q - 0.5 * sigma * sigma
        k[2] = r
        return k^

    @staticmethod
    def payoff[N: ADNum](x: N, p: Array[N, Self.NP]) -> N:
        return (1.0 - x.exp()).relu()

    @staticmethod
    def spot[N: ADNum](p: Array[N, Self.NP]) -> N:
        return p[0].constant(0.0)       # unused by the surface engine

    @staticmethod
    def maturity[N: ADNum](p: Array[N, Self.NP]) -> N:
        return p[0].constant(0.0)       # unused: maturities are the surface's time grid
