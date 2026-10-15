"""Black-Scholes European call / American put for the FD engine, written once against ADNum."""
from autodiff.ct import ADNum, FdModel


struct BsEuroCall(FdModel):
    """params: s0, r, sigma, T, K."""
    comptime NP = 5
    comptime AMERICAN = False

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
