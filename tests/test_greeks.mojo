"""Regression test: MC+AAD Greeks vs Black-Scholes, and compiled replay vs plain tape.

Run from repo root: mojo run -I . tests/test_greeks.mojo
"""
from std.memory.alloc import unsafe_alloc
from std.os import abort
from autodiff import Number, Tape
from vanilla_euro_call import EuropeanCallOption
import std.random

comptime W = 8
comptime DT = DType.float64
comptime N_STEPS = 52
comptime N_PATHS = 200_000 // W



def price[compiled: Bool, logspace: Bool](randoms: Pointer[Float32, _]) -> InlineResult:
    var tape = Tape[W, DT]()
    var s0 = Number[W, DT](100, tape)
    var r = Number[W, DT](0.01, tape)
    var sigma = Number[W, DT](0.2, tape)
    var T = Number[W, DT](1.0, tape)
    var K = Number[W, DT](100.0, tape)
    tape.set_mark()
    var p = EuropeanCallOption[W, DT, compiled, logspace](s0, r, sigma, T, K, N_STEPS, N_PATHS, randoms)
    var out = InlineResult(
        Float64(p.mean()), Float64(s0.mean_grad()), Float64(r.mean_grad()),
        Float64(sigma.mean_grad()), Float64(T.mean_grad()), Float64(K.mean_grad()),
    )
    tape.free()
    return out


@fieldwise_init
struct InlineResult(TrivialRegisterPassable):
    var price: Float64
    var delta: Float64
    var rho: Float64
    var vega: Float64
    var dT: Float64
    var dK: Float64

    def get(self, i: Int) -> Float64:
        if i == 0: return self.price
        if i == 1: return self.delta
        if i == 2: return self.rho
        if i == 3: return self.vega
        if i == 4: return self.dT
        return self.dK


def check[label: StaticString, logspace: Bool](randoms: Pointer[Float32, _]) -> Int:
    """Compiled replay must equal the plain tape bit-for-bit; Greeks within 4 SE of Black-Scholes."""
    var plain = price[False, logspace](randoms)
    var comp = price[True, logspace](randoms)
    # Black-Scholes closed form for S=K=100, r=0.01, sigma=0.2, T=1, and 4x MC standard errors
    var bs: List[Float64] = [8.433318690109608, 0.5596176923702425, 47.52845054691464, 39.447933090788894, 4.420077814548036, -0.4752845054691464]
    var tol: List[Float64] = [0.12, 0.0053, 0.44, 0.66, 0.2, 0.0044]
    var names: List[String] = ["price", "delta", "rho", "vega", "dV/dT", "dV/dK"]
    var failures = 0
    print("--", label)
    for i in range(6):
        var a = plain.get(i)
        var b = comp.get(i)
        var ok_replay = a == b
        var ok_bs = abs(b - bs[i]) <= tol[i]
        print(names[i], ": plain", a, " compiled", b, " BS", bs[i],
              " replay", "OK" if ok_replay else "MISMATCH", " BS", "OK" if ok_bs else "FAIL")
        if not (ok_replay and ok_bs):
            failures += 1
    return failures


def main() raises:
    var randoms = unsafe_alloc[Float32](N_STEPS * N_PATHS * W)
    std.random.seed(42)
    std.random.randn[DType.float32](randoms, N_STEPS * N_PATHS * W)
    var failures = check["exp-per-step", False](randoms) + check["log-space", True](randoms)
    randoms.unsafe_free()
    if failures > 0:
        abort("test_greeks FAILED")
    print("test_greeks PASSED")
