"""MC engine: compiled == tape oracle (same model source, f64), and GBM call vs Black-Scholes.

Run from repo root: mojo run -I . tests/test_mc_engine.mojo
"""
from std.os import abort
from std.memory.alloc import unsafe_alloc
from autodiff.ct import McModel, McResult, mc_compiled, mc_tape
from models.gbm import GbmCall, GbmAsianCall
from models.svi import SviCall
import std.random

comptime W = 8
comptime DT = DType.float64


def compare[M: McModel](label: StaticString, params: Array[Float64, M.NP], n_steps: Int, n_paths: Int,
                        z: Pointer[Float32, _]) -> Int:
    var c = mc_compiled[M, W, DT](params, n_steps, n_paths, z)
    var t = mc_tape[M, W, DT](params, n_steps, n_paths, z)
    print(label)
    print("  compiled", c)
    print("  tape    ", t)
    var bad = 0
    if abs(c.price - t.price) > 1e-10 * abs(t.price):
        bad += 1
    for j in range(M.NP):
        if abs(c.grad[j] - t.grad[j]) > 1e-10 * (abs(t.grad[j]) + 1e-12):
            print("  mismatch grad", j); bad += 1
    return bad


def main() raises:
    var n_steps = 52
    var n_paths = 200_000 // W
    var z = unsafe_alloc[Float32](n_paths * n_steps * W)
    std.random.seed(42)
    std.random.randn[DType.float32](z, n_paths * n_steps * W)
    var params = Array[Float64, 5](fill=0)
    params[0] = 100; params[1] = 0.01; params[2] = 0.2; params[3] = 1.0; params[4] = 100
    var failures = compare[GbmCall]("GBM call", params, n_steps, n_paths, z)
    failures += compare[GbmAsianCall]("GBM Asian call", params, n_steps, n_paths // 4, z)
    var sp = Array[Float64, 9](fill=0)
    sp[0] = 227.7899932861328; sp[1] = 0.04; sp[2] = 0.7287671233; sp[3] = 220
    sp[4] = 5.73431376e-02; sp[5] = 6.59987208e-04; sp[6] = -1.23157254; sp[7] = 2.22873847e+02; sp[8] = 9.99943135e-02
    failures += compare[SviCall]("SVI call (params: s0, r, T, K, a, b, rho, m, sigma)", sp, n_steps, n_paths, z)

    # GBM call vs Black-Scholes (4 MC standard errors), and vs the existing tape pricer (same randoms)
    var c = mc_compiled[GbmCall, W, DT](params, n_steps, n_paths, z)
    var bs: List[Float64] = [8.433318690109608, 0.5596176923702425, 47.52845054691464, 39.447933090788894, 4.420077814548036, -0.4752845054691464]
    var tol: List[Float64] = [0.12, 0.0053, 0.44, 0.66, 0.2, 0.0044]
    var ours: List[Float64] = [c.price, c.grad[0], c.grad[1], c.grad[2], c.grad[3], c.grad[4]]
    var legacy: List[Float64] = [8.461696909187388, 0.5612220088095556, 47.66050397176403, 39.576525496371865, 4.434257589354827, -0.4766050397176101]
    for i in range(6):
        if abs(ours[i] - bs[i]) > tol[i]:
            print("  BS check failed", i, ours[i], bs[i]); failures += 1
        if abs(ours[i] - legacy[i]) > 1e-9 * abs(legacy[i]):
            print("  legacy check failed", i, ours[i], legacy[i]); failures += 1
    z.unsafe_free()
    if failures > 0:
        abort("test_mc_engine FAILED")
    print("test_mc_engine PASSED")
