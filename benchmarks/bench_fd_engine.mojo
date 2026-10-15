"""FD engine timing: American put, price + 5 Greeks, several grids and SIMD widths.

Run from repo root: mojo run -I . benchmarks/bench_fd_engine.mojo
"""
from std.math import log
from autodiff.ct import fd_compiled, fd_price
from models.bs_fd import BsAmericanPut
import std.time


def run[W: Int, DT: DType](nx: Int, nt: Int):
    var p = Array[Float64, 5](fill=0)
    p[0] = 100; p[1] = 0.01; p[2] = 0.2; p[3] = 1.0; p[4] = 100
    var lo = log(100.0) - 1.2; var hi = log(100.0) + 1.2
    var best = 1e9; var chk = 0.0
    for _ in range(11):
        var t0 = Int(std.time.perf_counter_ns())
        var c = fd_compiled[BsAmericanPut, W, DT](p, lo, hi, nx, nt)
        var t1 = Int(std.time.perf_counter_ns())
        chk += c.price + c.grad[0] + c.grad[1] + c.grad[2] + c.grad[3] + c.grad[4]
        best = min(best, Float64(t1 - t0) * 1e-6)
    var pv = Array[SIMD[DT, W], 5](fill=0)
    for j in range(5):
        pv[j] = SIMD[DT, W](p[j])
    var bp = 1e9
    for _ in range(11):
        var t0 = Int(std.time.perf_counter_ns())
        var v = fd_price[BsAmericanPut, W, DT](pv, lo, hi, nx, nt)
        var t1 = Int(std.time.perf_counter_ns())
        chk += Float64(v[0].cast[DType.float64]())
        bp = min(bp, Float64(t1 - t0) * 1e-6)
    print("W", W, DT, "grid", nx, "x", nt, "| price only", bp, "ms | price+5 Greeks", best, "ms (", best / bp, "x price) | ns/node-step/lane: price", bp * 1e6 / Float64(nx * nt * W), " aad", best * 1e6 / Float64(nx * nt * W), "| chk", chk)


def main() raises:
    run[1, DType.float64](201, 200)
    run[4, DType.float64](201, 200)
    run[8, DType.float32](201, 200)
    run[1, DType.float64](401, 400)
    run[4, DType.float64](401, 400)
    run[1, DType.float64](801, 800)
