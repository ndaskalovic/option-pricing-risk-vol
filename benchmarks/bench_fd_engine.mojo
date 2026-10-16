"""FD engine timing: American put, price + 5 Greeks, several grids and SIMD widths.

Run from repo root: mojo run -I . benchmarks/bench_fd_engine.mojo
"""
from std.math import log
from autodiff import Vec
from engines import fd_greeks, fd_price
from models.bs_fd import BsAmericanPut
import std.time


def run[W: Int, DT: DType](nx: Int, nt: Int):
    comptime R = Vec[DT, W]
    var p = Array[R, 5](fill=R.splat(0))
    p[0] = R.splat(100); p[1] = R.splat(0.01); p[2] = R.splat(0.2); p[3] = R.splat(1.0); p[4] = R.splat(100)
    var lo = log(100.0) - 1.2; var hi = log(100.0) + 1.2
    var best = 1e9; var chk = 0.0
    for _ in range(11):
        var t0 = Int(std.time.perf_counter_ns())
        var c = fd_greeks[BsAmericanPut, R](p, lo, hi, nx, nt)
        var t1 = Int(std.time.perf_counter_ns())
        var s = c.price.value() + c.grad[0].value() + c.grad[1].value() + c.grad[2].value() + c.grad[3].value() + c.grad[4].value()
        chk += Float64(s.cast[DType.float64]().reduce_add())
        best = min(best, Float64(t1 - t0) * 1e-6)
    var bp = 1e9
    for _ in range(11):
        var t0 = Int(std.time.perf_counter_ns())
        var v = fd_price[BsAmericanPut, R](p, lo, hi, nx, nt).value()
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
