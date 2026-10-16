"""FD engine (compiled) vs the same scheme on the tape: price + 5 Greeks, one American put.

Run from repo root: mojo run -I . benchmarks/bench_fd.mojo
"""
from std.math import log
from engines import fd_compiled, fd_tape
from models.bs_fd import BsAmericanPut
import std.time


def run[W: Int, DT: DType](nx: Int, nt: Int):
    var p = Array[Float64, 5](fill=0)
    p[0] = 100; p[1] = 0.01; p[2] = 0.2; p[3] = 1.0; p[4] = 100
    var lo = log(100.0) - 1.2; var hi = log(100.0) + 1.2
    var tc = 1e9; var tt = 1e9; var v = 0.0
    for _ in range(5):
        var t0 = Int(std.time.perf_counter_ns())
        var c = fd_compiled[BsAmericanPut, W, DT](p, lo, hi, nx, nt)
        var t1 = Int(std.time.perf_counter_ns())
        var t = fd_tape[BsAmericanPut, W, DT](p, lo, hi, nx, nt)
        var t2 = Int(std.time.perf_counter_ns())
        v += c.price + c.grad[0] + c.grad[2] + t.price + t.grad[0]
        tc = min(tc, Float64(t1 - t0) * 1e-6); tt = min(tt, Float64(t2 - t1) * 1e-6)
    print("W", W, DT, "grid", nx, "x", nt, "| compiled", tc, "ms | tape", tt, "ms | tape/compiled", tt / tc, "| chk", v)


def main() raises:
    run[1, DType.float64](201, 200)
    run[4, DType.float64](201, 200)
    run[8, DType.float32](201, 200)
    run[4, DType.float64](401, 400)
