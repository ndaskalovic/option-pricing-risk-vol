"""Mojo compiled-engine timing on the same normals (run only when the machine is idle)."""
from std.memory.alloc import unsafe_alloc
from autodiff.ct import mc_compiled
from models.gbm import GbmCall
import std.random, std.time

comptime NS = 52
comptime W = 32


def main() raises:
    var nP = 200_000 // W
    var n = NS * nP * W
    var z = unsafe_alloc[Float32](n)
    std.random.seed(7); std.random.randn[DType.float32](z, n)   # identical to the dumped file
    var p = Array[Float64, 5](fill=0)
    p[0] = 100; p[1] = 0.01; p[2] = 0.2; p[3] = 1.0; p[4] = 100
    var best = 1e9; var chk = 0.0
    for _ in range(31):
        var t0 = Int(std.time.perf_counter_ns())
        var r = mc_compiled[GbmCall, W, DType.float32](p, NS, nP, z)
        var t1 = Int(std.time.perf_counter_ns())
        chk += r.price + r.grad[0] + r.grad[1] + r.grad[2] + r.grad[3] + r.grad[4]
        best = min(best, Float64(t1 - t0) * 1e-6)
    print("mojo  compiled engine (comptime graph)  best ms", best, " chk", chk)
    z.unsafe_free()
