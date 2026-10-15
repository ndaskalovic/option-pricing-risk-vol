"""Dump the exact normals (seed 7, W=32 layout) and Mojo's compiled-engine results (no timing)."""
from std.memory.alloc import unsafe_alloc
from autodiff.ct import mc_compiled
from models.gbm import GbmCall
import std.random

comptime NS = 52
comptime W = 32


def main() raises:
    var nP = 200_000 // W
    var n = NS * nP * W
    var z = unsafe_alloc[Float32](n)
    std.random.seed(7); std.random.randn[DType.float32](z, n)
    with open("experiments/rust_bench/normals_w32.f32", "w") as f:
        f.write_bytes(Span[Byte, MutUntrackedOrigin](unsafe_ptr=z.unsafe_bitcast[Byte](), length=n * 4))
    var p = Array[Float64, 5](fill=0)
    p[0] = 100; p[1] = 0.01; p[2] = 0.2; p[3] = 1.0; p[4] = 100
    print("mojo f64:", mc_compiled[GbmCall, W, DType.float64](p, NS, nP, z))
    print("mojo f32:", mc_compiled[GbmCall, W, DType.float32](p, NS, nP, z))
    z.unsafe_free()
