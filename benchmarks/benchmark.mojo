# Benchmark harness for European Call Option pricing with AAD
#
# Run from project root: mojo run benchmarks/benchmark.mojo
# Redirect output: mojo run benchmarks/benchmark.mojo > benchmarks/results/run.txt

from autodiff import Number, Tape
from vanilla_euro_call import EuropeanCallOption
import std.random
import std.time

comptime SIMD_WIDTH = 32
comptime FLOAT_TYPE = DType.float32
comptime NUM_TRIALS = 5


def run_benchmark[width: Int, type: DType](nPaths: Int, nTimeSteps: Int):
    """Run a single benchmark configuration and print results."""
    var total_time = 0.0
    var price = Float64(0)
    var delta = Float64(0)
    var rho = Float64(0)
    var vega = Float64(0)
    var theta = Float64(0)
    var peak_n = 0
    var total_p = 0
    var peak_bytes = 0

    for trial in range(NUM_TRIALS):
        var tape = Tape[width, type]()
        var s0 = Number[width, type](100, tape)
        var r = Number[width, type](0.01, s0.tape)
        var sigma = Number[width, type](0.2, s0.tape)
        var T = Number[width, type](1.0, s0.tape)
        var K = Number[width, type](100.0, s0.tape)
        tape.set_mark()

        var effective_paths = nPaths // width
        var randoms = alloc[Float32](nTimeSteps * effective_paths)
        std.random.randn[DType.float32](randoms, nTimeSteps * effective_paths)

        tape.reset_metrics()
        var start = std.time.perf_counter_ns()
        var optionPrice = EuropeanCallOption(s0, r, sigma, T, K, nTimeSteps, effective_paths, randoms)
        var end = std.time.perf_counter_ns()

        var elapsed = Float64(Int(end - start)) * 1e-9
        total_time += elapsed

        # Collect results from the last trial
        if trial == NUM_TRIALS - 1:
            price = Float64(optionPrice.mean())
            delta = Float64(s0.mean_grad())
            rho = Float64(r.mean_grad())
            vega = Float64(sigma.mean_grad())
            theta = Float64(T.mean_grad())
            peak_n = tape.peak_nodes()
            total_p = tape.total_pushes()
            peak_bytes = tape.peak_adjoint_bytes()

        _ = optionPrice
        _ = s0.mean_grad()
        _ = r.mean_grad()
        _ = sigma.mean_grad()
        _ = T.mean_grad()
        randoms.free()
        tape.free()

    var avg_time = total_time / NUM_TRIALS

    # Print row: paths, steps, width, time, price, delta, rho, vega, theta, peak_nodes, total_pushes, peak_bytes
    print(
        nPaths, ",",
        nTimeSteps, ",",
        width, ",",
        avg_time, ",",
        price, ",",
        delta, ",",
        rho, ",",
        vega, ",",
        theta, ",",
        peak_n, ",",
        total_p, ",",
        peak_bytes,
    )


def main() raises:
    print("===== AAD European Call Option Benchmark =====")
    print("Parameters: S0=100, r=0.01, sigma=0.2, T=1.0, K=100.0")
    print("Trials per config:", NUM_TRIALS)
    print("SIMD width:", SIMD_WIDTH)
    print("")
    print("paths, steps, width, avg_time_s, price, delta, rho, vega, theta, peak_nodes, total_pushes, peak_adj_bytes")

    # --- Sweep paths with fixed steps=52 ---
    run_benchmark[SIMD_WIDTH, FLOAT_TYPE](10_000, 52)
    run_benchmark[SIMD_WIDTH, FLOAT_TYPE](50_000, 52)
    run_benchmark[SIMD_WIDTH, FLOAT_TYPE](100_000, 52)
    run_benchmark[SIMD_WIDTH, FLOAT_TYPE](200_000, 52)
    run_benchmark[SIMD_WIDTH, FLOAT_TYPE](500_000, 52)

    # --- Sweep steps with fixed paths=100_000 ---
    run_benchmark[SIMD_WIDTH, FLOAT_TYPE](100_000, 12)
    run_benchmark[SIMD_WIDTH, FLOAT_TYPE](100_000, 26)
    # 100_000 x 52 already covered above
    run_benchmark[SIMD_WIDTH, FLOAT_TYPE](100_000, 104)
    run_benchmark[SIMD_WIDTH, FLOAT_TYPE](100_000, 252)

    # --- Full grid (remaining combos) ---
    # paths=10_000
    run_benchmark[SIMD_WIDTH, FLOAT_TYPE](10_000, 12)
    run_benchmark[SIMD_WIDTH, FLOAT_TYPE](10_000, 26)
    run_benchmark[SIMD_WIDTH, FLOAT_TYPE](10_000, 104)
    run_benchmark[SIMD_WIDTH, FLOAT_TYPE](10_000, 252)

    # paths=50_000
    run_benchmark[SIMD_WIDTH, FLOAT_TYPE](50_000, 12)
    run_benchmark[SIMD_WIDTH, FLOAT_TYPE](50_000, 26)
    run_benchmark[SIMD_WIDTH, FLOAT_TYPE](50_000, 104)
    run_benchmark[SIMD_WIDTH, FLOAT_TYPE](50_000, 252)

    # paths=200_000
    run_benchmark[SIMD_WIDTH, FLOAT_TYPE](200_000, 12)
    run_benchmark[SIMD_WIDTH, FLOAT_TYPE](200_000, 26)
    run_benchmark[SIMD_WIDTH, FLOAT_TYPE](200_000, 104)
    run_benchmark[SIMD_WIDTH, FLOAT_TYPE](200_000, 252)

    # paths=500_000
    run_benchmark[SIMD_WIDTH, FLOAT_TYPE](500_000, 12)
    run_benchmark[SIMD_WIDTH, FLOAT_TYPE](500_000, 26)
    run_benchmark[SIMD_WIDTH, FLOAT_TYPE](500_000, 104)
    run_benchmark[SIMD_WIDTH, FLOAT_TYPE](500_000, 252)

    print("")
    print("===== Benchmark complete =====")
