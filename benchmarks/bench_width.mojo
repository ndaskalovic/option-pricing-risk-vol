from autodiff import Number, Tape
from vanilla_euro_call import EuropeanCallOption
import std.random
import std.time

comptime FLOAT_TYPE = DType.float32
comptime NUM_TRIALS = 5


def run_benchmark[width: Int, type: DType](nPaths: Int, nTimeSteps: Int):
    var total_time = 0.0
    var price = Float64(0)

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

        if trial == NUM_TRIALS - 1:
            price = Float64(optionPrice.mean())

        _ = optionPrice
        _ = s0.mean_grad()
        _ = r.mean_grad()
        _ = sigma.mean_grad()
        _ = T.mean_grad()
        randoms.free()
        tape.free()

    var avg_time = total_time / NUM_TRIALS
    print(nPaths, ",", nTimeSteps, ",", width, ",", avg_time, ",", price)


def main() raises:
    print("paths, steps, width, avg_time_s, price")

    # Fixed: 200k paths, 52 steps — sweep widths
    run_benchmark[1, FLOAT_TYPE](200_000, 52)
    run_benchmark[2, FLOAT_TYPE](200_000, 52)
    run_benchmark[4, FLOAT_TYPE](200_000, 52)
    run_benchmark[8, FLOAT_TYPE](200_000, 52)
    run_benchmark[16, FLOAT_TYPE](200_000, 52)
    run_benchmark[32, FLOAT_TYPE](200_000, 52)
    run_benchmark[64, FLOAT_TYPE](200_000, 52)
    run_benchmark[128, FLOAT_TYPE](200_000, 52)
    run_benchmark[256, FLOAT_TYPE](200_000, 52)
