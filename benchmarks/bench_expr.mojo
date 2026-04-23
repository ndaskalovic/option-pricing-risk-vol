from autodiff import Number, Tape, Leaf, ExpMulExpr, ExprNumber, make_leaf
from vanilla_euro_call import EuropeanCallOption, MonteCarloPathSimul, max_mask
import std.random
import std.time

comptime FLOAT_TYPE = DType.float32
comptime NUM_TRIALS = 5


def MonteCarloPathSimulFused[W: Int, DT: DType](
    s0: Number[W, DT],
    r: Number[W, DT],
    sigma: Number[W, DT],
    T: Number[W, DT],
    nTimeSteps: Int,
    randoms: UnsafePointer[Float32, _]
    ) -> Number[W, DT]:
    var dt = T / nTimeSteps
    var sdt = dt.sqrt()
    var vsdt = sigma * sdt
    var drift = (r - ((sigma ** 2) / 2)) * dt
    var tape = s0.tape

    # First step: use standard Number ops to get a Number for path
    var vsdt_rand0 = vsdt * randoms.load[width=W](0).cast[DT]()
    var drift_leaf = Leaf[W, DT](drift.value, drift.idx)
    var vsdt_rand0_leaf = Leaf[W, DT](vsdt_rand0.value, vsdt_rand0.idx)
    var path_leaf = Leaf[W, DT](s0.value, s0.idx)
    # exp(drift + vsdt*rand) * path → 1 fused node instead of 3
    var fused = ExpMulExpr[W, DT](drift_leaf, vsdt_rand0_leaf, path_leaf)
    var result = fused.record(tape)
    var path_val = result.value
    var path_idx = result.idx

    for i in range(1, nTimeSteps):
        var vsdt_rand = vsdt * randoms.load[width=W](i * W).cast[DT]()
        var vr_leaf = Leaf[W, DT](vsdt_rand.value, vsdt_rand.idx)
        var d_leaf = Leaf[W, DT](drift.value, drift.idx)
        var p_leaf = Leaf[W, DT](path_val, path_idx)
        var f = ExpMulExpr[W, DT](d_leaf, vr_leaf, p_leaf)
        var r2 = f.record(tape)
        path_val = r2.value
        path_idx = r2.idx

    return Number[W, DT](path_val, tape, push=False)


def EuropeanCallOptionFused[W: Int, DT: DType](
    s0: Number[W, DT],
    r: Number[W, DT],
    sigma: Number[W, DT],
    T: Number[W, DT],
    K: Number[W, DT],
    nTimeSteps: Int,
    nPaths: Int,
    randoms: UnsafePointer[Float32, _]
    ) -> Number[W, DT]:

    var tape = s0.tape
    var psum = Number[W, DT](0.0, tape)

    for i in range(nPaths):
        var path = MonteCarloPathSimulFused(s0, r, sigma, T, nTimeSteps, randoms + i*nTimeSteps)
        var payoff = (path-K) * max_mask[W, DT](path-K, 0) * (-r * T).expon()
        payoff.propagate_adjoints(Scalar[DT](1.0 / Float64(nPaths)))
        tape.rewind_to_mark()
        psum = psum + payoff

        if i == 0:
            tape.compile()

    return psum / nPaths


def run_standard[width: Int, type: DType](nPaths: Int, nTimeSteps: Int):
    var total_time = 0.0
    var price = Float64(0)
    var delta = Float64(0)
    var peak = 0

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

        total_time += Float64(Int(end - start)) * 1e-9
        if trial == NUM_TRIALS - 1:
            price = Float64(optionPrice.mean())
            delta = Float64(s0.mean_grad())
            peak = tape.peak_nodes()

        _ = optionPrice
        _ = s0.mean_grad()
        _ = r.mean_grad()
        _ = sigma.mean_grad()
        _ = T.mean_grad()
        randoms.free()
        tape.free()

    print("standard ,", nPaths, ",", nTimeSteps, ",", width, ",",
          total_time / NUM_TRIALS, ",", price, ",", delta, ",", peak)


def run_fused[width: Int, type: DType](nPaths: Int, nTimeSteps: Int):
    var total_time = 0.0
    var price = Float64(0)
    var delta = Float64(0)
    var peak = 0

    for trial in range(NUM_TRIALS):
        var tape = Tape[width, type](500_000)
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
        var optionPrice = EuropeanCallOptionFused(s0, r, sigma, T, K, nTimeSteps, effective_paths, randoms)
        var end = std.time.perf_counter_ns()

        total_time += Float64(Int(end - start)) * 1e-9
        if trial == NUM_TRIALS - 1:
            price = Float64(optionPrice.mean())
            delta = Float64(s0.mean_grad())
            peak = tape.peak_nodes()

        _ = optionPrice
        _ = s0.mean_grad()
        _ = r.mean_grad()
        _ = sigma.mean_grad()
        _ = T.mean_grad()
        randoms.free()
        tape.free()

    print("fused    ,", nPaths, ",", nTimeSteps, ",", width, ",",
          total_time / NUM_TRIALS, ",", price, ",", delta, ",", peak)


def main() raises:
    print("mode, paths, steps, width, avg_time_s, price, delta, peak_nodes")

    # 52 weekly steps
    run_standard[32, FLOAT_TYPE](200_000, 52)
    run_fused[32, FLOAT_TYPE](200_000, 52)

    # 252 daily steps
    run_standard[32, FLOAT_TYPE](200_000, 252)
    run_fused[32, FLOAT_TYPE](200_000, 252)

    # 504 twice-daily
    run_standard[32, FLOAT_TYPE](200_000, 504)
    run_fused[32, FLOAT_TYPE](200_000, 504)

    # 1000 steps
    run_standard[32, FLOAT_TYPE](200_000, 1000)
    run_fused[32, FLOAT_TYPE](200_000, 1000)
