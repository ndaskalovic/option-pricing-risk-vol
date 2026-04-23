from autodiff import Number, Tape
import std.random
import std.time

comptime width = 32
comptime type = DType.float32

def max_mask[W: Int, T: DType](arg1: Number[W, T], arg2: SIMD[T,W]) -> SIMD[T,W]:
    return arg1.value.gt(arg2).select(SIMD[T, W](1), SIMD[T, W](0))

# Simulate a single Monte Carlo path
def MonteCarloPathSimul[W: Int, DT: DType](
    s0: Number[W, DT],
    r: Number[W, DT],
    sigma: Number[W, DT],
    T: Number[W, DT],
    nTimeSteps: Int,
    randoms: UnsafePointer[Float32, _]
    ) -> Number[W, DT]:
    var dt = T / nTimeSteps
    var path = s0
    var sdt = dt.sqrt()
    var vsdt = sigma*sdt
    var drift = (r-((sigma**2)/2))*dt
    for i in range(nTimeSteps):
        path = path * (drift + vsdt * randoms.load[width=W](i*W).cast[DT]()).expon()
    return path


def EuropeanCallOption[W: Int, DT: DType](
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
        var path = MonteCarloPathSimul(s0, r, sigma, T, nTimeSteps, randoms + i*nTimeSteps)
        var payoff = (path-K) * max_mask[W, DT](path-K, 0) * (-r * T).expon()
        # print(payoff.value)
        payoff.propagate_adjoints(Scalar[DT](1.0 / Float64(nPaths)))
        tape.rewind_to_mark()
        psum = psum + payoff

        # After first path, compile the tape — all subsequent paths replay
        if i == 0:
            tape.compile()

    return psum / nPaths


def run[width: Int, type: DType]() -> Float64:
    var tape = Tape[width, type]()
    var s0 = Number[width, type](100, tape)
    var r = Number[width, type](0.01, s0.tape)
    var sigma = Number[width, type](0.2, s0.tape)
    var T = Number[width, type](1.0, s0.tape)
    var K = Number[width, type](100.0, s0.tape)
    tape.set_mark()
    var nTimeSteps = 52
    var nPaths = 200000 // width
    var randoms = alloc[Float32](nTimeSteps*nPaths)
    std.random.randn[DType.float32](randoms, nTimeSteps*nPaths)
    var start = std.time.perf_counter_ns()
    var optionPrice = EuropeanCallOption(s0, r, sigma, T, K, nTimeSteps, nPaths, randoms)
    var end = std.time.perf_counter_ns()
    print(optionPrice.mean(), flush=True)
    _ = optionPrice
    print(optionPrice.mean(), tape.get_tail(), s0.mean_grad(), r.mean_grad(), sigma.mean_grad(), T.mean_grad(), K.mean_grad())
    _ = s0.mean_grad()
    _ = r.mean_grad()
    _ = sigma.mean_grad()
    _ = T.mean_grad()
    randoms.free()
    tape.free()
    return Float64(Int(end - start)) * 1e-9

def bench[width: Int, type: DType]():
    var total = 0.0
    for _ in range(10):
        total += run[width, type]()
    print(width, ",", total/10)


def main() raises:
    # bench[1, type]()
    # bench[2, type]()
    # bench[4, type]()
    # bench[8, type]()
    # bench[16, type]()
    # bench[32, type]()
    bench[64, type]()
