from std.memory.alloc import unsafe_alloc
from autodiff import Number, Tape
from std.math import exp
import std.random
import std.time

comptime width = 32
comptime type = DType.float32

# Simulate a single Monte Carlo path from precomputed (path-invariant) drift and vol*sqrt(dt)
def gbm_path[W: Int, DT: DType](
    s0: Number[W, DT],
    drift: Number[W, DT],
    vsdt: Number[W, DT],
    nTimeSteps: Int,
    randoms: Pointer[Float32, _]
    ) -> Number[W, DT]:
    comptime assert DT.is_floating_point(), "GBM path requires a floating-point dtype"
    var path = s0
    for i in range(nTimeSteps):
        # path' = path * exp(drift + vsdt * z), recorded as one node with analytic partials
        var z = randoms.unsafe_load[width=W](i*W).cast[DT]()
        var e = exp(drift.value + vsdt.value * z)
        var nxt = path.value * e
        path = Number[W, DT].from_partials(nxt, path, e, drift, nxt, vsdt, nxt * z)
    return path


# Same exact scheme in log space: x = log S, x' = x + drift + vsdt * z, one exp per path.
# For GBM this is identical in exact arithmetic and ~5x cheaper (no exp per step).
@always_inline
def gbm_log_path[W: Int, DT: DType](
    s0: Number[W, DT],
    drift: Number[W, DT],
    vsdt: Number[W, DT],
    nTimeSteps: Int,
    randoms: Pointer[Float32, _]
    ) -> Number[W, DT]:
    comptime assert DT.is_floating_point(), "GBM path requires a floating-point dtype"
    var x = s0.log()
    for i in range(nTimeSteps):
        var z = randoms.unsafe_load[width=W](i*W).cast[DT]()
        x = Number[W, DT].from_partials(x.value + drift.value + vsdt.value * z, x, 1, drift, 1, vsdt, z)
    return x.expon()


def gbm_coeffs[W: Int, DT: DType](
    r: Number[W, DT], sigma: Number[W, DT], T: Number[W, DT], nTimeSteps: Int
    ) -> Tuple[Number[W, DT], Number[W, DT]]:
    """(drift, vol*sqrt(dt)) per step of the exact log-Euler GBM scheme."""
    var dt = T / nTimeSteps
    var vsdt = sigma * dt.sqrt()
    var drift = (r - ((sigma**2) / 2)) * dt
    return (drift, vsdt)


def discounted_call[W: Int, DT: DType](
    path: Number[W, DT], K: Number[W, DT], disc: Number[W, DT]
    ) -> Number[W, DT]:
    """Discounted call payoff `disc * max(path - K, 0)` recorded as one node."""
    var itm = path.value.gt(K.value).select(SIMD[DT, W](1), SIMD[DT, W](0))
    var intrinsic = (path.value - K.value) * itm
    var dd = disc.value * itm
    return Number[W, DT].from_partials(disc.value * intrinsic, path, dd, K, -dd, disc, intrinsic)


def EuropeanCallOption[W: Int, DT: DType, compiled: Bool = True, logspace: Bool = True](
    s0: Number[W, DT],
    r: Number[W, DT],
    sigma: Number[W, DT],
    T: Number[W, DT],
    K: Number[W, DT],
    nTimeSteps: Int,
    nPaths: Int,
    randoms: Pointer[Float32, _]
    ) -> Number[W, DT]:

    var tape = s0.tape
    # Path-invariant nodes are recorded once, below the mark
    var c = gbm_coeffs(r, sigma, T, nTimeSteps)
    var disc = (-r * T).expon()
    tape.set_mark()
    var psum = SIMD[DT, W](0)

    for i in range(nPaths):
        var zp = randoms.unsafe_offset(i*nTimeSteps * W)
        var path: Number[W, DT]
        comptime if logspace:
            path = gbm_log_path(s0, c[0], c[1], nTimeSteps, zp)
        else:
            path = gbm_path(s0, c[0], c[1], nTimeSteps, zp)
        var payoff = discounted_call(path, K, disc)
        payoff.propagate_adjoints(Scalar[DT](1.0 / Float64(nPaths)))
        psum += payoff.value
        tape.rewind_to_mark()

        # After first path, compile the tape — all subsequent paths replay
        comptime if compiled:
            if i == 0:
                tape.compile()

    tape.propagate_mark_to_start()
    return Number[W, DT](psum / Scalar[DT](nPaths), tape, push=False)


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
    var randoms = unsafe_alloc[Float32](nTimeSteps*nPaths * width)
    std.random.randn[DType.float32](randoms, nTimeSteps*nPaths * width)
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
    randoms.unsafe_free()
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
