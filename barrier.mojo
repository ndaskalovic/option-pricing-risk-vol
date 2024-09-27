from autodiff import Number, Tape
import random

alias width = 4
alias type = DType.float32

fn max_mask[W: Int, T: DType](arg1: Number[W, T], arg2: SIMD[T,W]) -> SIMD[T,W]:
    return (arg1.value > arg2).cast[T]()

# Simulate a single Monte Carlo path
fn MonteCarloPathSimul[W: Int, DT: DType](
    s0: Number[W, DT],
    r: Number[W, DT],
    sigma: Number[W, DT],
    T: Number[W, DT],
    nTimeSteps: Int,
    randoms: UnsafePointer[Float32]
    ) -> Number[W, DT]:
    var dt = T / nTimeSteps
    var path = s0
    var sdt = dt.sqrt()
    var vsdt = sigma*sdt
    var drift = (r-((sigma**2)/2))*dt
    for i in range(nTimeSteps):
        path = path * (drift + vsdt * randoms.load[width=W](i*W).cast[DT]()).expon()
    return path


fn EuropeanCallOption[W: Int, DT: DType](
    s0: Number[W, DT],
    r: Number[W, DT],
    sigma: Number[W, DT],
    T: Number[W, DT],
    K: Number[W, DT],
    nTimeSteps: Int,
    nPaths: Int,
    randoms: UnsafePointer[Float32]
    ) -> Number[W, DT]:

    var tape = s0.tape
    var psum = Number[W, DT](0.0, tape)

    for i in range(nPaths):
        var path = MonteCarloPathSimul(s0, r, sigma, T, nTimeSteps, randoms + i*nTimeSteps)
        var payoff = (path-K) * max_mask[W, DT](path-K, 0) * (-r * T).expon()
        # print(payoff.value)
        payoff.propagate_adjoints((1/nPaths).cast[DT]())
        tape.rewind_to_mark()
        psum = psum + payoff

    return psum / nPaths


fn main():
    var tape = Tape[width, type]()
    var s0 = Number[width, type](100.0, tape)
    var r = Number[width, type](0.01, s0.tape)
    var sigma = Number[width, type](0.2, s0.tape)
    var T = Number[width, type](1.0, s0.tape)
    var K = Number[width, type](100.0, s0.tape)
    tape.set_mark()
    var nTimeSteps = 1
    var nPaths = 200000
    var randoms = UnsafePointer[Float32].alloc(nTimeSteps*nPaths)
    random.randn[type](randoms, nTimeSteps*nPaths)

    var optionPrice = EuropeanCallOption(s0, r, sigma, T, K, nTimeSteps, nPaths, randoms)
    print(optionPrice.mean(), tape.get_tail(), s0.mean_grad(), r.mean_grad(), sigma.mean_grad(), T.mean_grad(), K.mean_grad())
    # randoms.free()