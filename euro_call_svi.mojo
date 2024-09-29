from autodiff import Number, Tape
import random
from vanilla_euro_call import MonteCarloPathSimul, max_mask

alias width = 8
alias type = DType.float32

fn EuropeanCallOption[W: Int, DT: DType](
    s0: Number[W, DT],
    r: Number[W, DT],
    T: Number[W, DT],
    K: Number[W, DT],
    a: Number[W, DT],
    b: Number[W, DT],
    rho: Number[W, DT],
    m: Number[W, DT],
    sigma: Number[W, DT],
    nTimeSteps: Int,
    nPaths: Int,
    randoms: UnsafePointer[Float32]
    ) -> Number[W, DT]:

    var tape = s0.tape
    var psum = Number[W, DT](0.0, tape)

    for i in range(nPaths):
        # vol calculated using SVI model
        var vol = svi_vol[W, DT](a, b, rho, m, sigma, T, K)
        var path = MonteCarloPathSimul(s0, r, vol, T, nTimeSteps, randoms + i*nTimeSteps)
        var payoff = (path-K) * max_mask[W, DT](path-K, 0) * (-r * T).expon()
        payoff.propagate_adjoints((1/(nPaths)).cast[DT]())
        tape.rewind_to_mark()
        psum = psum + payoff

    return psum / nPaths

fn svi_vol[width: Int, type: DType](
    a: Number[width, type],
    b: Number[width, type],
    rho: Number[width, type],
    m: Number[width, type],
    sigma: Number[width, type],
    T: Number[width, type],
    K: Number[width, type]
    ) -> Number[width, type]:
    var w = (a + b*(rho*(K-m)+((K-m)**2+sigma**2)**0.5))
    return (w/T.value).sqrt()


fn main():
    tape = Tape[width, type]()
    s0 = Number[width, type](227.7899932861328, tape)
    r = Number[width, type](0.04, tape)
    k = Number[width, type](220, tape)
    T = Number[width, type](0.7287671233, tape)

    a = Number[width, type](5.73431376e-02, tape)
    b = Number[width, type](6.59987208e-04, tape)
    rho = Number[width, type](-1.23157254, tape)
    m = Number[width, type](2.22873847e+02, tape)
    sigma = Number[width, type](9.99943135e-02, tape)
    
    tape.set_mark()

    nTimeSteps = 52
    nPaths = 200000//width
    randoms = UnsafePointer[Float32].alloc(nTimeSteps*nPaths)
    random.randn[DType.float32](randoms, nTimeSteps*nPaths)

    price = EuropeanCallOption[width, type](s0, r, T, k, a, b, rho, m, sigma, nTimeSteps, nPaths, randoms)
    print("\nPrice: ", price.mean())
    print("Delta: ", s0.mean_grad())
    print("Theta: ", T.mean_grad())
    print("Rho:   ", r.mean_grad())
    print("\nSVI params sensitivities:")
    print("a:     ", a.mean_grad())
    print("b:     ", b.mean_grad())
    print("rho:   ", rho.mean_grad())
    print("m:     ", m.mean_grad())
    print("sigma: ", sigma.mean_grad())
