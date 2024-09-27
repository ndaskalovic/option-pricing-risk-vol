# Option Pricing, Risk and Volatility Modelling

This repository contains some of the code I developed as part of my master's thesis which studied the use of Adjoint Automatic differentiation (AAD) for calculating the Greeks on options priced using Monte Carlo under the Black Scholes model. It explores the benefits of vectorised computations for this task and shows the speedup that can be gained.

[My implementation of AAD](autodiff) uses a Tape structure to hold Nodes representing mathematical operations carried out in the program. This is a simple way of representing a computation graph DAG and is all abstracted behind the Number interface. Calculating adjoints is as easy as performing the backpropagation algorithm to accumulate them using the chain rule.

I thought the [Mojo]() programming language might be well suited to this task as it provides easy access to SIMD primitives which compile to CPU-native instructions.

As an extension to these initial results, I implemented a simply SVI volatility model and calculated the sensitivities of its parameters using AAD.

## Speeding up Monte Carlo Option Pricing

I priced a vanilla European call using the Black Scholes model and 200,000 Monte Carlo paths across weekly time steps for the sake of complexity. The test was ran on a Ryzen 9 5900x which has AVX2, so theoretically space for 8 32-bit floats in its registers. The actual speed up achieved is **6.7x**.


![](simd_speedup_raw.png)

## Fast Risk

Optimisations such as [pathwise adjoints]() are implemented to reduced memory consumption and improve performance. Similar overall performance improvemetns can be seen when pricing the European option and calculating the first-order Greeks (Delta, Rho, Theta, Vega). A maximum **8.1x** speed up is achieved here.

![](simd_speedup_aad.png)

The simulation which includes adjoint calculations exhibits a 3-4x slow down due to these extra calculations as well as memory access overhead.

## Some Volatility Modelling and Sensitivities

SVI volatility model from Apple stock data. Fit curve and get polynomial. Use it in a function to get volatility at each timestep when computing monte carlo paths.

## References

SVI paper
Glasserman pathwise adjoints
Antoine Savine book