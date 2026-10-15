//! The supported-operations API (same op set as the Mojo `ADNum`).
use std::ops::{Add, Div, Mul, Neg, Sub};

pub trait AdNum:
    Copy
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
    + Add<f64, Output = Self>
    + Sub<f64, Output = Self>
    + Mul<f64, Output = Self>
    + Div<f64, Output = Self>
{
    fn exp(self) -> Self;
    fn ln(self) -> Self;
    fn sqrt(self) -> Self;
    fn relu(self) -> Self;
    fn constant(self, k: f64) -> Self;
}
