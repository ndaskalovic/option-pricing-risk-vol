//! Kernel graphs and the code generator behind `raad`.
//!
//! A kernel is a branch-free function traced (or parsed) into a [`Graph`]: inputs, data,
//! constants and operations in evaluation order, plus the output nodes. The passes in
//! [`passes`] decide which nodes need adjoints and which forward values the reverse sweep
//! reads; [`emit`] turns the graph into straight-line Rust generic over `R: raad::Real`.

pub mod emit;
pub mod passes;

pub use emit::{Group, Signature};

/// Operand slot that is not used.
pub const NONE: usize = usize::MAX;

/// Node operation. `Const` and `Powk` carry Rust source of an `f64` expression, so the
/// macro front end can keep e.g. `Self::EPS` symbolic; the tracer writes literals.
#[derive(Clone, Debug, PartialEq)]
pub enum Op {
    /// Differentiable input number `i` (counted over input groups in signature order).
    Input(usize),
    /// Non-differentiable input number `i` (counted over data groups).
    Data(usize),
    Const(String),
    Add,
    Sub,
    Mul,
    Div,
    Neg,
    Exp,
    Ln,
    Sqrt,
    /// `a ** k`
    Powk(String),
    /// `max(a, 0)`
    Relu,
    /// `a > b ? c : d`; no derivative flows into `a`, `b`.
    SelGt,
    /// value of `a`, zero derivative
    Stop,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Node {
    pub op: Op,
    pub a: usize,
    pub b: usize,
    pub c: usize,
    pub d: usize,
}

impl Node {
    pub fn new(op: Op, a: usize, b: usize, c: usize, d: usize) -> Self {
        Node { op, a, b, c, d }
    }
    /// Operand indices actually used.
    pub fn args(&self) -> impl Iterator<Item = usize> + '_ {
        [self.a, self.b, self.c, self.d].into_iter().filter(|&i| i != NONE)
    }
    pub fn is_leaf(&self) -> bool {
        matches!(self.op, Op::Input(_) | Op::Data(_) | Op::Const(_))
    }
}

/// Nodes in evaluation order (operands always precede their users) and the output nodes.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Graph {
    pub nodes: Vec<Node>,
    pub outs: Vec<usize>,
}

impl Graph {
    pub fn push(&mut self, op: Op, a: usize, b: usize, c: usize, d: usize) -> usize {
        debug_assert!([a, b, c, d].iter().all(|&i| i == NONE || i < self.nodes.len()));
        self.nodes.push(Node::new(op, a, b, c, d));
        self.nodes.len() - 1
    }
    pub fn len(&self) -> usize {
        self.nodes.len()
    }
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }
}

/// Rust source for an `f64` literal that round-trips exactly.
pub fn f64_literal(k: f64) -> String {
    if k.is_nan() {
        "f64::NAN".into()
    } else if k.is_infinite() {
        if k > 0.0 { "f64::INFINITY".into() } else { "f64::NEG_INFINITY".into() }
    } else {
        let s = format!("{k:?}");
        if s.contains(['.', 'e', 'E']) { format!("{s}_f64") } else { format!("{s}.0_f64") }
    }
}
