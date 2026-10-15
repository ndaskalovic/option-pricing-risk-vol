//! Build-time tracer + reverse-mode code generator (used only by build.rs).
use crate::adnum::AdNum;
use std::cell::RefCell;
use std::fmt::Write;
use std::ops::{Add, Div, Mul, Neg, Sub};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Op { Input(usize), Data(usize), Const(f64), Add, Sub, Mul, Div, Neg, Exp, Ln, Sqrt, Relu }

#[derive(Clone, Copy, Debug)]
pub struct Node { pub op: Op, pub a: usize, pub b: usize }

thread_local! { static G: RefCell<Vec<Node>> = RefCell::new(Vec::new()); }

#[derive(Clone, Copy)]
pub struct T(pub usize);

fn push(op: Op, a: usize, b: usize) -> T {
    G.with(|g| { let mut g = g.borrow_mut(); g.push(Node { op, a, b }); T(g.len() - 1) })
}
fn k(v: f64) -> T { push(Op::Const(v), usize::MAX, usize::MAX) }

impl Add for T { type Output = T; fn add(self, r: T) -> T { push(Op::Add, self.0, r.0) } }
impl Sub for T { type Output = T; fn sub(self, r: T) -> T { push(Op::Sub, self.0, r.0) } }
impl Mul for T { type Output = T; fn mul(self, r: T) -> T { push(Op::Mul, self.0, r.0) } }
impl Div for T { type Output = T; fn div(self, r: T) -> T { push(Op::Div, self.0, r.0) } }
impl Neg for T { type Output = T; fn neg(self) -> T { push(Op::Neg, self.0, usize::MAX) } }
impl Add<f64> for T { type Output = T; fn add(self, r: f64) -> T { self + k(r) } }
impl Sub<f64> for T { type Output = T; fn sub(self, r: f64) -> T { self - k(r) } }
impl Mul<f64> for T { type Output = T; fn mul(self, r: f64) -> T { self * k(r) } }
impl Div<f64> for T { type Output = T; fn div(self, r: f64) -> T { self * k(1.0 / r) } }
impl AdNum for T {
    fn exp(self) -> T { push(Op::Exp, self.0, usize::MAX) }
    fn ln(self) -> T { push(Op::Ln, self.0, usize::MAX) }
    fn sqrt(self) -> T { push(Op::Sqrt, self.0, usize::MAX) }
    fn relu(self) -> T { push(Op::Relu, self.0, usize::MAX) }
    fn constant(self, v: f64) -> T { k(v) }
}

/// Trace a kernel with `ni` differentiable inputs and `nd` data inputs; returns (graph, outputs).
pub fn trace(ni: usize, nd: usize, f: impl Fn(&[T], &[T]) -> Vec<T>) -> (Vec<Node>, Vec<usize>) {
    G.with(|g| g.borrow_mut().clear());
    let x: Vec<T> = (0..ni).map(|i| push(Op::Input(i), usize::MAX, usize::MAX)).collect();
    let d: Vec<T> = (0..nd).map(|i| push(Op::Data(i), usize::MAX, usize::MAX)).collect();
    let y = f(&x, &d);
    (G.with(|g| g.borrow().clone()), y.iter().map(|t| t.0).collect())
}

fn is_leaf(op: Op) -> bool { matches!(op, Op::Input(_) | Op::Data(_) | Op::Const(_)) }
fn args(n: &Node) -> Vec<usize> { [n.a, n.b].into_iter().filter(|&i| i != usize::MAX).collect() }

/// Emit `<name>_fwd / _bwd / _vjp / _eval` for one kernel over the lane type `V`.
pub fn emit(out: &mut String, name: &str, g: &[Node], outs: &[usize], ni: usize, nd: usize) {
    let n = g.len();
    let mut active = vec![false; n];
    for (i, nd_) in g.iter().enumerate() {
        active[i] = match nd_.op { Op::Input(_) => true, op if is_leaf(op) => false, _ => args(nd_).iter().any(|&a| active[a]) };
    }
    let mut useful = vec![false; n];
    for &o in outs { useful[o] = true; }
    for i in (0..n).rev() { if useful[i] && !is_leaf(g[i].op) { for a in args(&g[i]) { useful[a] = true; } } }
    let need: Vec<bool> = (0..n).map(|i| active[i] && useful[i]).collect();
    let mut reads = vec![false; n];
    for i in 0..n {
        if !need[i] || is_leaf(g[i].op) { continue; }
        let (a, b) = (g[i].a, g[i].b);
        match g[i].op {
            Op::Mul => { if need[a] { reads[b] = true; } if need[b] { reads[a] = true; } }
            Op::Div => { if need[a] || need[b] { reads[b] = true; } if need[b] { reads[i] = true; } }
            Op::Exp | Op::Sqrt => reads[i] = true,
            Op::Ln | Op::Relu => reads[a] = true,
            _ => {}
        }
    }
    let res: Vec<usize> = (0..n).filter(|&i| reads[i] && !matches!(g[i].op, Op::Data(_) | Op::Const(_))).collect();
    let no = outs.len();
    let fwd_body = |s: &mut String| {
        for (i, nd_) in g.iter().enumerate() {
            let (a, b) = (nd_.a, nd_.b);
            let e = match nd_.op {
                Op::Input(j) => format!("x[{j}]"), Op::Data(j) => format!("d[{j}]"),
                Op::Const(v) => format!("V::splat({v:?} as f32)"),
                // explicit FMA: Rust never contracts a*b+c on its own (Mojo/LLVM there does)
                Op::Add if g[b].op == Op::Mul => format!("v{}.mul_add(v{}, v{a})", g[b].a, g[b].b),
                Op::Add if g[a].op == Op::Mul => format!("v{}.mul_add(v{}, v{b})", g[a].a, g[a].b),
                Op::Add => format!("v{a} + v{b}"), Op::Sub => format!("v{a} - v{b}"),
                Op::Mul => format!("v{a} * v{b}"), Op::Div => format!("v{a} / v{b}"), Op::Neg => format!("-v{a}"),
                Op::Exp => format!("v{a}.exp()"), Op::Ln => format!("v{a}.ln()"), Op::Sqrt => format!("v{a}.sqrt()"),
                Op::Relu => format!("v{a}.relu()"),
            };
            writeln!(s, "    let v{i}: V = {e};").unwrap();
        }
    };
    // Adjoint bookkeeping. A first contribution that is a product x*y is kept *pending* (not
    // materialised): a second contribution then fuses into one FMA, and if the node is an input
    // the product fuses straight into the caller's accumulator. Rust never contracts a*b+c itself.
    struct Adj { seen: Vec<bool>, pending: Vec<Option<(String, String)>> }
    impl Adj {
        fn mat(&mut self, s: &mut String, j: usize) {
            if let Some((x, y)) = self.pending[j].take() { writeln!(s, "    let mut a{j}: V = {x} * {y};").unwrap(); }
        }
        fn add_mul(&mut self, s: &mut String, need: &[bool], j: usize, x: String, y: String) {
            if !need[j] { return; }
            if !self.seen[j] { self.seen[j] = true; self.pending[j] = Some((x, y)); return; }
            self.mat(s, j);
            writeln!(s, "    a{j} = {x}.mul_add({y}, a{j});").unwrap();
        }
        fn add(&mut self, s: &mut String, need: &[bool], j: usize, e: String, neg: bool) {
            if !need[j] { return; }
            if !self.seen[j] { self.seen[j] = true; writeln!(s, "    let mut a{j}: V = {}{e};", if neg { "-" } else { "" }).unwrap(); return; }
            self.mat(s, j);
            writeln!(s, "    a{j} {} {e};", if neg { "-=" } else { "+=" }).unwrap();
        }
    }
    let rev_body = |s: &mut String| {
        let mut ad = Adj { seen: vec![false; n], pending: vec![None; n] };
        for (o, &i) in outs.iter().enumerate() { ad.add(s, &need, i, format!("a_y[{o}]"), false); }
        for i in (0..n).rev() {
            if !need[i] || !ad.seen[i] { continue; }
            let (a, b) = (g[i].a, g[i].b);
            if let Op::Input(j) = g[i].op {
                match ad.pending[i].take() {
                    Some((x, y)) => writeln!(s, "    a_x[{j}] = {x}.mul_add({y}, a_x[{j}]);").unwrap(),
                    None => writeln!(s, "    a_x[{j}] += a{i};").unwrap(),
                }
                continue;
            }
            ad.mat(s, i); // a{i} is used below
            match g[i].op {
                Op::Add => { ad.add(s, &need, a, format!("a{i}"), false); ad.add(s, &need, b, format!("a{i}"), false); }
                Op::Sub => { ad.add(s, &need, a, format!("a{i}"), false); ad.add(s, &need, b, format!("a{i}"), true); }
                Op::Mul => { ad.add_mul(s, &need, a, format!("a{i}"), format!("v{b}")); ad.add_mul(s, &need, b, format!("a{i}"), format!("v{a}")); }
                Op::Div => { ad.add(s, &need, a, format!("a{i} / v{b}"), false); ad.add(s, &need, b, format!("a{i} * v{i} / v{b}"), true); }
                Op::Neg => ad.add(s, &need, a, format!("a{i}"), true),
                Op::Exp => ad.add_mul(s, &need, a, format!("a{i}"), format!("v{i}")),
                Op::Ln => ad.add(s, &need, a, format!("a{i} / v{a}"), false),
                Op::Sqrt => ad.add(s, &need, a, format!("a{i} * V::splat(0.5) / v{i}"), false),
                Op::Relu => ad.add(s, &need, a, format!("v{a}.relu_mask(a{i})"), false),
                _ => {}
            }
        }
    };
    let outs_expr = outs.iter().map(|o| format!("v{o}")).collect::<Vec<_>>().join(", ");
    writeln!(out, "pub const {}_NRES: usize = {};", name.to_uppercase(), res.len()).unwrap();
    // eval
    writeln!(out, "#[inline(always)] #[allow(unused_variables, clippy::all)]\npub fn {name}_eval(x: &[V; {ni}], d: &[V; {nd}]) -> [V; {no}] {{").unwrap();
    fwd_body(out); writeln!(out, "    [{outs_expr}]\n}}").unwrap();
    // fwd
    writeln!(out, "#[inline(always)] #[allow(unused_variables, clippy::all)]\npub fn {name}_fwd(x: &[V; {ni}], d: &[V; {nd}], res: &mut [V]) -> [V; {no}] {{").unwrap();
    fwd_body(out);
    for (k_, r) in res.iter().enumerate() { writeln!(out, "    res[{k_}] = v{r};").unwrap(); }
    writeln!(out, "    [{outs_expr}]\n}}").unwrap();
    // bwd
    writeln!(out, "#[inline(always)] #[allow(unused_variables, unused_mut, clippy::all)]\npub fn {name}_bwd(d: &[V; {nd}], res: &[V], a_y: &[V; {no}], a_x: &mut [V; {ni}]) {{").unwrap();
    for (k_, r) in res.iter().enumerate() { writeln!(out, "    let v{r}: V = res[{k_}];").unwrap(); }
    for (i, nd_) in g.iter().enumerate() {
        match nd_.op {
            Op::Data(j) if reads[i] => writeln!(out, "    let v{i}: V = d[{j}];").unwrap(),
            Op::Const(v) if reads[i] => writeln!(out, "    let v{i}: V = V::splat({v:?} as f32);").unwrap(),
            _ => {}
        }
    }
    rev_body(out); writeln!(out, "}}").unwrap();
    // vjp (fused)
    writeln!(out, "#[inline(always)] #[allow(unused_variables, unused_mut, clippy::all)]\npub fn {name}_vjp(x: &[V; {ni}], d: &[V; {nd}], a_y: &[V; {no}], a_x: &mut [V; {ni}]) -> [V; {no}] {{").unwrap();
    fwd_body(out); rev_body(out); writeln!(out, "    [{outs_expr}]\n}}").unwrap();
}
