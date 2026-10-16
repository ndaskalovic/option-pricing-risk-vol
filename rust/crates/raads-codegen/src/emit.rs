//! Straight-line Rust for one kernel graph, generic over `R: ::raads::Real`.
//!
//! Four functions per kernel, all `#[inline(always)]`:
//! - `eval(inputs..) -> y`
//! - `vjp(inputs.., a_y, a_inputs..) -> y`: primal plus `a_inputs += Jᵀ a_y`
//! - `fwd(inputs.., res) -> y`: primal, saving the residuals `bwd` reads
//! - `bwd(data.., res, a_y, a_inputs..)`: `a_inputs += Jᵀ a_y` from residuals
//!
//! Rust never contracts `a * b + c`, so products are fused into `mul_add` explicitly:
//! in the forward pass when an `Add`/`Sub` consumes a `Mul`, and in the reverse pass a
//! product contribution stays pending until a second contribution (or the caller's
//! accumulator) can absorb it as one FMA.

use crate::passes::Analysis;
use crate::{Graph, NONE, Op};
use std::fmt::Write;

/// One parameter of the kernel: `name: R` (len `None`) or `name: &[R; len]`.
#[derive(Clone, Debug)]
pub struct Group {
    pub name: String,
    pub len: Option<usize>,
    /// Non-differentiable (no adjoint argument).
    pub data: bool,
}

/// Parameter groups in signature order and the output shape (`None`: a single `R`).
#[derive(Clone, Debug)]
pub struct Signature {
    pub groups: Vec<Group>,
    pub out: Option<usize>,
}

impl Signature {
    fn param(&self, g: &Group) -> String {
        match g.len {
            None => format!("{}: R", g.name),
            Some(n) => format!("{}: &[R; {n}]", g.name),
        }
    }
    fn adj_param(&self, g: &Group) -> String {
        match g.len {
            None => format!("a_{}: &mut R", g.name),
            Some(n) => format!("a_{}: &mut [R; {n}]", g.name),
        }
    }
    fn out_ty(&self) -> String {
        match self.out {
            None => "R".into(),
            Some(n) => format!("[R; {n}]"),
        }
    }
    fn a_y_param(&self) -> String {
        match self.out {
            None => "a_y: R".into(),
            Some(n) => format!("a_y: &[R; {n}]"),
        }
    }
    fn a_y(&self, o: usize) -> String {
        match self.out {
            None => "a_y".into(),
            Some(_) => format!("a_y[{o}]"),
        }
    }
    /// (value access, adjoint access) of differentiable input `i`, or data input `i`.
    fn slot(&self, i: usize, data: bool) -> (String, String) {
        let mut k = i;
        for g in self.groups.iter().filter(|g| g.data == data) {
            let n = g.len.unwrap_or(1);
            if k < n {
                return match g.len {
                    None => (g.name.clone(), format!("*a_{}", g.name)),
                    Some(_) => (format!("{}[{k}]", g.name), format!("a_{}[{k}]", g.name)),
                };
            }
            k -= n;
        }
        panic!("input {i} (data: {data}) outside the signature");
    }
}

/// Which of the four functions to emit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Eval,
    Vjp,
    Fwd,
    Bwd,
}

/// Number of residuals `fwd` saves.
pub fn nres(g: &Graph) -> usize {
    Analysis::new(g).res.len()
}

/// Rust source of one function `name` (with visibility/qualifiers `vis`, e.g. `"pub"`).
pub fn function(g: &Graph, sig: &Signature, kind: Kind, vis: &str, name: &str) -> String {
    let an = Analysis::new(g);
    let e = Emitter { g, sig, an: &an };
    let mut params: Vec<String> = Vec::new();
    match kind {
        Kind::Eval | Kind::Vjp | Kind::Fwd => params.extend(sig.groups.iter().map(|g| sig.param(g))),
        Kind::Bwd => params.extend(sig.groups.iter().filter(|g| g.data).map(|g| sig.param(g))),
    }
    if kind == Kind::Fwd {
        params.push("res: &mut [R]".into());
    }
    if kind == Kind::Bwd {
        params.push("res: &[R]".into());
    }
    if matches!(kind, Kind::Vjp | Kind::Bwd) {
        params.push(sig.a_y_param());
        params.extend(sig.groups.iter().filter(|g| !g.data).map(|g| sig.adj_param(g)));
    }
    let ret = if kind == Kind::Bwd { String::new() } else { format!(" -> {}", sig.out_ty()) };
    let doc = match kind {
        Kind::Eval => "The kernel's result.",
        Kind::Vjp => "The kernel's result; adds `Jᵀ a_y` to the input adjoints.",
        Kind::Fwd => "The kernel's result; saves the residuals `bwd` reads to `res`.",
        Kind::Bwd => "Adds `Jᵀ a_y` to the input adjoints, from the residuals `fwd` saved.",
    };
    let mut s = String::new();
    writeln!(
        s,
        "#[doc = {doc:?}]\n#[inline(always)]\n#[allow(unused_variables, unused_mut, clippy::all)]\n{vis} fn {name}<R: ::raads::Real>({}){ret} {{",
        params.join(", ")
    )
    .unwrap();
    match kind {
        Kind::Eval => {
            e.forward(&mut s);
            e.outputs(&mut s);
        }
        Kind::Vjp => {
            e.forward(&mut s);
            e.reverse(&mut s);
            e.outputs(&mut s);
        }
        Kind::Fwd => {
            e.forward(&mut s);
            if !an.res.is_empty() {
                writeln!(s, "    let res = &mut res[..{}];", an.res.len()).unwrap();
            }
            for (k, r) in an.res.iter().enumerate() {
                writeln!(s, "    res[{k}] = v{r};").unwrap();
            }
            e.outputs(&mut s);
        }
        Kind::Bwd => {
            if !an.res.is_empty() {
                writeln!(s, "    let res = &res[..{}];", an.res.len()).unwrap();
            }
            for (k, r) in an.res.iter().enumerate() {
                writeln!(s, "    let v{r}: R = res[{k}];").unwrap();
            }
            for (i, n) in g.nodes.iter().enumerate() {
                if let Op::Data(j) = n.op
                    && an.reads[i]
                {
                    writeln!(s, "    let v{i}: R = {};", sig.slot(j, true).0).unwrap();
                }
            }
            e.reverse(&mut s);
        }
    }
    s.push_str("}\n");
    s
}

struct Emitter<'a> {
    g: &'a Graph,
    sig: &'a Signature,
    an: &'a Analysis,
}

impl Emitter<'_> {
    fn konst(&self, i: usize) -> Option<&str> {
        match &self.g.nodes[i].op {
            Op::Const(k) => Some(k),
            _ => None,
        }
    }
    /// Value of node `i` as an `R` expression (constants splatted inline).
    fn r(&self, i: usize) -> String {
        match self.konst(i) {
            Some(k) => format!("R::splat({k})"),
            None => format!("v{i}"),
        }
    }
    /// `x * y` as a pair for `mul_add`, if node `i` is a product.
    fn product(&self, i: usize) -> Option<(String, String)> {
        let n = &self.g.nodes[i];
        (n.op == Op::Mul).then(|| (self.r(n.a), self.r(n.b)))
    }

    fn forward(&self, s: &mut String) {
        for (i, n) in self.g.nodes.iter().enumerate() {
            if !self.an.live[i] {
                continue;
            }
            let (a, b) = (n.a, n.b);
            let ka = if a != NONE { self.konst(a) } else { None };
            let kb = if b != NONE { self.konst(b) } else { None };
            let e = match &n.op {
                Op::Input(j) => self.sig.slot(*j, false).0,
                Op::Data(j) => self.sig.slot(*j, true).0,
                Op::Const(k) => format!("R::splat({k})"),
                Op::Add => match (ka, kb, self.product(a), self.product(b)) {
                    (_, Some(k), _, _) => format!("v{a} + ({k})"),
                    (Some(k), _, _, _) => format!("v{b} + ({k})"),
                    (_, _, _, Some((x, y))) => format!("{x}.mul_add({y}, v{a})"),
                    (_, _, Some((x, y)), _) => format!("{x}.mul_add({y}, v{b})"),
                    _ => format!("v{a} + v{b}"),
                },
                Op::Sub => match (ka, kb, self.product(b)) {
                    (_, Some(k), _) => format!("v{a} - ({k})"),
                    (Some(k), _, _) => format!("v{b}.rsub({k})"),
                    (_, _, Some((x, y))) => format!("{x}.nmul_add({y}, v{a})"),
                    _ => format!("v{a} - v{b}"),
                },
                Op::Mul => match (ka, kb) {
                    (_, Some(k)) => format!("v{a} * ({k})"),
                    (Some(k), _) => format!("v{b} * ({k})"),
                    _ => format!("v{a} * v{b}"),
                },
                Op::Div => match (ka, kb) {
                    (_, Some(k)) => format!("v{a} / ({k})"),
                    (Some(k), _) => format!("v{b}.rdiv({k})"),
                    _ => format!("v{a} / v{b}"),
                },
                Op::Neg => format!("-v{a}"),
                Op::Exp => format!("v{a}.exp()"),
                Op::Ln => format!("v{a}.ln()"),
                Op::Sqrt => format!("v{a}.sqrt()"),
                Op::Powk(k) => format!("v{a}.powf({k})"),
                Op::Relu => format!("v{a}.relu()"),
                Op::SelGt => format!("R::select_gt({}, {}, {}, {})", self.r(a), self.r(b), self.r(n.c), self.r(n.d)),
                Op::Stop => format!("v{a}.detach()"),
            };
            writeln!(s, "    let v{i}: R = {e};").unwrap();
        }
    }

    fn outputs(&self, s: &mut String) {
        let vals: Vec<String> = self.g.outs.iter().map(|&o| format!("v{o}")).collect();
        match self.sig.out {
            None => writeln!(s, "    {}", vals[0]).unwrap(),
            Some(_) => writeln!(s, "    [{}]", vals.join(", ")).unwrap(),
        }
    }

    fn reverse(&self, s: &mut String) {
        let (g, need) = (self.g, &self.an.need);
        let mut ad = Adj { seen: vec![false; g.len()], pending: vec![None; g.len()], need };
        for (o, &i) in g.outs.iter().enumerate() {
            ad.add(s, i, self.sig.a_y(o), false);
        }
        for i in (0..g.len()).rev() {
            if !need[i] || !ad.seen[i] {
                continue;
            }
            let n = &g.nodes[i];
            let (a, b) = (n.a, n.b);
            if let Op::Input(j) = n.op {
                let acc = self.sig.slot(j, false).1;
                match ad.pending[i].take() {
                    Some((x, y)) => writeln!(s, "    {acc} = {x}.mul_add({y}, {acc});").unwrap(),
                    None => writeln!(s, "    {acc} += a{i};").unwrap(),
                }
                continue;
            }
            ad.materialise(s, i);
            let ai = format!("a{i}");
            match &n.op {
                Op::Add => {
                    ad.add(s, a, ai.clone(), false);
                    ad.add(s, b, ai, false);
                }
                Op::Sub => {
                    ad.add(s, a, ai.clone(), false);
                    ad.add(s, b, ai, true);
                }
                Op::Mul => {
                    ad.add_mul(s, a, ai.clone(), self.r(b));
                    ad.add_mul(s, b, ai, self.r(a));
                }
                Op::Div => {
                    ad.add(s, a, format!("{ai} / {}", self.r(b)), false);
                    ad.add(s, b, format!("{ai} * v{i} / {}", self.r(b)), true);
                }
                Op::Neg => ad.add(s, a, ai, true),
                Op::Exp => ad.add_mul(s, a, ai, format!("v{i}")),
                Op::Ln => ad.add(s, a, format!("{ai} / v{a}"), false),
                Op::Sqrt => ad.add(s, a, format!("{ai} * 0.5_f64 / v{i}"), false),
                Op::Powk(k) => ad.add_mul(s, a, ai, format!("(v{a}.powf(({k}) - 1.0_f64) * ({k}))")),
                Op::Relu => ad.add(s, a, format!("R::select_gt(v{a}, R::splat(0.0_f64), {ai}, R::splat(0.0_f64))"), false),
                Op::SelGt => {
                    let (va, vb) = (self.r(a), self.r(b));
                    ad.add(s, n.c, format!("R::select_gt({va}, {vb}, {ai}, R::splat(0.0_f64))"), false);
                    ad.add(s, n.d, format!("R::select_gt({va}, {vb}, R::splat(0.0_f64), {ai})"), false);
                }
                Op::Input(_) | Op::Data(_) | Op::Const(_) | Op::Stop => {}
            }
        }
    }
}

/// Adjoint bookkeeping: first touch defines `a{j}`; a first contribution that is a
/// product stays pending so the next one (or the input accumulator) fuses it into an FMA.
struct Adj<'a> {
    seen: Vec<bool>,
    pending: Vec<Option<(String, String)>>,
    need: &'a [bool],
}

impl Adj<'_> {
    fn materialise(&mut self, s: &mut String, j: usize) {
        if let Some((x, y)) = self.pending[j].take() {
            writeln!(s, "    let mut a{j}: R = {x} * {y};").unwrap();
        }
    }
    fn add_mul(&mut self, s: &mut String, j: usize, x: String, y: String) {
        if !self.need[j] {
            return;
        }
        if !self.seen[j] {
            self.seen[j] = true;
            self.pending[j] = Some((x, y));
            return;
        }
        self.materialise(s, j);
        writeln!(s, "    a{j} = {x}.mul_add({y}, a{j});").unwrap();
    }
    fn add(&mut self, s: &mut String, j: usize, e: String, neg: bool) {
        if !self.need[j] {
            return;
        }
        if !self.seen[j] {
            self.seen[j] = true;
            writeln!(s, "    let mut a{j}: R = {}{e};", if neg { "-" } else { "" }).unwrap();
            return;
        }
        if neg {
            if let Some((x, y)) = self.pending[j].take() {
                writeln!(s, "    let mut a{j}: R = {x} * {y} - ({e});").unwrap();
            } else {
                writeln!(s, "    a{j} -= {e};").unwrap();
            }
        } else if let Some((x, y)) = self.pending[j].take() {
            writeln!(s, "    let mut a{j}: R = {x}.mul_add({y}, {e});").unwrap();
        } else {
            writeln!(s, "    a{j} += {e};").unwrap();
        }
    }
}
