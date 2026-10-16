//! Model attributes: compile each kernel of an `impl` block and implement the engine trait.

use crate::Compiled;
use crate::front::Shape;
use proc_macro2::{Span, TokenStream};
use quote::quote;
use syn::spanned::Spanned;
use syn::{Error, GenericParam, ImplItem, ImplItemFn, ItemFn, ItemImpl, Result};

/// Expected shape of a kernel parameter or result: `N`, `[N; k]`, or `[N; DIM]` with a
/// dimension shared across the kernels.
#[derive(Clone, Copy)]
enum Dim {
    Scalar,
    Fixed(usize),
    Var(&'static str),
}

/// Expected signature of one kernel: parameters (name, shape, `#[data]`) and result.
struct Spec {
    name: &'static str,
    params: &'static [(&'static str, Dim, bool)],
    ret: Dim,
}

const MC: [Spec; 4] = [
    Spec { name: "setup", params: &[("p", Dim::Var("NP"), false), ("n_steps", Dim::Scalar, true)], ret: Dim::Var("NC") },
    Spec { name: "init", params: &[("p", Dim::Var("NP"), false)], ret: Dim::Var("NS") },
    Spec {
        name: "step",
        params: &[("s", Dim::Var("NS"), false), ("c", Dim::Var("NC"), false), ("z", Dim::Var("NZ"), true)],
        ret: Dim::Var("NS"),
    },
    Spec {
        name: "payoff",
        params: &[("s", Dim::Var("NS"), false), ("p", Dim::Var("NP"), false), ("c", Dim::Var("NC"), false)],
        ret: Dim::Scalar,
    },
];

const FD: [Spec; 4] = [
    Spec { name: "coeffs", params: &[("x", Dim::Scalar, true), ("p", Dim::Var("NP"), false)], ret: Dim::Fixed(3) },
    Spec { name: "payoff", params: &[("x", Dim::Scalar, true), ("p", Dim::Var("NP"), false)], ret: Dim::Scalar },
    Spec { name: "spot", params: &[("p", Dim::Var("NP"), false)], ret: Dim::Scalar },
    Spec { name: "maturity", params: &[("p", Dim::Var("NP"), false)], ret: Dim::Scalar },
];

fn dim_str(d: Dim) -> String {
    match d {
        Dim::Scalar => "N".into(),
        Dim::Fixed(k) => format!("[N; {k}]"),
        Dim::Var(v) => format!("[N; {v}]"),
    }
}

impl Spec {
    fn signature(&self) -> String {
        let ps: Vec<String> =
            self.params.iter().map(|(n, d, data)| format!("{}{n}: {}", if *data { "#[data] " } else { "" }, dim_str(*d))).collect();
        format!("`{}({}) -> {}`", self.name, ps.join(", "), dim_str(self.ret))
    }
}

/// Kernels of the block in `specs` order, compiled; other items are kept as they are.
fn compile_all(imp: &ItemImpl, specs: &[Spec], krate: &TokenStream) -> Result<Vec<Compiled>> {
    if imp.trait_.is_some() {
        return Err(Error::new(imp.span(), "put the model attribute on an inherent `impl Model { ... }` block"));
    }
    // the generated methods are generic over `R`
    if let Some(p) = imp.generics.params.iter().find(|p| matches!(p, GenericParam::Type(t) if t.ident == "R")) {
        return Err(Error::new(p.span(), "the generic parameter name `R` is reserved by the generated code"));
    }
    let mut out = Vec::new();
    let mut errors: Option<Error> = None;
    for spec in specs {
        let found = imp.items.iter().find_map(|it| match it {
            ImplItem::Fn(f) if f.sig.ident == spec.name => Some(f),
            _ => None,
        });
        let r = match found {
            Some(f) => compile(spec.name, to_item_fn(f), krate),
            None => Err(Error::new(imp.self_ty.span(), format!("missing kernel {}", spec.signature()))),
        };
        match r {
            Ok(c) => out.push(c),
            Err(e) => match &mut errors {
                Some(all) => all.combine(e),
                None => errors = Some(e),
            },
        }
    }
    errors.map_or(Ok(out), Err)
}

/// Check every kernel against its spec, reporting each mismatch at the offending type;
/// returns the shared dimensions.
fn check(specs: &[Spec], ks: &[Compiled]) -> Result<Vec<(&'static str, usize)>> {
    // dimension -> (length, kernel that fixed it)
    let mut dims: Vec<(&'static str, usize, &'static str)> = Vec::new();
    let mut errors: Vec<Error> = Vec::new();
    for (spec, k) in specs.iter().zip(ks) {
        let sig = spec.signature();
        if k.params.len() != spec.params.len() {
            errors.push(Error::new(k.source.sig.paren_token.span.join(), format!("expected {sig}")));
            continue;
        }
        let mut shape = |d: Dim, s: Shape, span: Span| {
            let found = match (d, s) {
                (Dim::Scalar, Shape::Scalar) => return,
                (Dim::Fixed(k), Shape::Array(n)) if k == n => return,
                (Dim::Var(v), Shape::Array(n)) => match dims.iter().find(|(w, ..)| *w == v) {
                    None => return dims.push((v, n, spec.name)),
                    Some(&(_, m, _)) if m == n => return,
                    Some(&(_, m, from)) => format!("`[N; {m}]` ({v}, as in `{from}`)"),
                },
                _ => format!("`{}`", dim_str(d)),
            };
            errors.push(Error::new(span, format!("expected {found} in {sig}")));
        };
        for (&(_, d, _), p) in spec.params.iter().zip(&k.params) {
            shape(d, p.shape, p.span);
        }
        shape(spec.ret, k.ret, k.ret_span);
        for (&(name, _, data), p) in spec.params.iter().zip(&k.params) {
            if p.data != data {
                let msg = if data { "must be `#[data]`" } else { "must not be `#[data]`" };
                errors.push(Error::new(p.span, format!("`{name}` {msg} in {sig}")));
            }
        }
    }
    match errors.into_iter().reduce(|mut a, b| {
        a.combine(b);
        a
    }) {
        Some(e) => Err(e),
        None => Ok(dims.into_iter().map(|(v, n, _)| (v, n)).collect()),
    }
}

fn to_item_fn(f: &ImplItemFn) -> ItemFn {
    ItemFn { attrs: f.attrs.clone(), vis: f.vis.clone(), sig: f.sig.clone(), block: Box::new(f.block.clone()) }
}

/// Replace each kernel in the block by its rewritten (generic-valid) source.
fn replace_sources(imp: &mut ItemImpl, compiled: &[Compiled]) {
    for item in imp.items.iter_mut() {
        if let ImplItem::Fn(f) = item
            && let Some(c) = compiled.iter().find(|c| f.sig.ident == c.source.sig.ident)
        {
            f.sig = c.source.sig.clone();
            f.block = (*c.source.block).clone();
        }
    }
}

fn compile(name: &str, f: ItemFn, krate: &TokenStream) -> Result<Compiled> {
    let n = |k: &str| format!("{name}_{k}");
    crate::compile(f, "", [&n("eval"), &n("vjp"), &n("fwd"), &n("bwd")], krate)
}

fn dim(dims: &[(&str, usize)], v: &str) -> usize {
    dims.iter().find(|(w, _)| *w == v).unwrap().1
}

// The trait impls name the GAT parameter `__RaadsT` (the impl's own generics may use `T`)
// and hide the generated kernel functions, which are engine internals.

pub fn mc(mut imp: ItemImpl) -> Result<TokenStream> {
    let krate = crate::krate();
    let c = compile_all(&imp, &MC, &krate)?;
    let dims = check(&MC, &c)?;
    let [np, nc, ns, nz] = ["NP", "NC", "NS", "NZ"].map(|v| dim(&dims, v));
    let [setup, init, step, payoff] = [&c[0], &c[1], &c[2], &c[3]];
    let nres = step.nres;
    let fns = [&setup.eval, &setup.vjp, &init.eval, &init.vjp, &step.eval, &step.fwd, &step.bwd, &payoff.eval, &payoff.vjp];
    let ty = imp.self_ty.clone();
    let generics = imp.generics.clone();
    let (ig, _, wh) = generics.split_for_impl();
    replace_sources(&mut imp, &c);
    Ok(quote! {
        #imp

        #[allow(clippy::all)]
        impl #ig #krate::McModel for #ty #wh {
            type P<__RaadsT: Copy> = [__RaadsT; #np];
            type C<__RaadsT: Copy> = [__RaadsT; #nc];
            type S<__RaadsT: Copy> = [__RaadsT; #ns];
            type Z<__RaadsT: Copy> = [__RaadsT; #nz];
            #[doc(hidden)]
            const STEP_NRES: usize = #nres;
            #(#[doc(hidden)] #fns)*
        }
    })
}

pub fn fd(mut imp: ItemImpl) -> Result<TokenStream> {
    let krate = crate::krate();
    let c = compile_all(&imp, &FD, &krate)?;
    let np = dim(&check(&FD, &c)?, "NP");
    let ty = imp.self_ty.clone();
    // The engine constants stay inherent, so a kernel can read `Self::EXERCISE_EPS` and user
    // code `Model::AMERICAN` without the trait in scope. The trait impl forwards to them:
    // `<Ty>::NAME` resolves to the inherent const, which takes precedence over the trait's.
    let consts: Vec<TokenStream> = imp
        .items
        .iter()
        .filter_map(|it| match it {
            ImplItem::Const(k) if k.ident == "AMERICAN" || k.ident == "EXERCISE_EPS" => {
                let (id, t) = (&k.ident, &k.ty);
                Some(quote!(const #id: #t = <#ty>::#id;))
            }
            _ => None,
        })
        .collect();
    let fns = c.iter().flat_map(|k| [&k.eval, &k.vjp]);
    let generics = imp.generics.clone();
    let (ig, _, wh) = generics.split_for_impl();
    replace_sources(&mut imp, &c);
    Ok(quote! {
        #imp

        #[allow(clippy::all)]
        impl #ig #krate::FdModel for #ty #wh {
            type P<__RaadsT: Copy> = [__RaadsT; #np];
            #(#consts)*
            #(#[doc(hidden)] #fns)*
        }
    })
}
