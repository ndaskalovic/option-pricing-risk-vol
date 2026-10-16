//! Model attributes: compile each kernel of an `impl` block and implement the engine trait.

use crate::Compiled;
use crate::front::Shape;
use proc_macro2::TokenStream;
use quote::quote;
use syn::spanned::Spanned;
use syn::{Error, ImplItem, ImplItemFn, ItemFn, ItemImpl, Result};

/// Kernels of the block by name; other items are kept as they are.
fn kernels<'a>(imp: &mut ItemImpl, names: &[&'a str]) -> Result<Vec<(&'a str, ItemFn)>> {
    if imp.trait_.is_some() {
        return Err(Error::new(imp.span(), "put the model attribute on an inherent `impl Model { ... }` block"));
    }
    let mut found = Vec::new();
    for item in &imp.items {
        if let ImplItem::Fn(f) = item {
            if let Some(n) = names.iter().find(|n| f.sig.ident == **n) {
                found.push((*n, to_item_fn(f)));
            }
        }
    }
    for n in names {
        if !found.iter().any(|(m, _)| m == n) {
            return Err(Error::new(imp.self_ty.span(), format!("missing kernel `{n}`")));
        }
    }
    Ok(found)
}

fn to_item_fn(f: &ImplItemFn) -> ItemFn {
    ItemFn { attrs: f.attrs.clone(), vis: f.vis.clone(), sig: f.sig.clone(), block: Box::new(f.block.clone()) }
}

/// Replace each kernel in the block by its rewritten (generic-valid) source.
fn replace_sources(imp: &mut ItemImpl, compiled: &[(&str, Compiled)]) {
    for item in imp.items.iter_mut() {
        if let ImplItem::Fn(f) = item {
            if let Some((_, c)) = compiled.iter().find(|(n, _)| f.sig.ident == *n) {
                f.sig = c.source.sig.clone();
                f.block = (*c.source.block).clone();
            }
        }
    }
}

fn len(s: Shape, what: &str, span: proc_macro2::Span) -> Result<usize> {
    match s {
        Shape::Array(n) => Ok(n),
        Shape::Scalar => Err(Error::new(span, format!("{what} must be an array `[N; LEN]`"))),
    }
}

fn compile(name: &str, f: ItemFn) -> Result<Compiled> {
    let n = |k: &str| format!("{name}_{k}");
    crate::compile(f, "", [&n("eval"), &n("vjp"), &n("fwd"), &n("bwd")])
}

pub fn mc(mut imp: ItemImpl) -> Result<TokenStream> {
    let ks = kernels(&mut imp, &["setup", "init", "step", "payoff"])?;
    let sp = imp.self_ty.span();
    let mut c = Vec::new();
    for (name, f) in ks {
        c.push((name, compile(name, f)?));
    }
    let get = |n: &str| &c.iter().find(|(m, _)| *m == n).unwrap().1;
    let (setup, init, step, payoff) = (get("setup"), get("init"), get("step"), get("payoff"));
    if setup.shapes.len() != 2 || init.shapes.len() != 1 || step.shapes.len() != 3 || payoff.shapes.len() != 3 {
        return Err(Error::new(sp, "expected setup(p, n_steps), init(p), step(s, c, z), payoff(s, p, c)"));
    }
    let np = len(setup.shapes[0], "setup's parameters", sp)?;
    let nc = len(setup.ret, "setup's result", sp)?;
    let ns = len(init.ret, "init's result", sp)?;
    let nz = len(step.shapes[2], "step's normals", sp)?;
    let nres = step.nres;
    let (se, sv, ie, iv) = (&setup.eval, &setup.vjp, &init.eval, &init.vjp);
    let (te, tf, tb, pe, pv) = (&step.eval, &step.fwd, &step.bwd, &payoff.eval, &payoff.vjp);
    let ty = imp.self_ty.clone();
    let generics = imp.generics.clone();
    let (ig, _, wh) = generics.split_for_impl();
    replace_sources(&mut imp, &c);
    Ok(quote! {
        #imp

        #[allow(clippy::all)]
        impl #ig ::ctad::McModel for #ty #wh {
            type P<T: Copy> = [T; #np];
            type C<T: Copy> = [T; #nc];
            type S<T: Copy> = [T; #ns];
            type Z<T: Copy> = [T; #nz];
            const STEP_NRES: usize = #nres;
            #se #sv #ie #iv #te #tf #tb #pe #pv
        }
    })
}

pub fn fd(mut imp: ItemImpl) -> Result<TokenStream> {
    let ks = kernels(&mut imp, &["coeffs", "payoff", "spot", "maturity"])?;
    let sp = imp.self_ty.span();
    let mut c = Vec::new();
    for (name, f) in ks {
        c.push((name, compile(name, f)?));
    }
    let get = |n: &str| &c.iter().find(|(m, _)| *m == n).unwrap().1;
    let (coeffs, payoff, spot, maturity) = (get("coeffs"), get("payoff"), get("spot"), get("maturity"));
    if coeffs.shapes.len() != 2 || payoff.shapes.len() != 2 || spot.shapes.len() != 1 || maturity.shapes.len() != 1 {
        return Err(Error::new(sp, "expected coeffs(x, p), payoff(x, p), spot(p), maturity(p)"));
    }
    if coeffs.ret != Shape::Array(3) {
        return Err(Error::new(sp, "coeffs returns `[N; 3]`: a, b, c of V_tau = a V_xx + b V_x - c V"));
    }
    let np = len(spot.shapes[0], "spot's parameters", sp)?;
    // the engine constants move from the inherent block to the trait impl
    let mut consts = Vec::new();
    imp.items.retain(|it| match it {
        ImplItem::Const(k) if k.ident == "AMERICAN" || k.ident == "EXERCISE_EPS" => {
            let mut k = k.clone();
            k.vis = syn::Visibility::Inherited;
            consts.push(k);
            false
        }
        _ => true,
    });
    let fns = [coeffs, payoff, spot, maturity].map(|k| {
        let (e, v) = (&k.eval, &k.vjp);
        quote!(#e #v)
    });
    let ty = imp.self_ty.clone();
    let generics = imp.generics.clone();
    let (ig, _, wh) = generics.split_for_impl();
    replace_sources(&mut imp, &c);
    Ok(quote! {
        #imp

        #[allow(clippy::all)]
        impl #ig ::ctad::FdModel for #ty #wh {
            type P<T: Copy> = [T; #np];
            #(#consts)*
            #(#fns)*
        }
    })
}
