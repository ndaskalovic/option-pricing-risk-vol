pub const NS: usize = 52;
pub const NPATHS: usize = 200_000;

pub fn load_normals() -> Vec<f32> {
    let p = concat!(env!("CARGO_MANIFEST_DIR"), "/normals_w32.f32");
    let b = std::fs::read(p).expect("run mojo_ref.mojo first");
    b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

pub fn report(label: &str, price: f64, g: [f64; 5]) {
    println!("{label}: price={price:.7} grad=[{:.7}, {:.5}, {:.5}, {:.6}, {:.7}]", g[0], g[1], g[2], g[3], g[4]);
}

pub fn best_ms<F: FnMut() -> f64>(reps: usize, mut f: F) -> (f64, f64) {
    let mut best = f64::MAX; let mut chk = 0.0;
    for _ in 0..reps {
        let t = std::time::Instant::now();
        chk += std::hint::black_box(f());
        best = best.min(t.elapsed().as_secs_f64() * 1e3);
    }
    (best, chk)
}
