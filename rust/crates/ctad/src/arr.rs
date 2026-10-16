//! Fixed-size arrays as associated types of models (`type P<T: Copy> = [T; 5]`), so engines
//! stay generic over sizes on stable Rust.

/// `[T; N]` for some `N`.
pub trait Arr<T>: Copy + AsRef<[T]> + AsMut<[T]> {
    const LEN: usize;
    fn splat(t: T) -> Self;
}

impl<T: Copy, const N: usize> Arr<T> for [T; N] {
    const LEN: usize = N;
    #[inline(always)]
    fn splat(t: T) -> Self {
        [t; N]
    }
}
