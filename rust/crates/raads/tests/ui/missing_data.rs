use raads::{AdNum, mc_model};

pub struct Model;

#[mc_model]
impl Model {
    pub fn setup<N: AdNum>(p: [N; 5], #[data] n_steps: N) -> [N; 2] {
        let dt = p[3] / n_steps;
        [p[1] * dt, p[2] * dt.sqrt()]
    }

    pub fn init<N: AdNum>(p: [N; 5]) -> [N; 1] {
        [p[0].ln()]
    }

    pub fn step<N: AdNum>(s: [N; 1], c: [N; 2], z: [N; 1]) -> [N; 1] {
        [s[0] + c[0] + c[1] * z[0]]
    }

    pub fn payoff<N: AdNum>(s: [N; 1], p: [N; 5], _c: [N; 2]) -> N {
        (s[0].exp() - p[4]).relu()
    }
}

fn main() {}
