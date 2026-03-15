use std::env;

fn main() {
    if env::var_os("CARGO_FEATURE_METAL").is_some() {
        println!("cargo:rerun-if-changed=shaders/matrix_multiplication.metal");
    }
}
