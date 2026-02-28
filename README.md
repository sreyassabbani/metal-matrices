# Metal Matrices

Just a fun project I've been working on. I'm using the `metal-rs` crate to interface with Metal/MSL. I'm thinking of expanding this into something more than matrix operations.

## Tooling

- Enter the dev environment with `direnv allow` (uses `.envrc` + `flake.nix`).
- The flake provides a stable Rust toolchain (`rustc`, `cargo`, `clippy`, `rustfmt`, `rust-src`) and `just`.
- Build shaders with `just` (default recipe) or `just all`.
- Clean generated shader artifacts with `just clean`.
