# Metal Matrices

Just a fun project I've been working on. I'm using the `metal-rs` crate to interface with Metal/MSL. I'm thinking of expanding this into something more than matrix operations.

## Tooling

- Enter the dev environment with `direnv allow` (uses `.envrc` + `flake.nix`).
- The flake provides a stable Rust toolchain (`rustc`, `cargo`, `clippy`, `rustfmt`, `rust-src`) and `just`.
- Build shaders with `just` (default recipe) or `just all`.
- Clean generated shader artifacts with `just clean`.
- Metal shader compilation needs full Xcode selected via `xcode-select` (`metal`/`metallib` are not in CLT-only installs).

## Compute Backend API

- `Matrix<f32, M, N>::multiply_with(&other, ComputeOptions)` lets you select `CPU`, `GPU`, or `Auto`.
- `ComputeOptions::auto().with_fallback(true)` prefers GPU but falls back to CPU.
- `Matrix<f32, M, N>::multiply(&other)` is the default convenience path (CPU backend).
