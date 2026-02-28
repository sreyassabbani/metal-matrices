# Metal Matrices

Just a fun project I've been working on. I'm using the `metal-rs` crate to interface with Metal/MSL. I'm thinking of expanding this into something more than matrix operations.

## Tooling

- Enter the dev environment with `direnv allow` (uses `.envrc` + `flake.nix`).
- The flake provides a stable Rust toolchain (`rustc`, `cargo`, `clippy`, `rustfmt`, `rust-src`) and `just`.
- Build shaders with `just` (default recipe) or `just all`.
- Clean generated shader artifacts with `just clean`.
- Metal shader compilation needs full Xcode selected via `xcode-select` (`metal`/`metallib` are not in CLT-only installs).

## Compute Backend API

- `Matrix<T, M, N>::multiply_with(&other, ComputeOptions)` supports `CPU`, `GPU`, or `Auto`.
- `Matrix<T, M, N>::multiply(&other)` defaults to CPU for all numeric types.
- `ComputeOptions::gpu().with_fallback(true)` gives explicit GPU-first behavior with CPU fallback.
- Integer scalar types (`i8`, `u8`, `i16`, `u16`, `i32`, `u32`, `i64`, `u64`) always work on CPU.
- Current GPU path is implemented for `f32` only; requesting GPU for unsupported scalar types returns a clear error unless fallback is enabled.

## Benchmark Report

- Run `just bench-report` to execute Criterion benchmarks and generate a condensed statistical Markdown table.
- Output file: `docs/benchmark-comparison.md`.
- The report includes mean latency with 95% CI, relative speedup vs baseline, permutation-test p-values, and Cliff's delta effect size.
