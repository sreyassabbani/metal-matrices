# Metal Matrices

Fixed-size matrices and vectors with a CPU-first API and an explicit reusable Metal runtime for `f32` matrix multiplication.

## Tooling

- Enter the dev environment with `direnv allow` (uses `.envrc` + `flake.nix`).
- The flake provides a stable Rust toolchain (`rustc`, `cargo`, `clippy`, `rustfmt`, `rust-src`) and `just`.
- Cargo embeds a Metal library at build time via `build.rs`.
- If the full Metal toolchain is available through `xcrun`, the embedded library is rebuilt from `shaders/*.metal`.
- If `metallib` is unavailable, Cargo falls back to embedding the checked-in bootstrap `shaders/shaders.metallib`.
- `just all` and `just clean` remain optional developer convenience commands for manually rebuilding or deleting shader artifacts.

## Core API

- `Matrix<T, M, N>::multiply(&other)` is the canonical CPU multiplication path.
- Matrix arithmetic is shape-safe through const generics and stores entries in boxed contiguous arrays.
- Current generic scalar support covers the built-in numeric types implemented in [`Numeric`](src/numeric.rs).
- Unsupported linear algebra operations such as determinant and inverse return explicit `MatrixError` values.

## GPU API

- GPU execution is explicit and strict through `MetalRuntime`.
- `MetalRuntime::new()` initializes a reusable Metal device/queue/pipeline bundle.
- `MetalRuntime::multiply(&lhs, &rhs)` performs `f32` matrix multiplication on the GPU and returns `MetalError` on any failure.
- There is no automatic CPU fallback in the GPU path.

## Benchmark Report

- Run `just bench-report` to execute Criterion benchmarks and generate a condensed statistical Markdown table.
- Output file: `docs/benchmark-comparison.md`.
- The fixed-size benchmark report includes CPU, `ndarray`, warm GPU runtime, and cold init + multiply GPU timings.
- The report includes mean latency with 95% CI, relative speedup vs baseline, permutation-test p-values, and Cliff's delta effect size.
- Run `just bench-scaling-plots` to benchmark multiple matrix sizes and generate plots + Markdown summary.
- Outputs: `docs/benchmark-scaling.md` and `docs/plots/matrix-scaling-*.png`.
