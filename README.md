# Metal Matrices

Fixed-size matrices and vectors with a CPU-first API and an explicit reusable Metal runtime for `f32` matrix multiplication.

## Tooling

- Enter the dev environment with `direnv allow` (uses `.envrc` + `flake.nix`).
- The flake provides a stable Rust toolchain (`rustc`, `cargo`, `clippy`, `rustfmt`, `rust-src`) and `just`.
- Default builds are CPU-only and do not require Metal tooling.
- The `metal` feature is opt-in and macOS-only:
  - `cargo test`
  - `cargo test --features metal`
- The Metal runtime compiles the checked-in shader source in [`shaders/matrix_multiplication.metal`](shaders/matrix_multiplication.metal) when `MetalRuntime::new()` runs.
- `build.rs` only tracks shader source changes for Cargo rebuilds; there is no checked-in bootstrap `metallib` fallback.
- `just all` and `just clean` remain optional developer convenience commands for manually producing or deleting `.air` and `.metallib` artifacts.

## Core API

- `Matrix<T, M, N>::multiply(&other)` is the canonical CPU multiplication path.
- Matrix arithmetic is shape-safe through const generics and stores entries in boxed contiguous arrays.
- Current generic scalar support covers the built-in numeric types implemented in [`Numeric`](src/numeric.rs).
- Row-major construction and access are explicit through `from_row_major`, `as_row_major_slice`, `row`, and `row_slice`.
- `ndarray::Array2` interop is available for row-major conversions and CPU correctness cross-checks.
- Unsupported linear algebra operations such as determinant and inverse return explicit `MatrixError` values.

## GPU API

- GPU execution is explicit and strict through `MetalRuntime`.
- `MetalRuntime::new()` initializes a reusable Metal device/queue/pipeline bundle.
- `MetalRuntime::multiply(&lhs, &rhs)` performs tiled `f32` matrix multiplication on the GPU and returns `MetalError` on any failure.
- The runtime exposes `shader_kind()` and `threadgroup_shape()` for inspection.
- There is no automatic CPU fallback in the GPU path.

## Benchmark Report

- Run `just bench-report` to execute Criterion benchmarks and generate a condensed statistical Markdown table.
- Output file: `docs/benchmark-comparison.md`.
- On macOS, `just bench-report` runs the fixed-size benchmark with `--features metal`, so the report includes CPU, `ndarray`, warm GPU runtime, and cold init + multiply GPU timings.
- The report includes mean latency with 95% CI, relative speedup vs baseline, permutation-test p-values, and Cliff's delta effect size.
- Run `just bench-scaling-plots` to benchmark multiple matrix sizes and generate plots + Markdown summary.
- Outputs: `docs/benchmark-scaling.md` and `docs/plots/matrix-scaling-*.png`.
