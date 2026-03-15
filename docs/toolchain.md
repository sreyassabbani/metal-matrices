# Toolchain

- `direnv` activates the [Nix flake shell](../flake.nix)
  - Flake provides stable Rust + tools (cargo, clippy, rustfmt, rust-analyzer)
- Rust toolchain is also pinned via [`rust-toolchain.toml`](../rust-toolchain.toml)
- Default Cargo workflows are CPU-only and do not require Metal tooling
  - `cargo test`
  - `cargo check --examples`
- Metal support is opt-in through `--features metal` and only compiled on macOS
  - `cargo test --features metal`
  - `cargo check --examples --features metal`
- [`build.rs`](../build.rs) only tells Cargo to rebuild when shader source changes
- [`src/metal.rs`](../src/metal.rs) compiles the checked-in shader source at runtime when `MetalRuntime::new()` initializes
- [`justfile`](../justfile) keeps manual shader rebuild/cleanup commands as optional developer conveniences if you want `.air` or `.metallib` artifacts for debugging
