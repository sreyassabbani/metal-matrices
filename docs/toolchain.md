# Toolchain

- `direnv` activates the [Nix flake shell](../flake.nix)
  - Flake provides stable Rust + tools (cargo, clippy, rustfmt, rust-analyzer)
- Rust toolchain is also pinned via [`rust-toolchain.toml`](../rust-toolchain.toml)
- `cargo build`, `cargo test`, and `cargo bench` embed a Metal library through [`build.rs`](../build.rs)
  - Preferred path: compile `.metal` → `.air` and link `.air` → embedded `shaders.metallib`
  - Bootstrap path: if `metallib` is unavailable through `xcrun`, build falls back to embedding the checked-in `shaders/shaders.metallib`
- [`justfile`](../justfile) keeps manual shader rebuild/cleanup commands as optional developer conveniences
