# Toolchain

- `direnv` activates the [Nix flake shell](../flake.nix)
  - Flake provides stable Rust + tools (cargo, clippy, rustfmt, rust-analyzer)
- Rust toolchain is also pinned via [`rust-toolchain.toml`](../rust-toolchain.toml).
- [`justfile`](../justfile) shader build:
  - Compile `.metal` → `.air`
  - Link `.air` → `shaders/shaders.metallib`
