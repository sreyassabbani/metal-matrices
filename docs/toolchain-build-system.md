## 2. Toolchain/build system (important operational layer)

direnv activates the Nix flake shell: .envrc.
Flake provides stable Rust + tools (cargo, clippy, rustfmt, rust-analyzer, just): flake.nix.
Rust toolchain is also pinned via rust-toolchain.toml: rust-toolchain.toml.
Shader build moved from Make to Just:
Compile .metal -> .air
Link .air -> shaders/shaders.metallib
Recipes in justfile
