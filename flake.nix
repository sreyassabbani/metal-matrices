{
  inputs = {
    nixpkgs.url = "https://flakehub.com/f/DeterminateSystems/nixpkgs-weekly/0.tar.gz";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs { inherit system; };
        llvm = pkgs.llvmPackages_21;
      in
      {
        devShells.default = pkgs.mkShell {
          packages = with pkgs; [
            rustc
            cargo
            rustfmt
            clippy
            rust-analyzer
            rustPlatform.rustLibSrc
            llvm.lldb
            gemini-cli
            bacon
            python312
            uv
            just
            sourcekit-lsp
          ];

          RUST_SRC_PATH = "${pkgs.rustPlatform.rustLibSrc}/lib/rustlib/src/rust/library";

          shellHook = ''
            export UV_PROJECT_ENVIRONMENT="$PWD/.venv"
            if [ ! -d "$UV_PROJECT_ENVIRONMENT" ]; then
              echo "Creating Python virtual environment with uv at $UV_PROJECT_ENVIRONMENT"
              uv venv "$UV_PROJECT_ENVIRONMENT"
            fi
            source "$UV_PROJECT_ENVIRONMENT/bin/activate"

            echo "Nix dev shell activated"
            echo "rustc: $(rustc --version 2>/dev/null || echo 'not found')"
            echo "cargo: $(cargo --version 2>/dev/null || echo 'not found')"
            echo "rust-analyzer: $(rust-analyzer --version 2>/dev/null || echo 'not found')"
            echo "python: $(python --version 2>/dev/null || echo 'not found')"
            echo "uv: $(uv --version 2>/dev/null || echo 'not found')"
          '';
        };
      }
    );
}
