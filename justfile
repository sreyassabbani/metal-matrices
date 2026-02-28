set shell := ["bash", "-eu", "-o", "pipefail", "-c"]

# Default target builds all shaders into shaders/shaders.metallib.
default: all

all:
    #!/usr/bin/env bash
    set -euo pipefail
    shopt -s nullglob
    metal_files=(shaders/*.metal)
    if [ ${#metal_files[@]} -eq 0 ]; then
      echo "No .metal files found in shaders/"
      exit 1
    fi

    if ! metal_bin="$(xcrun -sdk macosx --find metal 2>/dev/null)"; then
      echo "Unable to locate 'metal' via xcrun. Install full Xcode and select it with xcode-select."
      exit 1
    fi
    if ! "${metal_bin}" -help >/dev/null 2>&1; then
      echo "Metal compiler found but Metal Toolchain is unavailable."
      echo "Run: xcodebuild -runFirstLaunch"
      echo "Then: xcodebuild -downloadComponent MetalToolchain"
      echo "If download fails, inspect version with: xcodebuild -showComponent MetalToolchain -json"
      exit 1
    fi
    if ! metallib_bin="$(xcrun -sdk macosx --find metallib 2>/dev/null)"; then
      echo "Unable to locate 'metallib' via xcrun."
      echo "This typically means the Metal Toolchain component is missing from Xcode."
      echo "Run: xcodebuild -downloadComponent MetalToolchain"
      echo "If needed, retry with: xcodebuild -downloadComponent MetalToolchain -buildVersion <value-from-showComponent>"
      exit 1
    fi

    air_files=()
    for metal in "${metal_files[@]}"; do
      air="${metal%.metal}.air"
      echo "Compiling ${metal} to ${air}"
      "${metal_bin}" -c "${metal}" -o "${air}"
      air_files+=("${air}")
    done

    echo "Creating shaders/shaders.metallib from ${air_files[*]}"
    "${metallib_bin}" "${air_files[@]}" -o shaders/shaders.metallib

clean:
    #!/usr/bin/env bash
    shopt -s nullglob
    files=(shaders/*.air shaders/shaders.metallib)
    if [ ${#files[@]} -eq 0 ]; then
      echo "Nothing to clean"
      exit 0
    fi

    echo "Removing files: ${files[*]}"
    rm -f "${files[@]}"
