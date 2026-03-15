use std::{
    env,
    ffi::OsStr,
    fs,
    path::{Path, PathBuf},
    process::Command,
};

fn main() {
    let shader_dir = Path::new("shaders");
    println!("cargo:rerun-if-changed={}", shader_dir.display());

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR must be set"));
    let shader_sources = collect_shader_sources(shader_dir);

    for shader in &shader_sources {
        println!("cargo:rerun-if-changed={}", shader.display());
    }

    if shader_sources.is_empty() {
        panic!("no .metal shader files found in {}", shader_dir.display());
    }

    let Some(metal_bin) = find_tool("metal") else {
        copy_prebuilt_metallib(shader_dir, &out_dir);
        return;
    };
    let Some(metallib_bin) = find_tool("metallib") else {
        copy_prebuilt_metallib(shader_dir, &out_dir);
        return;
    };

    let mut air_files = Vec::with_capacity(shader_sources.len());
    for shader in &shader_sources {
        let air_path = out_dir.join(
            shader
                .file_stem()
                .expect("shader filename must have a stem")
                .to_string_lossy()
                .to_string()
                + ".air",
        );
        run_tool(
            &metal_bin,
            [
                OsStr::new("-c"),
                shader.as_os_str(),
                OsStr::new("-o"),
                air_path.as_os_str(),
            ],
            "metal shader compilation",
        );
        air_files.push(air_path);
    }

    let metallib_path = out_dir.join("shaders.metallib");
    let mut args: Vec<&OsStr> = air_files.iter().map(|path| path.as_os_str()).collect();
    args.push(OsStr::new("-o"));
    args.push(metallib_path.as_os_str());
    run_tool(&metallib_bin, args, "metallib linking");

    if !metallib_path.exists() {
        panic!("expected metallib output at {}", metallib_path.display());
    }
}

fn collect_shader_sources(shader_dir: &Path) -> Vec<PathBuf> {
    let mut sources = Vec::new();
    let read_dir = fs::read_dir(shader_dir)
        .unwrap_or_else(|err| panic!("failed to read shader directory {}: {err}", shader_dir.display()));

    for entry in read_dir {
        let entry = entry.expect("failed to read shader directory entry");
        let path = entry.path();
        if path.extension() == Some(OsStr::new("metal")) {
            sources.push(path);
        }
    }

    sources.sort();
    sources
}

fn find_tool(tool: &str) -> Option<PathBuf> {
    let output = Command::new("xcrun")
        .args(["-sdk", "macosx", "--find", tool])
        .output()
        .unwrap_or_else(|err| panic!("failed to execute xcrun while locating {tool}: {err}"));

    if !output.status.success() {
        return None;
    }

    Some(PathBuf::from(
        String::from_utf8(output.stdout)
            .expect("xcrun output must be valid utf-8")
            .trim(),
    ))
}

fn run_tool<I>(tool: &Path, args: I, label: &str)
where
    I: IntoIterator,
    I::Item: AsRef<OsStr>,
{
    let output = Command::new(tool)
        .args(args)
        .output()
        .unwrap_or_else(|err| panic!("failed to run {label} with {}: {err}", tool.display()));

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        panic!("{label} failed with {}: {stderr}", tool.display());
    }
}

fn copy_prebuilt_metallib(shader_dir: &Path, out_dir: &Path) {
    let prebuilt = shader_dir.join("shaders.metallib");
    if !prebuilt.exists() {
        panic!(
            "unable to locate Metal build tools via xcrun and no prebuilt metallib was found at {}",
            prebuilt.display()
        );
    }

    let target = out_dir.join("shaders.metallib");
    fs::copy(&prebuilt, &target).unwrap_or_else(|err| {
        panic!(
            "failed to copy prebuilt metallib from {} to {}: {err}",
            prebuilt.display(),
            target.display()
        )
    });
    println!(
        "cargo:warning=using prebuilt {} because the full Metal toolchain is unavailable through xcrun",
        prebuilt.display()
    );
}
