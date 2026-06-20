//! Build script for akh-medu.
//!
//! When the `rocm` feature is enabled (AMD GPU backend, ADR 068), compile the HIP
//! dequantization kernel to a shared library with amdclang++. The library is dlopen'd at
//! runtime by `src/nlu/hip.rs` from `$OUT_DIR` (exposed to the crate via `env!("OUT_DIR")`).
//!
//! If the ROCm toolchain is absent the build does NOT fail — it warns and skips, so a
//! `--features rocm` build still succeeds on a box without ROCm (the runtime then reports the
//! kernel as unavailable and the quantized GPU path falls back). The non-quantized resident
//! path (rocBLAS via dlopen) does not need this kernel.

use std::path::Path;
use std::process::Command;

/// Find the highest-versioned gcc install dir (e.g. `/usr/lib/gcc/x86_64-linux-gnu/13`).
fn newest_gcc_install_dir() -> Option<String> {
    let base = "/usr/lib/gcc/x86_64-linux-gnu";
    let mut versions: Vec<(u32, String)> = std::fs::read_dir(base)
        .ok()?
        .flatten()
        .filter_map(|e| {
            let name = e.file_name().into_string().ok()?;
            let v: u32 = name.parse().ok()?;
            // Must have gcc support files AND the matching libstdc++ C++ headers (clang's HIP
            // device pass does `#include_next <cmath>`); a compiler-only gcc dir without
            // `libstdc++-<v>-dev` fails the device compile.
            let has_crt = e.path().join("crtbegin.o").exists();
            let has_cxx_headers = Path::new(&format!("/usr/include/c++/{v}/cmath")).exists();
            if has_crt && has_cxx_headers {
                Some((v, format!("{base}/{name}")))
            } else {
                None
            }
        })
        .collect();
    versions.sort_by_key(|(v, _)| *v);
    versions.pop().map(|(_, p)| p)
}

fn main() {
    // Only do anything for the rocm feature.
    if std::env::var("CARGO_FEATURE_ROCM").is_err() {
        return;
    }
    println!("cargo:rerun-if-changed=src/nlu/hip_kernels/dequant.hip");

    let rocm_path = std::env::var("ROCM_PATH").unwrap_or_else(|_| "/opt/rocm".to_string());
    // Prefer hipcc, fall back to amdclang++.
    let compiler = ["bin/hipcc", "bin/amdclang++"]
        .iter()
        .map(|c| format!("{rocm_path}/{c}"))
        .find(|p| Path::new(p).exists());

    let compiler = match compiler {
        Some(c) => c,
        None => {
            println!(
                "cargo:warning=ROCm HIP compiler not found under {rocm_path}; \
                 quantized GPU kernel not built (runtime will fall back to CPU/f32)."
            );
            return;
        }
    };

    let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR set by cargo");
    let out_lib = format!("{out_dir}/libakh_hip_kernels.so");

    let mut args: Vec<String> = vec![
        "-x".into(),
        "hip".into(),
        "--offload-arch=gfx1100".into(),
        format!("--rocm-path={rocm_path}"),
        "-O3".into(),
        "-fPIC".into(),
        "-shared".into(),
        format!("-I{rocm_path}/include"),
        "src/nlu/hip_kernels/dequant.hip".into(),
        "-o".into(),
        out_lib.clone(),
        format!("-L{rocm_path}/lib"),
        "-lamdhip64".into(),
    ];
    // clang's HIP device pass needs an explicit gcc install dir to resolve libstdc++ headers
    // (`#include_next <cmath>`) on Debian/Ubuntu-family layouts.
    if let Some(gcc_dir) = newest_gcc_install_dir() {
        args.push(format!("--gcc-install-dir={gcc_dir}"));
    }

    let status = Command::new(&compiler).args(&args).status();

    match status {
        Ok(s) if s.success() => {
            println!("cargo:warning=built HIP dequant kernel: {out_lib}");
        }
        Ok(s) => {
            println!(
                "cargo:warning=HIP kernel compile failed (exit {s}); quantized GPU path will \
                 fall back. Compiler: {compiler}"
            );
        }
        Err(e) => {
            println!("cargo:warning=could not invoke HIP compiler {compiler}: {e}");
        }
    }
}
