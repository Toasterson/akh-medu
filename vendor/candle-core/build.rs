//! Build script for the vendored candle-core fork (Phase 45 / ADR 073).
//!
//! When the `hip` feature is enabled, compile the elementwise HIP kernels to a shared
//! library with hipcc/amdclang++. `src/hip_backend.rs` dlopen's it at runtime from
//! `$OUT_DIR`. If the ROCm toolchain is absent the build does NOT fail — it warns and
//! skips, so a `--features hip` build still succeeds without ROCm (the runtime then
//! reports the kernels unavailable and elementwise ops error with a clear message).
//! Without the `hip` feature this is a no-op (default builds are unaffected).

use std::path::Path;
use std::process::Command;

/// Highest-versioned gcc install dir with libstdc++ headers (clang's HIP device pass
/// does `#include_next <cmath>` on Debian/Ubuntu layouts).
fn newest_gcc_install_dir() -> Option<String> {
    let base = "/usr/lib/gcc/x86_64-linux-gnu";
    let mut versions: Vec<(u32, String)> = std::fs::read_dir(base)
        .ok()?
        .flatten()
        .filter_map(|e| {
            let name = e.file_name().into_string().ok()?;
            let v: u32 = name.parse().ok()?;
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

/// Compile one `.hip` source to a shared library with the HIP toolchain. Graceful:
/// a missing toolchain / failed compile warns and returns (the dlopen at runtime then
/// reports the kernels unavailable), never aborts the build.
fn compile_hip(compiler: &str, rocm_path: &str, src: &str, out_lib: &str, label: &str) {
    let mut args: Vec<String> = vec![
        "-x".into(),
        "hip".into(),
        "--offload-arch=gfx1100".into(),
        format!("--rocm-path={rocm_path}"),
        "-O3".into(),
        "-fPIC".into(),
        "-shared".into(),
        format!("-I{rocm_path}/include"),
        src.into(),
        "-o".into(),
        out_lib.into(),
        format!("-L{rocm_path}/lib"),
        "-lamdhip64".into(),
    ];
    if let Some(gcc_dir) = newest_gcc_install_dir() {
        args.push(format!("--gcc-install-dir={gcc_dir}"));
    }
    match Command::new(compiler).args(&args).status() {
        Ok(s) if s.success() => println!("cargo:warning=built candle HIP {label} kernels: {out_lib}"),
        Ok(s) => println!(
            "cargo:warning=candle HIP {label} kernel compile failed (exit {s}); those ops will \
             error. Compiler: {compiler}"
        ),
        Err(e) => println!("cargo:warning=could not invoke HIP compiler {compiler}: {e}"),
    }
}

fn main() {
    if std::env::var("CARGO_FEATURE_HIP").is_err() {
        return;
    }
    println!("cargo:rerun-if-changed=src/hip_kernels/elementwise.hip");
    println!("cargo:rerun-if-changed=src/hip_kernels/quant.hip");

    let rocm_path = std::env::var("ROCM_PATH").unwrap_or_else(|_| "/opt/rocm".to_string());
    let compiler = ["bin/hipcc", "bin/amdclang++"]
        .iter()
        .map(|c| format!("{rocm_path}/{c}"))
        .find(|p| Path::new(p).exists());
    let compiler = match compiler {
        Some(c) => c,
        None => {
            println!(
                "cargo:warning=ROCm HIP compiler not found under {rocm_path}; candle HIP \
                 kernels not built (those ops will error at runtime)."
            );
            return;
        }
    };

    let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR set by cargo");

    // Elementwise kernels (affine/unary/binary/reduce/cast/index_select/where_cond/ucopy).
    compile_hip(
        &compiler,
        &rocm_path,
        "src/hip_kernels/elementwise.hip",
        &format!("{out_dir}/libcandle_hip_kernels.so"),
        "elementwise",
    );

    // Quantized dequant kernels (Q4_K → f32, Phase 45e).
    compile_hip(
        &compiler,
        &rocm_path,
        "src/hip_kernels/quant.hip",
        &format!("{out_dir}/libcandle_hip_quant.so"),
        "quant",
    );
}
