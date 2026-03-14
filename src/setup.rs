//! NLU model and runtime setup.
//!
//! Downloads ML model files and the ONNX Runtime shared library so the NLU
//! pipeline can use Tier 2 (NER) and Tier 3 (LLM) inference.
//!
//! All assets are stored in the shared data directory
//! (`$XDG_DATA_HOME/akh-medu/models/`) so they are reused across workspaces.

use std::fs;
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};

use miette::Diagnostic;
use thiserror::Error;

use crate::paths::AkhPaths;

// ── Errors ─────────────────────────────────────────────────────────────────

#[derive(Debug, Error, Diagnostic)]
pub enum SetupError {
    #[error("failed to resolve XDG paths")]
    #[diagnostic(code(akh::setup::paths))]
    Paths(#[from] crate::paths::PathError),

    #[error("HTTP request failed: {reason}")]
    #[diagnostic(code(akh::setup::http), help("Check your network connection."))]
    Http { reason: String },

    #[error("I/O error: {0}")]
    #[diagnostic(code(akh::setup::io))]
    Io(#[from] io::Error),

    #[error("unsupported platform: {os}/{arch}")]
    #[diagnostic(
        code(akh::setup::platform),
        help("ONNX Runtime setup supports macOS and Linux on x86_64 and aarch64.")
    )]
    UnsupportedPlatform { os: String, arch: String },
}

pub type SetupResult<T> = Result<T, SetupError>;

// ── Model manifest ─────────────────────────────────────────────────────────

/// NER model files (Tier 2).
const NER_BASE_URL: &str =
    "https://huggingface.co/Xenova/distilbert-base-multilingual-cased-ner-hrl/resolve/main";
const NER_FILES: &[(&str, &str)] = &[
    ("onnx/model_quantized.onnx", "model.onnx"),
    ("tokenizer.json", "tokenizer.json"),
    ("config.json", "config.json"),
];
const NER_SUBDIR: &str = "ner";

/// LLM model file (Tier 3).
const LLM_URL: &str = "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf";
const LLM_SUBDIR: &str = "llm";
const LLM_FILENAME: &str = "qwen2.5-1.5b-instruct-q4_k_m.gguf";

/// Pinned ONNX Runtime version (must match `ort` crate requirement >= 1.23.x).
const ORT_VERSION: &str = "1.23.2";

// ── Public API ─────────────────────────────────────────────────────────────

/// Download NER model files to the shared models directory.
pub fn download_ner_models(paths: &AkhPaths) -> SetupResult<()> {
    let dest = paths.models_dir().join(NER_SUBDIR);
    fs::create_dir_all(&dest)?;

    for &(remote_path, local_name) in NER_FILES {
        let url = format!("{NER_BASE_URL}/{remote_path}");
        let target = dest.join(local_name);
        if target.exists() {
            eprintln!("  [skip] {} (already exists)", target.display());
            continue;
        }
        download_file(&url, &target)?;
    }

    eprintln!("NER models ready: {}", dest.display());
    Ok(())
}

/// Download the LLM model file to the shared models directory.
pub fn download_llm_model(paths: &AkhPaths) -> SetupResult<()> {
    let dest = paths.models_dir().join(LLM_SUBDIR);
    fs::create_dir_all(&dest)?;

    let target = dest.join(LLM_FILENAME);
    if target.exists() {
        eprintln!("  [skip] {} (already exists)", target.display());
    } else {
        download_file(LLM_URL, &target)?;
    }

    eprintln!("LLM model ready: {}", dest.display());
    Ok(())
}

/// Download and install the ONNX Runtime shared library.
pub fn install_onnx_runtime(version: &str) -> SetupResult<PathBuf> {
    let platform = ort_platform_suffix()?;
    let url = format!(
        "https://github.com/microsoft/onnxruntime/releases/download/v{version}/onnxruntime-{platform}-{version}.tgz"
    );

    let install_dir = ort_install_dir()?;
    fs::create_dir_all(&install_dir)?;

    let lib_name = ort_lib_name();
    let lib_path = install_dir.join(lib_name);

    if lib_path.exists() {
        eprintln!("  [skip] {} (already installed)", lib_path.display());
        return Ok(lib_path);
    }

    // Download tarball to a temp file, then extract the shared library.
    eprintln!("Downloading ONNX Runtime v{version} for {platform}...");
    let resp = ureq::get(&url)
        .call()
        .map_err(|e| SetupError::Http {
            reason: format!("{e}"),
        })?;

    let mut tarball = Vec::new();
    resp.into_reader()
        .read_to_end(&mut tarball)
        .map_err(|e| SetupError::Http {
            reason: format!("body read: {e}"),
        })?;

    // Extract the .so/.dylib from the tarball.
    extract_ort_lib(&tarball, &lib_path, lib_name)?;

    eprintln!("ONNX Runtime installed: {}", lib_path.display());
    eprintln!(
        "\nSet this environment variable before running akhomed:\n  \
         export ORT_DYLIB_PATH=\"{}\"",
        lib_path.display()
    );
    Ok(lib_path)
}

/// Check the current NLU setup status and report.
pub fn check_setup(paths: &AkhPaths) -> SetupResult<()> {
    let models = paths.models_dir();
    println!("NLU Setup Status");
    println!("================");
    println!("Models directory: {}", models.display());
    println!();

    // NER files
    let ner_dir = models.join(NER_SUBDIR);
    println!("Tier 2 — NER (ONNX DistilBERT):");
    for &(_, local_name) in NER_FILES {
        let path = ner_dir.join(local_name);
        let status = if path.exists() {
            let meta = fs::metadata(&path)?;
            format!("ok ({:.1} MB)", meta.len() as f64 / 1_048_576.0)
        } else {
            "MISSING".to_string()
        };
        println!("  {local_name}: {status}");
    }

    // LLM file
    let llm_path = models.join(LLM_SUBDIR).join(LLM_FILENAME);
    println!("\nTier 3 — LLM (Qwen2.5-1.5B GGUF):");
    let llm_status = if llm_path.exists() {
        let meta = fs::metadata(&llm_path)?;
        format!("ok ({:.1} MB)", meta.len() as f64 / 1_048_576.0)
    } else {
        "MISSING".to_string()
    };
    println!("  {LLM_FILENAME}: {llm_status}");

    // ONNX Runtime
    println!("\nONNX Runtime:");
    let ort_env = std::env::var("ORT_DYLIB_PATH").ok();
    match ort_env {
        Some(ref p) if Path::new(p).exists() => {
            println!("  ORT_DYLIB_PATH={p} (found)");
        }
        Some(ref p) => {
            println!("  ORT_DYLIB_PATH={p} (FILE NOT FOUND)");
        }
        None => {
            // Check default install location
            if let Ok(dir) = ort_install_dir() {
                let lib = dir.join(ort_lib_name());
                if lib.exists() {
                    println!("  Found at default path: {}", lib.display());
                    println!("  Set: export ORT_DYLIB_PATH=\"{}\"", lib.display());
                } else {
                    println!("  NOT CONFIGURED — run `akh setup onnx-runtime`");
                }
            } else {
                println!("  NOT CONFIGURED — run `akh setup onnx-runtime`");
            }
        }
    }

    println!("\nTo download missing components, run:");
    if !ner_dir.join("model.onnx").exists() || !llm_path.exists() {
        println!("  akh setup models");
    }
    if ort_env
        .as_ref()
        .map(|p| !Path::new(p).exists())
        .unwrap_or(true)
    {
        println!("  akh setup onnx-runtime");
    }

    Ok(())
}

/// Default ONNX Runtime version used by setup.
pub fn default_ort_version() -> &'static str {
    ORT_VERSION
}

// ── Internals ──────────────────────────────────────────────────────────────

/// Download a file from `url` to `dest`, printing progress to stderr.
fn download_file(url: &str, dest: &Path) -> SetupResult<()> {
    let filename = dest
        .file_name()
        .map(|f| f.to_string_lossy().into_owned())
        .unwrap_or_else(|| "file".to_string());
    eprintln!("  Downloading {filename}...");

    let resp = ureq::get(url)
        .call()
        .map_err(|e| SetupError::Http {
            reason: format!("{e}"),
        })?;

    let content_length: Option<u64> = resp
        .header("content-length")
        .and_then(|v| v.parse().ok());

    let mut body = resp.into_reader();
    let mut out = fs::File::create(dest)?;
    let mut buf = vec![0u8; 64 * 1024];
    let mut downloaded = 0u64;
    let mut last_pct = 0u8;

    loop {
        let n = body.read(&mut buf).map_err(|e| SetupError::Http {
            reason: format!("read: {e}"),
        })?;
        if n == 0 {
            break;
        }
        out.write_all(&buf[..n])?;
        downloaded += n as u64;

        // Print progress every 5%
        if let Some(total) = content_length {
            let pct = ((downloaded * 100) / total) as u8;
            if pct >= last_pct + 5 {
                eprint!("\r  {filename}: {pct}% ({:.1} MB)", downloaded as f64 / 1_048_576.0);
                last_pct = pct;
            }
        }
    }
    out.flush()?;

    if content_length.is_some() {
        eprintln!(
            "\r  {filename}: 100% ({:.1} MB)    ",
            downloaded as f64 / 1_048_576.0
        );
    } else {
        eprintln!(
            "  {filename}: {:.1} MB",
            downloaded as f64 / 1_048_576.0
        );
    }

    Ok(())
}

/// ONNX Runtime install directory.
fn ort_install_dir() -> SetupResult<PathBuf> {
    let home =
        std::env::var("HOME").map_err(|_| SetupError::Io(io::Error::other("HOME not set")))?;

    #[cfg(target_os = "macos")]
    {
        Ok(PathBuf::from(home).join("Library/Frameworks/onnxruntime"))
    }
    #[cfg(target_os = "linux")]
    {
        Ok(PathBuf::from(home).join(".local/lib/onnxruntime"))
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        Err(SetupError::UnsupportedPlatform {
            os: std::env::consts::OS.to_string(),
            arch: std::env::consts::ARCH.to_string(),
        })
    }
}

/// Platform suffix for ONNX Runtime release archives.
fn ort_platform_suffix() -> SetupResult<&'static str> {
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    {
        Ok("osx-arm64")
    }
    #[cfg(all(target_os = "macos", target_arch = "x86_64"))]
    {
        Ok("osx-x86_64")
    }
    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    {
        Ok("linux-x64")
    }
    #[cfg(all(target_os = "linux", target_arch = "aarch64"))]
    {
        Ok("linux-aarch64")
    }
    #[cfg(not(any(
        all(target_os = "macos", target_arch = "aarch64"),
        all(target_os = "macos", target_arch = "x86_64"),
        all(target_os = "linux", target_arch = "x86_64"),
        all(target_os = "linux", target_arch = "aarch64"),
    )))]
    {
        Err(SetupError::UnsupportedPlatform {
            os: std::env::consts::OS.to_string(),
            arch: std::env::consts::ARCH.to_string(),
        })
    }
}

/// Shared library filename for the current OS.
fn ort_lib_name() -> &'static str {
    #[cfg(target_os = "macos")]
    {
        "libonnxruntime.dylib"
    }
    #[cfg(target_os = "linux")]
    {
        "libonnxruntime.so"
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        "libonnxruntime.so"
    }
}

/// Extract the ONNX Runtime shared library from a `.tgz` archive.
fn extract_ort_lib(tarball: &[u8], dest: &Path, lib_name: &str) -> SetupResult<()> {
    use flate2::read::GzDecoder;
    use tar::Archive;

    let gz = GzDecoder::new(tarball);
    let mut archive = Archive::new(gz);

    for entry in archive.entries().map_err(|e| SetupError::Http {
        reason: format!("tar read: {e}"),
    })? {
        let mut entry = entry.map_err(|e| SetupError::Http {
            reason: format!("tar entry: {e}"),
        })?;
        let path = entry.path().map_err(|e| SetupError::Http {
            reason: format!("tar path: {e}"),
        })?;
        let path_str = path.to_string_lossy();

        // Look for lib/libonnxruntime.{dylib,so} inside the archive
        if path_str.ends_with(lib_name) && path_str.contains("lib/") {
            let mut data = Vec::new();
            entry.read_to_end(&mut data).map_err(|e| SetupError::Http {
                reason: format!("tar extract: {e}"),
            })?;
            fs::write(dest, &data)?;
            eprintln!(
                "  Extracted {} ({:.1} MB)",
                lib_name,
                data.len() as f64 / 1_048_576.0
            );
            return Ok(());
        }
    }

    Err(SetupError::Http {
        reason: format!("{lib_name} not found in archive"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ort_platform_suffix_resolves() {
        // Should not error on any supported build target.
        let result = ort_platform_suffix();
        assert!(result.is_ok(), "platform suffix should resolve: {result:?}");
    }

    #[test]
    fn ort_lib_name_is_valid() {
        let name = ort_lib_name();
        assert!(
            name.starts_with("libonnxruntime"),
            "unexpected lib name: {name}"
        );
    }

    #[test]
    fn models_dir_under_data_dir() {
        let paths = AkhPaths::resolve().unwrap();
        let models = paths.models_dir();
        assert!(
            models.starts_with(&paths.data_dir),
            "models dir should be under data_dir"
        );
        assert!(models.ends_with("models"));
    }
}
