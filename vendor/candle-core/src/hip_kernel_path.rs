//! Relocatable lookup for the HIP kernel libraries built with this runtime.

use std::path::{Path, PathBuf};

/// Prefer an explicit absolute bundle directory, then `lib/` beside the
/// executable, then the compile-time development artifact. An explicit override
/// is authoritative: a missing library is an error, never a fallback. Callers
/// must treat a dlopen/symbol failure at the selected path the same way.
pub fn resolve(name: &str, development: &str) -> Result<PathBuf, String> {
    let override_dir = std::env::var_os("AKH_HIP_KERNEL_DIR").map(PathBuf::from);
    let executable = std::env::current_exe()
        .map_err(|error| format!("cannot locate executable for HIP kernel bundle: {error}"))?;
    select(
        name,
        override_dir.as_deref(),
        &executable,
        Path::new(development),
        Path::is_file,
    )
}

fn select(
    name: &str,
    override_dir: Option<&Path>,
    executable: &Path,
    development: &Path,
    exists: impl Fn(&Path) -> bool,
) -> Result<PathBuf, String> {
    if name.is_empty() || Path::new(name).file_name() != Some(std::ffi::OsStr::new(name)) {
        return Err(format!("HIP kernel name must be a basename: {name:?}"));
    }
    if let Some(directory) = override_dir {
        if !directory.is_absolute() {
            return Err("AKH_HIP_KERNEL_DIR must be an absolute directory".into());
        }
        let path = directory.join(name);
        return if exists(&path) {
            Ok(path)
        } else {
            Err(format!(
                "explicit HIP kernel bundle is missing {}",
                path.display()
            ))
        };
    }
    let bundle = executable
        .parent()
        .ok_or_else(|| "executable has no parent directory".to_string())?
        .join("lib")
        .join(name);
    if exists(&bundle) {
        return Ok(bundle);
    }
    if development.is_absolute() && exists(development) {
        return Ok(development.to_path_buf());
    }
    Err(format!("HIP kernel {name} missing from {} and {}; set AKH_HIP_KERNEL_DIR to an absolute matching bundle directory", bundle.display(), development.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lookup(override_dir: Option<&str>, files: &[&str]) -> Result<PathBuf, String> {
        select(
            "libtest.so",
            override_dir.map(Path::new),
            Path::new("/opt/runtime/akhrtd"),
            Path::new("/build/libtest.so"),
            |path| files.iter().any(|file| path == Path::new(file)),
        )
    }

    #[test]
    fn installed_bundle_precedes_development_artifact() {
        assert_eq!(
            lookup(None, &["/opt/runtime/lib/libtest.so", "/build/libtest.so"]).unwrap(),
            Path::new("/opt/runtime/lib/libtest.so")
        );
    }

    #[test]
    fn development_artifact_remains_available_without_bundle() {
        assert_eq!(
            lookup(None, &["/build/libtest.so"]).unwrap(),
            Path::new("/build/libtest.so")
        );
    }

    #[test]
    fn explicit_directory_wins_and_never_falls_back() {
        assert_eq!(
            lookup(
                Some("/pinned"),
                &[
                    "/pinned/libtest.so",
                    "/opt/runtime/lib/libtest.so",
                    "/build/libtest.so"
                ]
            )
            .unwrap(),
            Path::new("/pinned/libtest.so")
        );
        assert!(lookup(Some("/missing"), &["/build/libtest.so"]).is_err());
        assert!(lookup(
            Some("relative"),
            &["relative/libtest.so", "/build/libtest.so"]
        )
        .is_err());
        assert!(lookup(Some(""), &["/build/libtest.so"]).is_err());
    }

    #[test]
    fn missing_files_and_non_basename_names_are_rejected() {
        assert!(lookup(None, &[]).is_err());
        for name in [
            "",
            ".",
            "..",
            "../libtest.so",
            "/libtest.so",
            "sub/libtest.so",
        ] {
            assert!(select(
                name,
                None,
                Path::new("/opt/akhrtd"),
                Path::new("/build/libtest.so"),
                |_| true
            )
            .is_err());
        }
    }
}
