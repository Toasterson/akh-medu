//! Support for the WI-262 verification tests: locate the fixture crate and
//! read its sources as (repo-relative path, source) pairs.

use std::path::{Path, PathBuf};

/// Absolute path of `tests/fixtures/demo_crate`.
pub fn demo_crate_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/demo_crate")
}

/// The crate root we present to the resolver: repo-relative paths *under* the
/// fixture crate root (`src/alpha.rs` etc.), which is what `FILE_PATH` keys
/// the module chain off.
pub const VIRTUAL_REPO: &str = "demo_crate";

/// All fixture files as (`repo_relative_path`, `source`) pairs.
pub fn demo_crate_files() -> Vec<(String, String)> {
    let root = demo_crate_root();
    let mut out = Vec::new();
    let mut stack = vec![root.clone()];
    while let Some(dir) = stack.pop() {
        for entry in std::fs::read_dir(&dir).expect("fixture dir readable") {
            let entry = entry.expect("fixture entry readable");
            let p = entry.path();
            if p.is_dir() {
                stack.push(p);
            } else if p.extension().is_some_and(|e| e == "rs") {
                let rel = p
                    .strip_prefix(&root)
                    .expect("fixture under root")
                    .to_string_lossy()
                    .replace('\\', "/");
                out.push((
                    rel,
                    std::fs::read_to_string(&p).expect("fixture file readable"),
                ));
            }
        }
    }
    out.sort();
    out
}

/// One fixture file by repo-relative path, e.g. `fixture("src/consumer.rs")`.
pub fn fixture(rel: &str) -> String {
    std::fs::read_to_string(demo_crate_root().join(rel)).expect("fixture file readable")
}
