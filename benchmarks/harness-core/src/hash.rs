//! Deterministic task-set content hashing.
//!
//! ADR 086 §4: "Once published, a task-set hash freezes — task improvements
//! fork a v2 series." That guarantee only holds if the hash is a pure
//! function of task *content* — independent of directory walk order,
//! absolute path prefix, or filesystem metadata (mtimes, permissions).
//!
//! The scheme: collect `(relative_path, content_bytes)` pairs, sort by
//! relative path (byte order), then feed a single blake3 hasher
//! `relative_path_bytes ++ 0x00 ++ content_len_le ++ content_bytes` per
//! entry in sorted order. Sorting makes the result independent of collection
//! order; using relative paths (not absolute) makes it independent of where
//! the task-set checkout lives; the length-prefix prevents path/content
//! boundary ambiguity (`"ab" + "c"` vs `"a" + "bc"`).

use std::path::{Path, PathBuf};

use walkdir::WalkDir;

use crate::error::{HarnessError, HarnessResult};
use crate::schema::{Blake3Hex, TaskSetHash};

/// One file's worth of task-set content, keyed by a path relative to the
/// task-set root (forward-slash separated so the hash is stable across
/// Windows/Unix path separators, though this harness only targets Unix).
pub struct TaskFile {
    pub relative_path: String,
    pub content: Vec<u8>,
}

/// Hash an explicit, caller-provided list of task files. Order of `files`
/// does not affect the result (they are sorted internally by
/// `relative_path`), so callers can pass them in whatever order is
/// convenient (e.g. directory-walk order, or a hand-curated list).
pub fn hash_task_set(files: &[TaskFile]) -> TaskSetHash {
    let mut sorted: Vec<&TaskFile> = files.iter().collect();
    sorted.sort_by(|a, b| a.relative_path.cmp(&b.relative_path));

    let mut hasher = blake3::Hasher::new();
    for file in sorted {
        hasher.update(file.relative_path.as_bytes());
        hasher.update(&[0u8]);
        hasher.update(&(file.content.len() as u64).to_le_bytes());
        hasher.update(&file.content);
    }
    TaskSetHash::from_blake3(Blake3Hex::from_hash(hasher.finalize()))
}

/// Hash every regular file under `dir` (recursively), using paths relative
/// to `dir` as the stable keys. This is the convenience entry point task-set
/// authors call: `hash_task_set_dir("tests/vm/tasks/illumos-e2e-v1")`.
pub fn hash_task_set_dir(dir: &Path) -> HarnessResult<TaskSetHash> {
    let mut files = Vec::new();
    for entry in WalkDir::new(dir).sort_by_file_name() {
        let entry = entry.map_err(|e| HarnessError::Io {
            path: e.path().map(Path::to_path_buf).unwrap_or_else(|| dir.to_path_buf()),
            source: std::io::Error::other(e.to_string()),
        })?;
        if !entry.file_type().is_file() {
            continue;
        }
        let abs_path: PathBuf = entry.path().to_path_buf();
        let relative = abs_path
            .strip_prefix(dir)
            .unwrap_or(&abs_path)
            .to_string_lossy()
            .replace(std::path::MAIN_SEPARATOR, "/");
        let content = std::fs::read(&abs_path).map_err(|source| HarnessError::Io {
            path: abs_path.clone(),
            source,
        })?;
        files.push(TaskFile {
            relative_path: relative,
            content,
        });
    }
    Ok(hash_task_set(&files))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn tf(path: &str, content: &str) -> TaskFile {
        TaskFile {
            relative_path: path.to_string(),
            content: content.as_bytes().to_vec(),
        }
    }

    #[test]
    fn hash_is_independent_of_input_order() {
        let a = vec![
            tf("setup.sh", "echo setup"),
            tf("prompt.md", "do the thing"),
            tf("verify.sh", "exit 0"),
        ];
        let b = vec![
            tf("verify.sh", "exit 0"),
            tf("setup.sh", "echo setup"),
            tf("prompt.md", "do the thing"),
        ];
        assert_eq!(hash_task_set(&a), hash_task_set(&b));
    }

    #[test]
    fn hash_changes_when_content_changes() {
        let a = vec![tf("verify.sh", "exit 0")];
        let b = vec![tf("verify.sh", "exit 1")];
        assert_ne!(hash_task_set(&a), hash_task_set(&b));
    }

    #[test]
    fn hash_changes_when_a_file_is_added_or_removed() {
        let a = vec![tf("verify.sh", "exit 0")];
        let b = vec![tf("verify.sh", "exit 0"), tf("setup.sh", "echo hi")];
        assert_ne!(hash_task_set(&a), hash_task_set(&b));
    }

    #[test]
    fn hash_is_boundary_safe_across_path_and_content() {
        // Without a length prefix, ("ab", "c") and ("a", "bc") — treated as
        // raw concatenation — would collide. Confirm they don't.
        let a = vec![tf("ab", "c")];
        let b = vec![tf("a", "bc")];
        assert_ne!(hash_task_set(&a), hash_task_set(&b));
    }

    #[test]
    fn hash_task_set_dir_matches_manual_collection_and_ignores_walk_order() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("setup.sh"), "echo setup").unwrap();
        fs::write(dir.path().join("verify.sh"), "exit 0").unwrap();
        fs::create_dir(dir.path().join("nested")).unwrap();
        fs::write(dir.path().join("nested").join("prompt.md"), "do it").unwrap();

        let from_dir = hash_task_set_dir(dir.path()).unwrap();

        let manual = hash_task_set(&[
            tf("nested/prompt.md", "do it"),
            tf("setup.sh", "echo setup"),
            tf("verify.sh", "exit 0"),
        ]);
        assert_eq!(from_dir, manual);
    }
}
