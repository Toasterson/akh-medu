//! Load a skillpack's natural-language procedures by path (53g PR 7).
//!
//! Some procedures are not activated into the akh's own knowledge graph but
//! handed to someone else: the akh delegates a task to a native worker and
//! sends the skill's `procedures.md` as the worker's instructions. Keeping
//! that text in a skillpack on disk (not in Rust) lets the operator revise a
//! worker policy without a rebuild, and keeps natural language out of code.
//!
//! [`SkillProcedures::load`] reads `skill.json` from the directory, then the
//! file its `procedures_file` names (default `procedures.md`). The file must
//! stay inside the skill directory and is bounded in size.

use std::path::{Component, Path, PathBuf};

use crate::error::{SkillError, SkillResult};

use super::SkillManifest;

/// The procedures file used when the manifest names none.
pub const DEFAULT_PROCEDURES_FILE: &str = "procedures.md";

/// Upper bound on a procedures file. Worker instructions are a page or two;
/// anything larger is a packaging mistake, not a policy.
pub const MAX_PROCEDURES_BYTES: u64 = 64 * 1024;

/// A skillpack's manifest and its procedures text.
#[derive(Debug, Clone)]
pub struct SkillProcedures {
    /// The skill directory the procedures were read from.
    pub dir: PathBuf,
    /// The parsed `skill.json`.
    pub manifest: SkillManifest,
    /// The procedures text, exactly as on disk.
    pub text: String,
}

impl SkillProcedures {
    /// Read `dir/skill.json` and its procedures file.
    ///
    /// # Errors
    ///
    /// [`SkillError::InvalidManifest`] for a missing or malformed manifest, a
    /// procedures path that escapes the directory, an oversized or empty
    /// procedures file; [`SkillError::Io`] when a file cannot be read.
    pub fn load(dir: &Path) -> SkillResult<Self> {
        let manifest_path = dir.join("skill.json");
        let raw = std::fs::read_to_string(&manifest_path).map_err(|source| SkillError::Io {
            skill_id: dir.display().to_string(),
            source,
        })?;
        let manifest: SkillManifest =
            serde_json::from_str(&raw).map_err(|error| SkillError::InvalidManifest {
                path: manifest_path.display().to_string(),
                message: error.to_string(),
            })?;
        let relative = manifest
            .procedures_file
            .as_deref()
            .unwrap_or(DEFAULT_PROCEDURES_FILE);
        let invalid = |message: String| SkillError::InvalidManifest {
            path: manifest_path.display().to_string(),
            message,
        };
        if Path::new(relative)
            .components()
            .any(|c| !matches!(c, Component::Normal(_)))
        {
            return Err(invalid(format!(
                "procedures_file `{relative}` must be a relative path inside the skill directory"
            )));
        }
        let path = dir.join(relative);
        let length = std::fs::metadata(&path)
            .map_err(|source| SkillError::Io {
                skill_id: manifest.id.clone(),
                source,
            })?
            .len();
        if length > MAX_PROCEDURES_BYTES {
            return Err(invalid(format!(
                "procedures_file `{relative}` is {length} bytes; the limit is {MAX_PROCEDURES_BYTES}"
            )));
        }
        let text = std::fs::read_to_string(&path).map_err(|source| SkillError::Io {
            skill_id: manifest.id.clone(),
            source,
        })?;
        if text.trim().is_empty() {
            return Err(invalid(format!("procedures_file `{relative}` is empty")));
        }
        Ok(Self {
            dir: dir.to_path_buf(),
            manifest,
            text,
        })
    }

    /// `id@version`, for provenance and status output.
    pub fn label(&self) -> String {
        format!("{}@{}", self.manifest.id, self.manifest.version)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn skill(dir: &Path, manifest: serde_json::Value, procedures: Option<(&str, &str)>) {
        std::fs::create_dir_all(dir).unwrap();
        std::fs::write(dir.join("skill.json"), manifest.to_string()).unwrap();
        if let Some((name, text)) = procedures {
            std::fs::write(dir.join(name), text).unwrap();
        }
    }

    fn manifest(procedures_file: Option<&str>) -> serde_json::Value {
        let mut value = serde_json::json!({
            "id": "worker", "name": "Worker", "version": "1.0.0", "domains": ["delegation"]
        });
        if let Some(file) = procedures_file {
            value["procedures_file"] = serde_json::json!(file);
        }
        value
    }

    #[test]
    fn loads_the_default_and_a_named_procedures_file() {
        let dir = tempfile::tempdir().unwrap();
        skill(
            dir.path(),
            manifest(None),
            Some(("procedures.md", "step one\n")),
        );
        let loaded = SkillProcedures::load(dir.path()).unwrap();
        assert_eq!(loaded.text, "step one\n");
        assert_eq!(loaded.label(), "worker@1.0.0");

        let other = dir.path().join("other");
        skill(
            &other,
            manifest(Some("policy.md")),
            Some(("policy.md", "policy\n")),
        );
        assert_eq!(SkillProcedures::load(&other).unwrap().text, "policy\n");
    }

    #[test]
    fn rejects_escapes_empty_oversized_and_missing_files() {
        let dir = tempfile::tempdir().unwrap();
        skill(dir.path(), manifest(Some("../secret.md")), None);
        assert!(matches!(
            SkillProcedures::load(dir.path()),
            Err(SkillError::InvalidManifest { .. })
        ));
        skill(dir.path(), manifest(Some("/etc/passwd")), None);
        assert!(matches!(
            SkillProcedures::load(dir.path()),
            Err(SkillError::InvalidManifest { .. })
        ));
        skill(dir.path(), manifest(None), Some(("procedures.md", "  \n")));
        assert!(matches!(
            SkillProcedures::load(dir.path()),
            Err(SkillError::InvalidManifest { .. })
        ));
        let big = "x".repeat(usize::try_from(MAX_PROCEDURES_BYTES).unwrap() + 1);
        skill(dir.path(), manifest(None), Some(("procedures.md", &big)));
        assert!(matches!(
            SkillProcedures::load(dir.path()),
            Err(SkillError::InvalidManifest { .. })
        ));
        let empty = tempfile::tempdir().unwrap();
        assert!(matches!(
            SkillProcedures::load(empty.path()),
            Err(SkillError::Io { .. })
        ));
        std::fs::write(empty.path().join("skill.json"), "{").unwrap();
        assert!(matches!(
            SkillProcedures::load(empty.path()),
            Err(SkillError::InvalidManifest { .. })
        ));
    }
}
