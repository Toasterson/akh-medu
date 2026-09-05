//! Skill sharing protocol: export and import `.akhskill` archives.
//!
//! Phase 31f — skills are portable, self-contained archives (`.akhskill`,
//! a gzip-compressed tar) with an embedded `METADATA.json` carrying a CRC-32
//! integrity checksum. Recipes can reference remote URLs; the import path
//! validates and installs any archive.
//!
//! Archive layout:
//! ```text
//! my-skill.akhskill (tar.gz)
//! ├── METADATA.json     ← checksum + manifest summary
//! ├── skill.json        ← SkillManifest
//! ├── triples.json      ← triples (optional)
//! ├── rules.txt         ← rules (optional)
//! └── curriculum.json   ← curriculum (optional)
//! ```

use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use flate2::Compression;
use flate2::Crc;
use flate2::read::GzDecoder;
use flate2::write::GzEncoder;
use serde::{Deserialize, Serialize};

use crate::error::{SkillError, SkillResult};

use super::SkillManifest;

/// Metadata embedded in the `METADATA.json` file inside a `.akhskill` archive.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillArchiveMetadata {
    /// Archive format version (currently `1`).
    pub format_version: u32,
    /// Skill ID from the manifest.
    pub skill_id: String,
    /// Skill version from the manifest.
    pub skill_version: String,
    /// CRC-32 checksum of the raw `skill.json` content.
    pub manifest_checksum: u32,
    /// CRC-32 checksum of all content files concatenated (skill.json +
    /// triples.json + rules.txt + curriculum.json), in that order, skipping
    /// absent files.
    pub content_checksum: u32,
}

/// Compute a CRC-32 checksum over a byte slice.
fn crc32(data: &[u8]) -> u32 {
    let mut crc = Crc::new();
    crc.update(data);
    crc.sum()
}

/// Export a skill directory into a `.akhskill` archive at `output_path`.
///
/// Creates a gzip-compressed tar containing `METADATA.json`, `skill.json`,
/// and any optional content files present in the skill directory.
///
/// Returns the path to the created archive.
pub fn export_skill(skill_dir: &Path, output_path: &Path) -> SkillResult<PathBuf> {
    // ── Read skill.json ───────────────────────────────────────────────────
    let manifest_path = skill_dir.join("skill.json");
    let manifest_bytes = std::fs::read(&manifest_path).map_err(|e| SkillError::Io {
        skill_id: "<export>".into(),
        source: e,
    })?;
    let manifest: SkillManifest =
        serde_json::from_slice(&manifest_bytes).map_err(|e| SkillError::InvalidManifest {
            path: manifest_path.display().to_string(),
            message: e.to_string(),
        })?;

    // ── Collect optional content files ────────────────────────────────────
    let optional_files: &[&str] = &["triples.json", "rules.txt", "curriculum.json"];

    let mut content_crc = Crc::new();
    content_crc.update(&manifest_bytes);

    let manifest_checksum = crc32(&manifest_bytes);

    for &fname in optional_files {
        let fpath = skill_dir.join(fname);
        if fpath.exists() {
            let data = std::fs::read(&fpath).map_err(|e| SkillError::Io {
                skill_id: manifest.id.clone(),
                source: e,
            })?;
            content_crc.update(&data);
        }
    }

    let content_checksum = content_crc.sum();

    // ── Build METADATA.json ───────────────────────────────────────────────
    let metadata = SkillArchiveMetadata {
        format_version: 1,
        skill_id: manifest.id.clone(),
        skill_version: manifest.version.clone(),
        manifest_checksum,
        content_checksum,
    };
    let metadata_bytes =
        serde_json::to_vec_pretty(&metadata).map_err(|e| SkillError::ArchiveError {
            message: format!("metadata serialization failed: {e}"),
        })?;

    // ── Write tar.gz archive ──────────────────────────────────────────────
    let output_file = std::fs::File::create(output_path).map_err(|e| SkillError::Io {
        skill_id: manifest.id.clone(),
        source: e,
    })?;
    let gz = GzEncoder::new(output_file, Compression::best());
    let mut builder = tar::Builder::new(gz);

    // Append METADATA.json.
    append_bytes_as_file(&mut builder, "METADATA.json", &metadata_bytes, &manifest.id)?;

    // Append skill.json.
    append_bytes_as_file(&mut builder, "skill.json", &manifest_bytes, &manifest.id)?;

    // Append optional files.
    for &fname in optional_files {
        let fpath = skill_dir.join(fname);
        if fpath.exists() {
            let data = std::fs::read(&fpath).map_err(|e| SkillError::Io {
                skill_id: manifest.id.clone(),
                source: e,
            })?;
            append_bytes_as_file(&mut builder, fname, &data, &manifest.id)?;
        }
    }

    let gz = builder.into_inner().map_err(|e| SkillError::ArchiveError {
        message: format!("tar finalize failed: {e}"),
    })?;
    gz.finish().map_err(|e| SkillError::ArchiveError {
        message: format!("gz finalize failed: {e}"),
    })?;

    Ok(output_path.to_path_buf())
}

/// Import a `.akhskill` archive into `skills_dir`.
///
/// Steps:
/// 1. Extract and parse `METADATA.json`.
/// 2. Recompute checksums and compare to the stored values.
/// 3. Extract all skill files to `skills_dir/{skill_id}/`.
/// 4. Return the installed skill ID.
pub fn import_skill(archive_path: &Path, skills_dir: &Path) -> SkillResult<String> {
    // ── Pass 1: read all entries into memory ──────────────────────────────
    let archive_file = std::fs::File::open(archive_path).map_err(|e| SkillError::Io {
        skill_id: "<import>".into(),
        source: e,
    })?;
    let gz = GzDecoder::new(archive_file);
    let mut archive = tar::Archive::new(gz);

    let mut entries: std::collections::HashMap<String, Vec<u8>> = std::collections::HashMap::new();

    for entry in archive.entries().map_err(|e| SkillError::ArchiveError {
        message: format!("reading archive entries: {e}"),
    })? {
        let mut entry = entry.map_err(|e| SkillError::ArchiveError {
            message: format!("reading entry: {e}"),
        })?;
        let path = entry
            .path()
            .map_err(|e| SkillError::ArchiveError {
                message: format!("entry path: {e}"),
            })?
            .to_string_lossy()
            .to_string();
        let mut data = Vec::new();
        entry.read_to_end(&mut data).map_err(|e| SkillError::Io {
            skill_id: "<import>".into(),
            source: e,
        })?;
        entries.insert(path, data);
    }

    // ── Parse METADATA.json ───────────────────────────────────────────────
    let metadata_bytes = entries
        .get("METADATA.json")
        .ok_or_else(|| SkillError::ArchiveError {
            message: "archive missing METADATA.json".into(),
        })?;
    let metadata: SkillArchiveMetadata =
        serde_json::from_slice(metadata_bytes).map_err(|e| SkillError::ArchiveError {
            message: format!("METADATA.json parse error: {e}"),
        })?;

    if metadata.format_version != 1 {
        return Err(SkillError::ArchiveError {
            message: format!(
                "unsupported archive format version {}; expected 1",
                metadata.format_version
            ),
        });
    }

    // ── Parse and validate skill.json ─────────────────────────────────────
    let manifest_bytes = entries
        .get("skill.json")
        .ok_or_else(|| SkillError::ArchiveError {
            message: "archive missing skill.json".into(),
        })?;

    let manifest_checksum = crc32(manifest_bytes);
    if manifest_checksum != metadata.manifest_checksum {
        return Err(SkillError::ChecksumMismatch {
            expected: metadata.manifest_checksum,
            actual: manifest_checksum,
        });
    }

    let _manifest: SkillManifest =
        serde_json::from_slice(manifest_bytes).map_err(|e| SkillError::InvalidManifest {
            path: "skill.json (from archive)".into(),
            message: e.to_string(),
        })?;

    // ── Verify content checksum ───────────────────────────────────────────
    let optional_files: &[&str] = &["triples.json", "rules.txt", "curriculum.json"];
    let mut content_crc = Crc::new();
    content_crc.update(manifest_bytes);
    for &fname in optional_files {
        if let Some(data) = entries.get(fname) {
            content_crc.update(data);
        }
    }
    let content_checksum = content_crc.sum();
    if content_checksum != metadata.content_checksum {
        return Err(SkillError::ChecksumMismatch {
            expected: metadata.content_checksum,
            actual: content_checksum,
        });
    }

    // ── Install: write files to skills_dir/{skill_id}/ ───────────────────
    let skill_id = metadata.skill_id.clone();
    let dest_dir = skills_dir.join(&skill_id);
    std::fs::create_dir_all(&dest_dir).map_err(|e| SkillError::Io {
        skill_id: skill_id.clone(),
        source: e,
    })?;

    for (fname, data) in &entries {
        if fname == "METADATA.json" {
            continue;
        }
        let dest = dest_dir.join(fname);
        std::fs::write(&dest, data).map_err(|e| SkillError::Io {
            skill_id: skill_id.clone(),
            source: e,
        })?;
    }

    Ok(skill_id)
}

/// Helper: append in-memory bytes as a file entry in a tar builder.
fn append_bytes_as_file<W: Write>(
    builder: &mut tar::Builder<W>,
    name: &str,
    data: &[u8],
    skill_id: &str,
) -> SkillResult<()> {
    let mut header = tar::Header::new_gnu();
    header.set_size(data.len() as u64);
    header.set_mode(0o644);
    header.set_cksum();

    builder
        .append_data(&mut header, name, data)
        .map_err(|e| SkillError::ArchiveError {
            message: format!("appending {name} for skill {skill_id}: {e}"),
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn make_skill_dir(dir: &TempDir) -> PathBuf {
        let skill_dir = dir.path().join("astronomy");
        std::fs::create_dir_all(&skill_dir).unwrap();

        let manifest = serde_json::json!({
            "id": "astronomy",
            "name": "Astronomy",
            "version": "1.0.0",
            "domains": ["science"],
            "depends": [],
            "conflicts": [],
            "provides": ["celestial_bodies"],
            "requires": []
        });
        std::fs::write(
            skill_dir.join("skill.json"),
            serde_json::to_string_pretty(&manifest).unwrap(),
        )
        .unwrap();
        std::fs::write(
            skill_dir.join("triples.json"),
            r#"[{"subject":"star","predicate":"is-a","object":"celestial_body","confidence":1.0}]"#,
        )
        .unwrap();
        std::fs::write(
            skill_dir.join("rules.txt"),
            "(similar ?x ?y) => (similar ?y ?x)\n",
        )
        .unwrap();

        skill_dir
    }

    #[test]
    fn export_and_import_roundtrip() {
        let src_dir = TempDir::new().unwrap();
        let skill_dir = make_skill_dir(&src_dir);

        let archive_path = src_dir.path().join("astronomy.akhskill");
        export_skill(&skill_dir, &archive_path).unwrap();
        assert!(archive_path.exists());

        let dest_dir = TempDir::new().unwrap();
        let skills_dest = dest_dir.path().to_path_buf();
        let skill_id = import_skill(&archive_path, &skills_dest).unwrap();
        assert_eq!(skill_id, "astronomy");

        // The installed skill directory should contain skill.json.
        assert!(skills_dest.join("astronomy").join("skill.json").exists());
        assert!(skills_dest.join("astronomy").join("triples.json").exists());
        assert!(skills_dest.join("astronomy").join("rules.txt").exists());
    }

    #[test]
    fn tampered_archive_fails_checksum() {
        use flate2::Compression;
        use flate2::read::GzDecoder;
        use flate2::write::GzEncoder;

        let src_dir = TempDir::new().unwrap();
        let skill_dir = make_skill_dir(&src_dir);

        // Export a valid archive.
        let valid_path = src_dir.path().join("valid.akhskill");
        export_skill(&skill_dir, &valid_path).unwrap();

        // Decompress and collect all entries.
        let valid_bytes = std::fs::read(&valid_path).unwrap();
        let gz = GzDecoder::new(valid_bytes.as_slice());
        let mut tar_archive = tar::Archive::new(gz);
        let mut entries: std::collections::HashMap<String, Vec<u8>> =
            std::collections::HashMap::new();
        for entry in tar_archive.entries().unwrap() {
            let mut entry = entry.unwrap();
            let name = entry.path().unwrap().to_string_lossy().to_string();
            let mut data = Vec::new();
            entry.read_to_end(&mut data).unwrap();
            entries.insert(name, data);
        }

        // Corrupt triples.json while keeping the original METADATA.json (stale checksums).
        if let Some(triples) = entries.get_mut("triples.json") {
            assert!(!triples.is_empty(), "triples.json should not be empty");
            triples[0] ^= 0xFF;
        } else {
            panic!("triples.json not found in archive");
        }

        // Repack into a tampered archive (METADATA.json checksums will no longer match).
        let tampered_path = src_dir.path().join("tampered.akhskill");
        {
            let out_file = std::fs::File::create(&tampered_path).unwrap();
            let gz_enc = GzEncoder::new(out_file, Compression::best());
            let mut builder = tar::Builder::new(gz_enc);
            for name in &[
                "METADATA.json",
                "skill.json",
                "triples.json",
                "rules.txt",
                "curriculum.json",
            ] {
                if let Some(data) = entries.get(*name) {
                    super::append_bytes_as_file(&mut builder, name, data, "astronomy").unwrap();
                }
            }
            builder.into_inner().unwrap().finish().unwrap();
        }

        let dest_dir = TempDir::new().unwrap();
        let result = import_skill(&tampered_path, dest_dir.path());
        assert!(
            result.is_err(),
            "tampered archive should fail checksum validation"
        );
    }
}
