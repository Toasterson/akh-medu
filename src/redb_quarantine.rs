//! Bounded retention for quarantined host-local redb images (ADR 049).
//!
//! On every authoritative hydrate, `akhomed` moves the workspace's host-local
//! `akh-medu.redb` aside as `akh-medu.pre-residency-<epoch>[-<attempt>].redb`
//! (forensic evidence: Postgres is the truth, the local image is only kept in
//! case the hydrate goes wrong). Nothing ever pruned those copies, so a long
//! lived host grew one full redb image per restart per akh.
//!
//! This module owns the retention policy:
//!
//! * [`QuarantineKeep`] is the per-workspace bound. It is a `NonZeroU32`
//!   newtype: **`0` is refused** at config load. "Keep none" would delete the
//!   just-quarantined image (the only copy of the host-local state) the moment
//!   the hydrate succeeds, defeating the quarantine's purpose; an operator who
//!   really wants that can delete the files by hand.
//! * [`prune_after_hydrate`] runs the pruning only when the hydrate outcome is
//!   `Ok`, so a failed wake leaves every copy in place.
//! * [`prune_quarantined_redbs`] deletes only regular files whose name matches
//!   exactly the quarantine pattern in one directory. Symlinks and anything
//!   else are never touched, and nothing is followed.

use std::num::NonZeroU32;
use std::path::{Path, PathBuf};

use miette::Diagnostic;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use thiserror::Error;

const QUARANTINE_PREFIX: &str = "akh-medu.pre-residency-";
const QUARANTINE_SUFFIX: &str = ".redb";

/// How many quarantined redb images to keep per workspace.
///
/// Serialized as a plain integer; `0` is rejected on load.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QuarantineKeep(NonZeroU32);

impl QuarantineKeep {
    /// Default retention: the newest three images per workspace.
    pub const DEFAULT: Self = match NonZeroU32::new(3) {
        Some(n) => Self(n),
        None => unreachable!(),
    };

    /// Build from a raw count, refusing `0`.
    pub fn new(count: u32) -> Result<Self, QuarantineRetentionError> {
        NonZeroU32::new(count)
            .map(Self)
            .ok_or(QuarantineRetentionError::ZeroKeep)
    }

    /// The bound as a count.
    pub fn get(self) -> u32 {
        self.0.get()
    }

    /// True when this is the default (used to keep default configs terse).
    pub fn is_default(&self) -> bool {
        *self == Self::DEFAULT
    }
}

impl Default for QuarantineKeep {
    fn default() -> Self {
        Self::DEFAULT
    }
}

impl Serialize for QuarantineKeep {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_u32(self.get())
    }
}

impl<'de> Deserialize<'de> for QuarantineKeep {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let raw = u32::deserialize(deserializer)?;
        Self::new(raw).map_err(serde::de::Error::custom)
    }
}

/// Errors from the quarantine retention policy.
#[derive(Debug, Error, Diagnostic)]
pub enum QuarantineRetentionError {
    #[error("redb_quarantine_keep must be at least 1 (got 0)")]
    #[diagnostic(
        code(akh::quarantine::zero_keep),
        help(
            "Keeping none would delete the just-quarantined host-local redb as soon as the \
             hydrate succeeds. Use a positive count (default 3), or delete the files by hand."
        )
    )]
    ZeroKeep,

    #[error("cannot list quarantined redb images in {dir}")]
    #[diagnostic(
        code(akh::quarantine::read_dir),
        help(
            "Retention is best-effort; the images are left in place. Check the directory permissions."
        )
    )]
    ReadDir {
        dir: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("cannot remove quarantined redb image {path}")]
    #[diagnostic(
        code(akh::quarantine::remove),
        help("The image is left in place and will be retried on the next start.")
    )]
    Remove {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
}

/// One quarantined image that was deleted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrunedImage {
    pub path: PathBuf,
    pub bytes: u64,
}

/// What a prune pass did.
#[derive(Debug, Default)]
pub struct PruneReport {
    /// Images deleted, oldest first.
    pub pruned: Vec<PrunedImage>,
    /// Images that matched for deletion but could not be removed.
    pub failures: Vec<QuarantineRetentionError>,
}

impl PruneReport {
    /// Total bytes reclaimed by the deleted images.
    pub fn reclaimed_bytes(&self) -> u64 {
        self.pruned.iter().map(|p| p.bytes).sum()
    }
}

/// Sort key parsed from an exact quarantine file name:
/// `akh-medu.pre-residency-<epoch>.redb` or `…-<epoch>-<attempt>.redb`.
fn parse_quarantine_name(name: &str) -> Option<(u64, u64)> {
    let middle = name
        .strip_prefix(QUARANTINE_PREFIX)?
        .strip_suffix(QUARANTINE_SUFFIX)?;
    let all_digits = |s: &str| !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit());
    match middle.split_once('-') {
        None if all_digits(middle) => Some((middle.parse().ok()?, 0)),
        Some((epoch, attempt)) if all_digits(epoch) && all_digits(attempt) => {
            Some((epoch.parse().ok()?, attempt.parse().ok()?))
        }
        _ => None,
    }
}

/// Delete all but the newest `keep` quarantined redb images in `kg_dir`.
///
/// "Newest" is the residency epoch (monotonic, claimed in Postgres), then the
/// retry attempt within that epoch. Only regular files with an exact
/// quarantine name are considered; symlinks and everything else are ignored.
/// A per-file removal failure is recorded in the report and does not stop the
/// pass. A missing directory is an empty pass.
pub fn prune_quarantined_redbs(
    kg_dir: &Path,
    keep: QuarantineKeep,
) -> Result<PruneReport, QuarantineRetentionError> {
    let read = match std::fs::read_dir(kg_dir) {
        Ok(read) => read,
        Err(source) if source.kind() == std::io::ErrorKind::NotFound => {
            return Ok(PruneReport::default());
        }
        Err(source) => {
            return Err(QuarantineRetentionError::ReadDir {
                dir: kg_dir.to_path_buf(),
                source,
            });
        }
    };

    let mut images: Vec<((u64, u64), String, PathBuf, u64)> = Vec::new();
    for entry in read {
        let entry = entry.map_err(|source| QuarantineRetentionError::ReadDir {
            dir: kg_dir.to_path_buf(),
            source,
        })?;
        let Ok(file_type) = entry.file_type() else {
            continue;
        };
        // `DirEntry::file_type` does not follow symlinks: a link reports
        // `is_symlink`, never `is_file`.
        if !file_type.is_file() {
            continue;
        }
        let Some(name) = entry.file_name().to_str().map(str::to_owned) else {
            continue;
        };
        let Some(key) = parse_quarantine_name(&name) else {
            continue;
        };
        let bytes = entry.metadata().map(|m| m.len()).unwrap_or(0);
        images.push((key, name, entry.path(), bytes));
    }

    // Oldest first; the name breaks ties between e.g. `-7` and `-07`.
    images.sort_by(|a, b| (a.0, &a.1).cmp(&(b.0, &b.1)));
    let excess = images.len().saturating_sub(keep.get() as usize);

    let mut report = PruneReport::default();
    for (_, _, path, bytes) in images.into_iter().take(excess) {
        match std::fs::remove_file(&path) {
            Ok(()) => report.pruned.push(PrunedImage { path, bytes }),
            Err(source) => report
                .failures
                .push(QuarantineRetentionError::Remove { path, source }),
        }
    }
    Ok(report)
}

/// Prune only if the hydrate succeeded; pass the outcome through untouched.
///
/// A failed wake returns its error and leaves every quarantined image. On
/// success the prune is best-effort: its failures never fail the wake, they
/// are returned in the report (or as the `Err` of the second tuple element
/// when the directory itself cannot be listed) for the caller to log.
pub fn prune_after_hydrate<T, E>(
    hydrated: Result<T, E>,
    kg_dir: &Path,
    keep: QuarantineKeep,
) -> Result<(T, Result<PruneReport, QuarantineRetentionError>), E> {
    let value = hydrated?;
    Ok((value, prune_quarantined_redbs(kg_dir, keep)))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn touch(dir: &Path, name: &str, len: usize) {
        std::fs::write(dir.join(name), vec![0u8; len]).unwrap();
    }

    fn names(dir: &Path) -> Vec<String> {
        let mut v: Vec<String> = std::fs::read_dir(dir)
            .unwrap()
            .map(|e| e.unwrap().file_name().into_string().unwrap())
            .collect();
        v.sort();
        v
    }

    fn keep(n: u32) -> QuarantineKeep {
        QuarantineKeep::new(n).unwrap()
    }

    #[test]
    fn keeps_the_newest_k_by_epoch_not_lexically() {
        let tmp = tempfile::tempdir().unwrap();
        for epoch in [2, 9, 10, 11, 100] {
            touch(
                tmp.path(),
                &format!("akh-medu.pre-residency-{epoch}.redb"),
                10,
            );
        }
        let report = prune_quarantined_redbs(tmp.path(), keep(3)).unwrap();
        assert_eq!(report.pruned.len(), 2);
        assert_eq!(report.reclaimed_bytes(), 20);
        assert_eq!(
            names(tmp.path()),
            vec![
                "akh-medu.pre-residency-10.redb",
                "akh-medu.pre-residency-100.redb",
                "akh-medu.pre-residency-11.redb",
            ]
        );
    }

    #[test]
    fn retry_attempts_sort_after_the_base_image_of_the_same_epoch() {
        let tmp = tempfile::tempdir().unwrap();
        touch(tmp.path(), "akh-medu.pre-residency-5.redb", 1);
        touch(tmp.path(), "akh-medu.pre-residency-5-1.redb", 1);
        touch(tmp.path(), "akh-medu.pre-residency-5-2.redb", 1);
        touch(tmp.path(), "akh-medu.pre-residency-4.redb", 1);
        prune_quarantined_redbs(tmp.path(), keep(2)).unwrap();
        assert_eq!(
            names(tmp.path()),
            vec![
                "akh-medu.pre-residency-5-1.redb",
                "akh-medu.pre-residency-5-2.redb"
            ]
        );
    }

    #[test]
    fn ignores_non_matching_files_and_directories() {
        let tmp = tempfile::tempdir().unwrap();
        for i in 1..=4 {
            touch(tmp.path(), &format!("akh-medu.pre-residency-{i}.redb"), 1);
        }
        let others = [
            "akh-medu.redb",
            "akh-medu.pre-residency-.redb",
            "akh-medu.pre-residency-x.redb",
            "akh-medu.pre-residency-1.redb.bak",
            "akh-medu.pre-residency-1-.redb",
            "akh-medu.pre-residency-1-2-3.redb",
            "other.pre-residency-1.redb",
            "akh-medu.pre-residency-1.txt",
        ];
        for name in others {
            touch(tmp.path(), name, 1);
        }
        std::fs::create_dir(tmp.path().join("akh-medu.pre-residency-0.redb")).unwrap();

        let report = prune_quarantined_redbs(tmp.path(), keep(1)).unwrap();
        assert_eq!(report.pruned.len(), 3);
        let left = names(tmp.path());
        for name in others {
            assert!(left.contains(&name.to_string()), "{name} must survive");
        }
        assert!(left.contains(&"akh-medu.pre-residency-0.redb".to_string()));
        assert!(left.contains(&"akh-medu.pre-residency-4.redb".to_string()));
    }

    #[cfg(unix)]
    #[test]
    fn never_follows_or_removes_symlinks() {
        let tmp = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        let target = outside.path().join("precious.redb");
        std::fs::write(&target, b"keep me").unwrap();
        std::os::unix::fs::symlink(&target, tmp.path().join("akh-medu.pre-residency-1.redb"))
            .unwrap();
        touch(tmp.path(), "akh-medu.pre-residency-2.redb", 1);
        touch(tmp.path(), "akh-medu.pre-residency-3.redb", 1);

        let report = prune_quarantined_redbs(tmp.path(), keep(1)).unwrap();
        assert_eq!(report.pruned.len(), 1, "only the real file 2 is excess");
        assert!(target.exists());
        assert!(
            tmp.path()
                .join("akh-medu.pre-residency-1.redb")
                .symlink_metadata()
                .is_ok(),
            "the link itself is not a candidate"
        );
    }

    #[test]
    fn fewer_than_k_and_missing_dir_are_no_ops() {
        let tmp = tempfile::tempdir().unwrap();
        touch(tmp.path(), "akh-medu.pre-residency-1.redb", 1);
        assert!(
            prune_quarantined_redbs(tmp.path(), keep(3))
                .unwrap()
                .pruned
                .is_empty()
        );
        let missing = tmp.path().join("absent");
        assert!(
            prune_quarantined_redbs(&missing, keep(3))
                .unwrap()
                .pruned
                .is_empty()
        );
    }

    #[test]
    fn failed_hydrate_prunes_nothing_and_passes_the_error_through() {
        let tmp = tempfile::tempdir().unwrap();
        for i in 1..=6 {
            touch(tmp.path(), &format!("akh-medu.pre-residency-{i}.redb"), 1);
        }
        let outcome: Result<(), &str> = Err("hydrate failed");
        let result = prune_after_hydrate(outcome, tmp.path(), keep(1));
        assert_eq!(result.unwrap_err(), "hydrate failed");
        assert_eq!(names(tmp.path()).len(), 6);
    }

    #[test]
    fn successful_hydrate_prunes_and_returns_the_value() {
        let tmp = tempfile::tempdir().unwrap();
        for i in 1..=6 {
            touch(tmp.path(), &format!("akh-medu.pre-residency-{i}.redb"), 2);
        }
        let outcome: Result<u32, &str> = Ok(42);
        let (value, report) = prune_after_hydrate(outcome, tmp.path(), keep(2)).unwrap();
        assert_eq!(value, 42);
        let report = report.unwrap();
        assert_eq!(report.pruned.len(), 4);
        assert_eq!(report.reclaimed_bytes(), 8);
        assert_eq!(names(tmp.path()).len(), 2);
    }

    #[test]
    fn zero_keep_is_refused_with_a_diagnostic() {
        assert!(matches!(
            QuarantineKeep::new(0),
            Err(QuarantineRetentionError::ZeroKeep)
        ));
        #[derive(Deserialize)]
        struct Holder {
            #[allow(dead_code)]
            k: QuarantineKeep,
        }
        assert!(toml::from_str::<Holder>("k = 0").is_err());
        let ok: Holder = toml::from_str("k = 5").unwrap();
        assert_eq!(ok.k.get(), 5);
        assert_eq!(QuarantineKeep::default().get(), 3);
    }
}
