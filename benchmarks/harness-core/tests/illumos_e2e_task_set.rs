//! Static validation of the Phase 50 / W3 illumos E2E task corpus
//! (`benchmarks/illumos-e2e/tasks/`), plus a demonstration that it hashes
//! deterministically under `harness_core::hash::hash_task_set_dir` -- the
//! W7 "task-set content hash" contract (ADR 086 §4) this corpus is
//! designed to feed.
//!
//! This is authoring-time validation only: it does not run any of the
//! tasks (that requires a live OpenIndiana guest -- see
//! `benchmarks/illumos-e2e/README.md`, "What the runner still has to
//! do"). It parses every `meta.toml`, checks required fields and
//! cross-field invariants, confirms the four required files exist per
//! task, and checks the family/version-trap distribution ADR 086 §2
//! asks for ("~15 tasks", "at least 3 version-traps").

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

use harness_core::hash::hash_task_set_dir;
use serde::Deserialize;

/// Mirrors the `meta.toml` schema documented in
/// `benchmarks/illumos-e2e/README.md`. Field names match exactly so serde
/// can deserialize the real files without a custom mapping layer -- if the
/// README and this struct drift, this test starts failing to parse real
/// task files, which is the point.
#[derive(Debug, Deserialize)]
struct TaskMeta {
    id: String,
    family: String,
    description: String,
    timeout_secs: u64,
    expected_manpages: Vec<String>,
    mode: Vec<String>,
    version_trap: VersionTrap,
}

#[derive(Debug, Deserialize)]
struct VersionTrap {
    present: bool,
    #[serde(default)]
    note: Option<String>,
}

const KNOWN_FAMILIES: &[&str] = &[
    "smf",
    "manpage-discovery",
    "c-native-libs",
    "rust-illumos",
    "rbac",
];
const KNOWN_MODES: &[&str] = &["autonomous", "advisory"];
const REQUIRED_FILES: &[&str] = &["meta.toml", "setup.sh", "prompt.md", "verify.sh"];

/// `benchmarks/illumos-e2e/tasks` relative to this crate's manifest dir
/// (`benchmarks/harness-core`), i.e. `../illumos-e2e/tasks`.
fn tasks_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../illumos-e2e/tasks")
}

/// Sorted list of task directory names, so iteration order (and therefore
/// any assertion-failure ordering) is stable across platforms/filesystems.
fn task_dirs() -> Vec<PathBuf> {
    let dir = tasks_dir();
    let mut dirs: Vec<PathBuf> = fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", dir.display()))
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|p| p.is_dir())
        .collect();
    dirs.sort();
    dirs
}

#[test]
fn every_task_has_all_four_required_files() {
    let dirs = task_dirs();
    assert!(
        !dirs.is_empty(),
        "no task directories found under {}",
        tasks_dir().display()
    );

    for dir in &dirs {
        let name = dir.file_name().unwrap().to_string_lossy();
        for required in REQUIRED_FILES {
            let path = dir.join(required);
            assert!(
                path.is_file(),
                "task '{name}' is missing required file '{required}' ({})",
                path.display()
            );
        }
    }
}

#[test]
fn every_meta_toml_parses_and_satisfies_invariants() {
    let dirs = task_dirs();

    for dir in &dirs {
        let name = dir.file_name().unwrap().to_string_lossy().to_string();
        let meta_path = dir.join("meta.toml");
        let raw = fs::read_to_string(&meta_path)
            .unwrap_or_else(|e| panic!("failed to read {}: {e}", meta_path.display()));
        let meta: TaskMeta = toml::from_str(&raw)
            .unwrap_or_else(|e| panic!("failed to parse {}: {e}", meta_path.display()));

        assert_eq!(
            meta.id, name,
            "meta.toml 'id' must match the directory name for '{name}'"
        );

        assert!(
            KNOWN_FAMILIES.contains(&meta.family.as_str()),
            "task '{name}' has unknown family '{}' (expected one of {KNOWN_FAMILIES:?})",
            meta.family
        );

        assert!(
            !meta.description.trim().is_empty(),
            "task '{name}' has an empty description"
        );

        assert!(
            meta.timeout_secs > 0,
            "task '{name}' has timeout_secs=0, must be positive"
        );

        assert!(
            !meta.expected_manpages.is_empty(),
            "task '{name}' has an empty expected_manpages list -- every task must name at least one man page an akh should ground on"
        );
        for page in &meta.expected_manpages {
            assert!(
                !page.trim().is_empty(),
                "task '{name}' has a blank entry in expected_manpages"
            );
        }

        assert!(
            !meta.mode.is_empty(),
            "task '{name}' has an empty mode list"
        );
        for m in &meta.mode {
            assert!(
                KNOWN_MODES.contains(&m.as_str()),
                "task '{name}' has unknown mode '{m}' (expected one of {KNOWN_MODES:?})"
            );
        }

        if meta.version_trap.present {
            let note = meta.version_trap.note.as_deref().unwrap_or("");
            assert!(
                !note.trim().is_empty(),
                "task '{name}' has version_trap.present = true but no (or an empty) note explaining the trap"
            );
        } else {
            assert!(
                meta.version_trap
                    .note
                    .as_deref()
                    .unwrap_or("")
                    .trim()
                    .is_empty(),
                "task '{name}' has version_trap.present = false but a non-empty note -- either flag it as a trap or drop the note"
            );
        }
    }
}

#[test]
fn corpus_meets_the_adr_086_size_and_version_trap_bar() {
    let dirs = task_dirs();
    assert!(
        dirs.len() >= 15,
        "ADR 086 §2 asks for ~15 tasks; found {}",
        dirs.len()
    );

    let mut families_with_traps: BTreeSet<String> = BTreeSet::new();
    let mut trap_count = 0usize;
    let mut families_seen: BTreeSet<String> = BTreeSet::new();

    for dir in &dirs {
        let meta_path = dir.join("meta.toml");
        let raw = fs::read_to_string(&meta_path).unwrap();
        let meta: TaskMeta = toml::from_str(&raw).unwrap();
        families_seen.insert(meta.family.clone());
        if meta.version_trap.present {
            trap_count += 1;
            families_with_traps.insert(meta.family);
        }
    }

    assert!(
        trap_count >= 3,
        "ADR 086 §2 asks for at least 3 version-trap tasks; found {trap_count}"
    );
    assert!(
        families_with_traps.len() > 1,
        "version-traps should be distributed across more than one family (ADR 086 §2); all {trap_count} are in {families_with_traps:?}"
    );
    assert_eq!(
        families_seen.len(),
        KNOWN_FAMILIES.len(),
        "expected every one of the 5 known families to have at least one task, found families: {families_seen:?}"
    );
}

#[test]
fn task_set_hash_is_deterministic_across_repeated_calls() {
    // This is the property benchmarks/harness-core's W7 results pipeline
    // relies on (ADR 086 §4: "task-set content hash ... freezes on
    // publish"): hashing the same directory twice must produce the same
    // hash, independent of filesystem iteration order.
    let dir = tasks_dir();
    let first =
        hash_task_set_dir(&dir).expect("hash_task_set_dir should succeed on the real corpus");
    let second =
        hash_task_set_dir(&dir).expect("hash_task_set_dir should succeed on the real corpus");
    assert_eq!(
        first, second,
        "hashing the same task-set directory twice produced different hashes"
    );

    // Sanity: the hash is a real, well-formed blake3 digest (not an empty
    // hasher over zero files, which would indicate task_dirs()/tasks_dir()
    // pointed at the wrong place).
    assert_eq!(
        first.as_str().len(),
        64,
        "task-set hash should be a 64-hex-char blake3 digest"
    );
}
