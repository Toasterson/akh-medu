//! Inbox watcher: synchronous poll loop that detects new files and ingests them.
//!
//! Watches a directory for new files, ingests each one via the library pipeline,
//! and moves successfully processed files to a `done/` subdirectory.

use std::path::{Path, PathBuf};
use std::time::Duration;

use crate::engine::Engine;
use crate::library::catalog::LibraryCatalog;
use crate::library::error::{LibraryError, LibraryResult};
use crate::library::ingest::{IngestConfig, ingest_file};

/// Configuration for the inbox watcher.
pub struct InboxConfig {
    /// Directory to watch for new files.
    pub inbox_dir: PathBuf,
    /// Poll interval.
    pub poll_interval: Duration,
    /// Directory to move successfully ingested files to.
    pub done_dir: PathBuf,
    /// Library catalog directory.
    pub library_dir: PathBuf,
}

impl InboxConfig {
    /// Create config from an inbox directory, deriving done_dir as `inbox_dir/done/`.
    pub fn new(inbox_dir: PathBuf, library_dir: PathBuf) -> Self {
        let done_dir = inbox_dir.join("done");
        Self {
            inbox_dir,
            poll_interval: Duration::from_secs(5),
            done_dir,
            library_dir,
        }
    }
}

/// Run the inbox watcher loop. Blocks until interrupted.
///
/// On each poll:
/// 1. Scans the inbox directory for files.
/// 2. Skips directories and the `done/` subdirectory.
/// 3. Attempts to ingest each file.
/// 4. On success, moves the file to `done/`.
/// 5. On failure, prints an error and leaves the file in place.
pub fn watch_inbox(engine: &Engine, config: &InboxConfig) -> LibraryResult<()> {
    std::fs::create_dir_all(&config.inbox_dir).map_err(|e| LibraryError::CatalogIo {
        message: format!("create inbox dir {}: {e}", config.inbox_dir.display()),
    })?;
    std::fs::create_dir_all(&config.done_dir).map_err(|e| LibraryError::CatalogIo {
        message: format!("create done dir {}: {e}", config.done_dir.display()),
    })?;

    println!(
        "Watching inbox: {} (poll every {}s)",
        config.inbox_dir.display(),
        config.poll_interval.as_secs()
    );

    loop {
        let entries = match std::fs::read_dir(&config.inbox_dir) {
            Ok(e) => e,
            Err(e) => {
                eprintln!("Error reading inbox: {e}");
                std::thread::sleep(config.poll_interval);
                continue;
            }
        };

        for entry in entries.flatten() {
            let path = entry.path();

            // Skip directories (including done/).
            if path.is_dir() {
                continue;
            }

            // Skip hidden files.
            if path
                .file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.starts_with('.'))
            {
                continue;
            }

            println!("Detected: {}", path.display());
            match process_inbox_file(engine, &path, &config.library_dir, &config.done_dir) {
                Ok(()) => {
                    println!("  OK — ingested and moved to done/");
                }
                Err(e) => {
                    eprintln!("  FAILED — {e}");
                }
            }
        }

        std::thread::sleep(config.poll_interval);
    }
}

/// Process a single file from the inbox.
fn process_inbox_file(
    engine: &Engine,
    path: &Path,
    library_dir: &Path,
    done_dir: &Path,
) -> LibraryResult<()> {
    let mut catalog = LibraryCatalog::open(library_dir)?;

    let result = ingest_file(engine, &mut catalog, path, IngestConfig::default())?;

    println!(
        "  Ingested: \"{}\" ({} chunks, {} triples)",
        result.record.title, result.chunk_count, result.triple_count,
    );

    // Move file to done/.
    if let Some(filename) = path.file_name() {
        let dest = done_dir.join(filename);
        std::fs::rename(path, &dest).map_err(|e| LibraryError::CatalogIo {
            message: format!("move {} -> {}: {e}", path.display(), dest.display()),
        })?;
    }

    Ok(())
}
