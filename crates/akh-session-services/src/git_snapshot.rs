//! Read exact local objects through a private bare Git view. Workspace refs,
//! config, attributes, index, grafts and worktree files are not authority here.
use crate::{config::GitSnapshotProfile, process, types::*, workspace::Workspace};
use akh_session_types::repository::{CaptureGitSnapshot, GitSnapshot, is_commit_id as oid};
use sha2::{Digest, Sha256};
use std::{fs::File, io::Read, os::fd::AsRawFd, path::Path, time::Duration};
use tokio::process::Command;

const LIMIT: usize = 1024 * 1024;
const TIMEOUT: Duration = Duration::from_secs(30);

fn executable_digest(path: &Path) -> Result<String> {
    let mut file = File::open(path)
        .map_err(|_| Fault::new("git_unavailable", "configured Git could not be opened"))?;
    if !file
        .metadata()
        .is_ok_and(|m| m.is_file() && m.len() <= 256 * 1024 * 1024)
    {
        return Err(Fault::invalid(
            "configured Git must be a bounded regular executable",
        ));
    }
    let mut hash = Sha256::new();
    let mut buffer = [0; 65536];
    let mut read = 0;
    loop {
        let size = file.read(&mut buffer).map_err(|_| {
            Fault::new(
                "git_unavailable",
                "configured Git could not be fingerprinted",
            )
        })?;
        if size == 0 {
            break;
        }
        read += size;
        if read > 256 * 1024 * 1024 {
            return Err(Fault::invalid("configured Git exceeds fingerprint limit"));
        }
        hash.update(&buffer[..size]);
    }
    Ok(format!("{:x}", hash.finalize()))
}

pub(crate) fn capture(
    ws: &Workspace,
    profile: &GitSnapshotProfile,
    request: CaptureGitSnapshot,
) -> Result<GitSnapshot> {
    profile.validate()?;
    request.validate()?;
    if executable_digest(&profile.executable)? != profile.executable_sha256 {
        return Err(Fault::conflict(
            "configured Git executable fingerprint changed",
        ));
    }
    // Managed bootstrap produces a standalone .git directory. Linked worktrees,
    // symlinks and object borrowing need a separate explicit checkout contract.
    let objects = ws.directory(".git/objects")?;
    let source = Workspace::from_directory(
        objects
            .try_clone()
            .map_err(|_| Fault::invalid("object directory unavailable"))?,
        ws.path.join(".git/objects"),
    );
    if source
        .list("info")?
        .iter()
        .any(|name| matches!(name.as_str(), "alternates" | "http-alternates"))
    {
        return Err(Fault::new(
            "unsupported",
            "snapshot requires self-contained local Git objects",
        ));
    }
    let profile = profile.clone();
    // operate() is synchronous and may also be called from a Tokio test/runtime.
    // Own this bounded read on another thread instead of nesting a runtime.
    std::thread::spawn(move || {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|_| Fault::new("git_unavailable", "snapshot runtime unavailable"))?;
        runtime.block_on(capture_inner(objects, profile, request))
    })
    .join()
    .map_err(|_| Fault::new("git_unavailable", "snapshot worker failed"))?
}

struct View {
    directory: tempfile::TempDir,
    objects: File,
    profile: GitSnapshotProfile,
    deadline: tokio::time::Instant,
}

impl View {
    async fn run(&self, args: &[&str], limit: usize) -> Result<Vec<u8>> {
        let remaining = self
            .deadline
            .checked_duration_since(tokio::time::Instant::now())
            .ok_or_else(|| Fault::new("git_timeout", "Git snapshot exceeded 30 seconds"))?;
        let fd = self.objects.as_raw_fd();
        let mut command = Command::new(&self.profile.executable);
        command
            .current_dir(self.directory.path())
            .env_clear()
            .env("PATH", "/usr/bin:/bin")
            .env("LANG", "C")
            .env("LC_ALL", "C")
            .env("GIT_CONFIG_NOSYSTEM", "1")
            .env("GIT_CONFIG_GLOBAL", "/dev/null")
            .env("GIT_CONFIG_SYSTEM", "/dev/null")
            .env("GIT_ATTR_NOSYSTEM", "1")
            .env("GIT_NO_REPLACE_OBJECTS", "1")
            .env("GIT_NO_LAZY_FETCH", "1")
            .env("GIT_OPTIONAL_LOCKS", "0")
            .env("GIT_TERMINAL_PROMPT", "0")
            .env("GIT_ASKPASS", "/bin/false")
            .env("GIT_OBJECT_DIRECTORY", ".")
            .env("GIT_DIR", self.directory.path())
            .args([
                "--no-pager",
                "--no-replace-objects",
                "--no-lazy-fetch",
                "-c",
                "protocol.allow=never",
                "-c",
                "core.hooksPath=/dev/null",
                "-c",
                "core.fsmonitor=false",
                "-c",
                "core.commitGraph=false",
                "-c",
                "core.attributesFile=/dev/null",
            ])
            .args(args);
        // Anchor the child in the checked object directory. macOS /dev/fd/N
        // cannot be traversed as a directory path; fchdir works on both hosts.
        unsafe {
            command.pre_exec(move || {
                if libc::fchdir(fd) < 0 {
                    return Err(std::io::Error::last_os_error());
                }
                Ok(())
            });
        }
        // Let the subprocess helper kill and reap on its own deadline. An outer
        // timeout would drop that cleanup future before it can await the child.
        process::output(command, remaining, limit)
            .await
            .map_err(|error| {
                Fault::new(
                    &error.code,
                    format!(
                        "Git snapshot {} failed, timed out or exceeded its output limit",
                        args[0]
                    ),
                )
            })
    }

    async fn text(&self, args: &[&str], limit: usize) -> Result<String> {
        String::from_utf8(self.run(args, limit).await?).map_err(|_| {
            Fault::new(
                "unsupported",
                "snapshot output requires UTF-8; no lossy patch conversion",
            )
        })
    }
}

async fn capture_inner(
    objects: File,
    profile: GitSnapshotProfile,
    request: CaptureGitSnapshot,
) -> Result<GitSnapshot> {
    let directory = tempfile::Builder::new()
        .prefix("akh-git-snapshot-")
        .tempdir()
        .map_err(|_| Fault::new("git_unavailable", "private Git view could not be created"))?;
    let config = if request.head_revision.len() == 64 {
        "[core]\nrepositoryformatversion = 1\nbare = true\n[extensions]\nobjectformat = sha256\n"
    } else {
        "[core]\nrepositoryformatversion = 0\nbare = true\n"
    };
    std::fs::write(directory.path().join("config"), config)
        .and_then(|_| {
            std::fs::write(
                directory.path().join("HEAD"),
                "ref: refs/heads/akh-snapshot\n",
            )
        })
        .and_then(|_| std::fs::create_dir(directory.path().join("refs")))
        .map_err(|_| Fault::new("git_unavailable", "private Git view initialization failed"))?;
    let view = View {
        directory,
        objects,
        profile,
        deadline: tokio::time::Instant::now() + TIMEOUT,
    };
    let base = request.base_revision.as_str();
    let head = request.head_revision.as_str();
    let version = view.text(&["--version"], 256).await?;
    for revision in [base, head] {
        if view.text(&["cat-file", "-t", revision], 64).await?.trim() != "commit" {
            return Err(Fault::invalid(
                "snapshot revisions must name commits, not tags or trees",
            ));
        }
    }
    view.run(&["merge-base", "--is-ancestor", base, head], 64)
        .await?;
    // The private config cannot suppress fsck errors. Corrupt local objects do
    // not become a successful receipt just because rev-parse printed their name.
    view.run(
        &[
            "fsck",
            "--strict",
            "--no-reflogs",
            "--no-dangling",
            base,
            head,
        ],
        LIMIT,
    )
    .await?;
    let base_tree = view
        .text(&["rev-parse", "--verify", &format!("{base}^{{tree}}")], 128)
        .await?
        .trim()
        .to_owned();
    let head_tree = view
        .text(&["rev-parse", "--verify", &format!("{head}^{{tree}}")], 128)
        .await?
        .trim()
        .to_owned();
    if !oid(&base_tree)
        || !oid(&head_tree)
        || base_tree.len() != base.len()
        || head_tree.len() != head.len()
    {
        return Err(Fault::conflict("Git returned invalid tree identities"));
    }
    let common = [
        "diff",
        "--no-ext-diff",
        "--no-textconv",
        "--no-renames",
        "--no-color",
        "--no-indent-heuristic",
        "--diff-algorithm=myers",
        "--src-prefix=a/",
        "--dst-prefix=b/",
        "--ignore-submodules=none",
        "--submodule=short",
        "--unified=3",
        "--inter-hunk-context=0",
    ];
    let mut args = common.to_vec();
    args.extend(["--binary", "--full-index", base, head, "--"]);
    let patch = view.text(&args, LIMIT).await?;
    let mut args = common.to_vec();
    args.extend(["--no-patch", "--numstat", "-z", base, head, "--"]);
    let (files_changed, lines_added, lines_removed, binary_files) =
        statistics(&view.run(&args, LIMIT).await?)?;
    Ok(GitSnapshot {
        schema_version: SCHEMA_VERSION,
        base_tree,
        head_tree,
        patch_sha256: crate::digest(patch.as_bytes()),
        patch,
        request,
        files_changed,
        lines_added,
        lines_removed,
        binary_files,
        git_version: version.trim().to_owned(),
        executable_sha256: view.profile.executable_sha256.clone(),
    })
}

fn statistics(bytes: &[u8]) -> Result<(i32, i32, i32, i32)> {
    let mut stats = (0i32, 0i32, 0i32, 0i32);
    if !bytes.is_empty() && !bytes.ends_with(&[0]) {
        return Err(Fault::conflict("incomplete Git statistics"));
    }
    if bytes.is_empty() {
        return Ok(stats);
    }
    for record in bytes[..bytes.len() - 1].split(|b| *b == 0) {
        let mut fields = record.splitn(3, |b| *b == b'\t');
        let added = fields.next().unwrap_or_default();
        let removed = fields.next().unwrap_or_default();
        if fields.next().is_none_or(|path| path.is_empty()) {
            return Err(Fault::conflict("invalid Git statistics"));
        }
        stats.0 = stats
            .0
            .checked_add(1)
            .ok_or_else(|| Fault::invalid("too many changed files"))?;
        if added == b"-" && removed == b"-" {
            stats.3 += 1;
        } else {
            for (value, total) in [(added, &mut stats.1), (removed, &mut stats.2)] {
                let n = std::str::from_utf8(value)
                    .ok()
                    .filter(|s| !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit()))
                    .and_then(|s| s.parse::<i32>().ok())
                    .ok_or_else(|| Fault::conflict("invalid Git line count"))?;
                *total = total
                    .checked_add(n)
                    .ok_or_else(|| Fault::invalid("Git line count overflow"))?;
            }
        }
    }
    Ok(stats)
}
