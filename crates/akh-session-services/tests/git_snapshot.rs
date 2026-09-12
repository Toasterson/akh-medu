//! Native Git and durable executor receipts in an owned temporary workspace.
#![cfg(feature = "executor")]
use akh_session_services::{config::Config, executor::Executor, store::Store, types::*};
use akh_session_types::repository::GitSnapshot;
use serde_json::json;
use std::{path::Path, process::Command, sync::Arc};

fn git(root: &Path, args: &[&str]) -> String {
    let output = Command::new("/usr/bin/git")
        .current_dir(root)
        .env_clear()
        .env("PATH", "/usr/bin:/bin")
        .env("GIT_CONFIG_GLOBAL", "/dev/null")
        .env("GIT_CONFIG_SYSTEM", "/dev/null")
        .args([
            "-c",
            "core.hooksPath=/dev/null",
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
        ])
        .args(args)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout).unwrap().trim().into()
}

struct Fixture {
    dir: tempfile::TempDir,
    executor: Executor,
    config: Config,
    base: String,
    head: String,
}
impl Fixture {
    fn new(format: &str) -> Self {
        let dir = tempfile::tempdir().unwrap();
        let config: Config = serde_json::from_value(json!({
            "service_id":"executor", "listen":"127.0.0.1:0", "data_dir":dir.path().join("state"),
            "workspace_root":dir.path().join("workspaces"),
            "git_snapshot":{"executable":"/usr/bin/git", "executable_sha256":akh_session_services::digest(&std::fs::read("/usr/bin/git").unwrap())},
            "principals":[{"id":"sessions","roles":["session"],"token_file":"/unused"}]
        })).unwrap();
        config.validate().unwrap();
        let executor = Executor::new(
            Arc::new(config.clone()),
            Store::open(&dir.path().join("state.redb")).unwrap(),
        )
        .unwrap();
        executor
            .attach(&AttachWorkspace {
                session_id: "session".into(),
                binding: WorkspaceBinding {
                    id: "workspace".into(),
                    sandbox_id: "sandbox".into(),
                    executor: "executor".into(),
                    generation: 1,
                    site: "local".into(),
                    root: dir.path().join("workspaces/repo").to_string_lossy().into(),
                    platform: "unknown".into(),
                    confinement: "unknown".into(),
                },
            })
            .unwrap();
        let root = dir.path().join("workspaces/repo");
        git(
            &root,
            &[
                "init",
                "--quiet",
                "--template=",
                &format!("--object-format={format}"),
            ],
        );
        std::fs::write(root.join("accepted.txt"), "accepted work\n").unwrap();
        git(&root, &["add", "."]);
        git(&root, &["commit", "--quiet", "-m", "accepted"]);
        let base = git(&root, &["rev-parse", "HEAD"]);
        std::fs::write(root.join("accepted.txt"), "accepted work\nimplementation\n").unwrap();
        std::fs::write(root.join("binary.bin"), b"\0binary\xff\n").unwrap();
        std::fs::write(root.join("odd\tname\n.txt"), "odd filename\n").unwrap();
        git(&root, &["add", "."]);
        git(&root, &["commit", "--quiet", "-m", "implementation"]);
        let head = git(&root, &["rev-parse", "HEAD"]);
        Self {
            dir,
            executor,
            config,
            base,
            head,
        }
    }
    fn root(&self) -> std::path::PathBuf {
        self.dir.path().join("workspaces/repo")
    }
    fn request(&self, id: &str) -> WorkspaceOperation {
        WorkspaceOperation {
            id: id.into(),
            session_id: "session".into(),
            turn_id: "turn".into(),
            workspace_id: "workspace".into(),
            generation: 1,
            operation: "git_snapshot".into(),
            arguments: json!({"base_revision":self.base,"head_revision":self.head}),
        }
    }
    fn capture(&self, id: &str) -> (OperationReceipt, GitSnapshot) {
        let receipt = self.executor.operate(self.request(id)).unwrap();
        let result = receipt.result.as_ref().unwrap();
        assert!(result.success, "{}", result.text);
        assert_eq!(result.receipt.as_deref(), Some(id));
        let snapshot = serde_json::from_value(result.data.clone().unwrap()).unwrap();
        (receipt, snapshot)
    }
}

#[test]
fn native_commit_snapshot_preserves_binary_patch_and_replays_after_workspace_change_and_restart() {
    for format in ["sha1", "sha256"] {
        let mut fixture = Fixture::new(format);
        let (receipt, snapshot) = fixture.capture("capture");
        assert_eq!(snapshot.request.base_revision, fixture.base);
        assert_eq!(snapshot.request.head_revision, fixture.head);
        assert_eq!(
            (
                snapshot.files_changed,
                snapshot.lines_added,
                snapshot.lines_removed,
                snapshot.binary_files
            ),
            (3, 2, 0, 1)
        );
        assert!(snapshot.patch.contains("GIT binary patch"));
        assert_eq!(
            snapshot.patch_sha256,
            akh_session_services::digest(snapshot.patch.as_bytes())
        );
        assert_eq!(
            snapshot.head_tree,
            git(&fixture.root(), &["rev-parse", "HEAD^{tree}"])
        );
        // Apply the exact receipt to an independent checkout of its base and
        // compare the resulting Git tree, including binary data and odd paths.
        let validation = fixture.dir.path().join("validation");
        git(
            fixture.dir.path(),
            &[
                "clone",
                "--quiet",
                "--no-hardlinks",
                fixture.root().to_str().unwrap(),
                "validation",
            ],
        );
        git(
            &validation,
            &["checkout", "--quiet", "--detach", &fixture.base],
        );
        let patch = fixture.dir.path().join("result.patch");
        std::fs::write(&patch, &snapshot.patch).unwrap();
        git(&validation, &["apply", "--index", patch.to_str().unwrap()]);
        assert_eq!(git(&validation, &["write-tree"]), snapshot.head_tree);
        std::fs::write(
            fixture.root().join("accepted.txt"),
            "later uncommitted editor buffer\n",
        )
        .unwrap();
        assert_eq!(fixture.capture("fresh-same-commit").1, snapshot);
        git(&fixture.root(), &["add", "."]);
        git(&fixture.root(), &["commit", "--quiet", "-m", "later"]);
        assert_eq!(
            fixture
                .executor
                .operate(fixture.request("capture"))
                .unwrap(),
            receipt
        );
        let mut changed = fixture.request("capture");
        changed.arguments["head_revision"] = json!(git(&fixture.root(), &["rev-parse", "HEAD"]));
        assert!(fixture.executor.operate(changed).is_err());
        let store = fixture.executor.store.clone();
        fixture.executor = Executor::new(Arc::new(fixture.config.clone()), store).unwrap();
        assert_eq!(
            fixture
                .executor
                .operate(fixture.request("capture"))
                .unwrap(),
            receipt
        );
        assert_eq!(fixture.capture("after-restart").1, snapshot);
    }
}

#[test]
fn native_snapshot_ignores_workspace_config_attributes_replace_refs_and_grafts() {
    let fixture = Fixture::new("sha1");
    let expected = fixture.capture("before").1;
    // If the original config is loaded, the included path makes Git fail. If
    // any diff/filter helper runs, it writes a detectable marker.
    let marker = fixture.dir.path().join("must-not-run");
    let hook = format!("touch {}", marker.display());
    git(&fixture.root(), &["config", "diff.external", &hook]);
    git(&fixture.root(), &["config", "diff.hostile.textconv", &hook]);
    std::fs::write(fixture.root().join(".gitattributes"), "* diff=hostile\n").unwrap();
    std::fs::create_dir_all(fixture.root().join(".git/info")).unwrap();
    std::fs::write(fixture.root().join(".git/info/attributes"), "* -diff\n").unwrap();
    git(&fixture.root(), &["replace", &fixture.head, &fixture.base]);
    std::fs::write(
        fixture.root().join(".git/info/grafts"),
        format!("{}\n", fixture.head),
    )
    .unwrap();
    git(
        &fixture.root(),
        &["config", "include.path", "/dev/null/not-a-config"],
    );
    assert_eq!(fixture.capture("after").1, expected);
    assert!(!marker.exists());
}

#[test]
fn snapshot_refuses_invalid_identity_objects_installation_and_workspace() {
    let fixture = Fixture::new("sha1");
    for (id, value) in [
        ("zero", "0".repeat(40)),
        ("short", "a".repeat(12)),
        ("missing", "a".repeat(40)),
        ("option", "--help".into()),
        ("tree", git(&fixture.root(), &["rev-parse", "HEAD^{tree}"])),
    ] {
        let mut request = fixture.request(id);
        request.arguments["head_revision"] = json!(value);
        assert!(
            !fixture
                .executor
                .operate(request)
                .unwrap()
                .result
                .unwrap()
                .success,
            "{id}"
        );
    }
    let mut backwards = fixture.request("backwards");
    backwards.arguments = json!({"base_revision":fixture.head,"head_revision":fixture.base});
    assert!(
        !fixture
            .executor
            .operate(backwards)
            .unwrap()
            .result
            .unwrap()
            .success
    );
    let mut foreign = fixture.request("foreign");
    foreign.session_id = "other".into();
    assert!(fixture.executor.operate(foreign).is_err());
    let mut stale = fixture.request("stale");
    stale.generation = 2;
    assert!(fixture.executor.operate(stale).is_err());
    let mut injected = fixture.request("injected");
    injected.arguments["executable"] = json!("/bin/sh");
    assert!(
        !fixture
            .executor
            .operate(injected)
            .unwrap()
            .result
            .unwrap()
            .success
    );
    let mut config = fixture.config.clone();
    config.git_snapshot.as_mut().unwrap().executable_sha256 = "a".repeat(64);
    let wrong = Executor::new(Arc::new(config), fixture.executor.store.clone()).unwrap();
    assert!(
        !wrong
            .operate(fixture.request("installation-drift"))
            .unwrap()
            .result
            .unwrap()
            .success
    );
    std::fs::write(
        fixture.root().join(".git/objects/info/alternates"),
        "/untrusted/objects\n",
    )
    .unwrap();
    assert!(
        !fixture
            .executor
            .operate(fixture.request("alternates"))
            .unwrap()
            .result
            .unwrap()
            .success
    );
}

#[test]
fn native_snapshot_reads_packed_objects_but_refuses_corruption_and_oversized_patches() {
    let fixture = Fixture::new("sha1");
    let expected = fixture.capture("loose").1;
    git(&fixture.root(), &["repack", "-ad"]);
    assert_eq!(fixture.capture("packed").1, expected);
    // A text patch over the submission limit must fail without a truncated
    // successful receipt. All child processes are bounded and reaped.
    std::fs::write(
        fixture.root().join("large.txt"),
        "large line\n".repeat(110000),
    )
    .unwrap();
    git(&fixture.root(), &["add", "."]);
    git(&fixture.root(), &["commit", "--quiet", "-m", "large"]);
    let mut large = fixture.request("large");
    large.arguments["head_revision"] = json!(git(&fixture.root(), &["rev-parse", "HEAD"]));
    assert!(
        !fixture
            .executor
            .operate(large)
            .unwrap()
            .result
            .unwrap()
            .success
    );
    let blob = git(&fixture.root(), &["rev-parse", "HEAD:large.txt"]);
    let path = fixture
        .root()
        .join(".git/objects")
        .join(&blob[..2])
        .join(&blob[2..]);
    std::fs::remove_file(&path).unwrap();
    std::fs::write(path, "corrupt object").unwrap();
    assert!(
        !fixture
            .executor
            .operate(fixture.request("corruption"))
            .unwrap()
            .result
            .unwrap()
            .success
    );
}

#[test]
fn native_submitted_patch_must_reproduce_exact_tree_without_changing_workspace_or_objects() {
    for format in ["sha1", "sha256"] {
        let fixture = Fixture::new(format);
        let expected = fixture.capture("reference").1;
        let index = std::fs::read(fixture.root().join(".git/index")).unwrap();
        let object_counts = git(&fixture.root(), &["count-objects", "-v"]);
        let alternate_diff = format!(
            "{}\n",
            git(
                &fixture.root(),
                &[
                    "diff",
                    "--binary",
                    "--unified=0",
                    &fixture.base,
                    &fixture.head
                ]
            )
        );
        let email = format!(
            "{}\n",
            git(
                &fixture.root(),
                &["format-patch", "--stdout", "--binary", "-1", &fixture.head]
            )
        );
        assert_ne!(email, expected.patch);
        for (id, patch) in [
            ("canonical", expected.patch.clone()),
            ("no-context", alternate_diff),
            ("email", email),
        ] {
            let mut request = fixture.request(id);
            request.arguments["expected_patch"] = json!(patch);
            let receipt = fixture.executor.operate(request.clone()).unwrap();
            let result = receipt.result.as_ref().unwrap();
            assert!(result.success, "{id}: {}", result.text);
            let snapshot: GitSnapshot =
                serde_json::from_value(result.data.clone().unwrap()).unwrap();
            assert_eq!(
                snapshot.expected_patch_sha256,
                Some(akh_session_services::digest(patch.as_bytes()))
            );
            assert!(snapshot.request.expected_patch.is_none());
            assert_eq!(snapshot.patch, expected.patch);
            assert_eq!(snapshot.head_tree, expected.head_tree);
            assert_eq!(fixture.executor.operate(request).unwrap(), receipt);
        }
        let mut different = fixture.request("different-tree");
        different.arguments["expected_patch"] = json!(
            expected
                .patch
                .replace("+implementation", "+different implementation")
        );
        let receipt = fixture.executor.operate(different).unwrap();
        assert!(!receipt.result.as_ref().unwrap().success);
        assert!(
            receipt
                .result
                .unwrap()
                .text
                .contains("does not produce the selected commit tree")
        );
        assert_eq!(
            std::fs::read(fixture.root().join(".git/index")).unwrap(),
            index
        );
        assert_eq!(
            git(&fixture.root(), &["count-objects", "-v"]),
            object_counts
        );
        assert_eq!(git(&fixture.root(), &["rev-parse", "HEAD"]), fixture.head);
        assert!(git(&fixture.root(), &["status", "--porcelain"]).is_empty());
    }
}
