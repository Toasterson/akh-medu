//! Native Git and durable executor receipts in an owned temporary workspace.
#![cfg(feature = "executor")]
use akh_session_services::{config::Config, executor::Executor, store::Store, types::*};
use akh_session_types::repository::GitSnapshot;
use serde_json::json;
use std::{path::Path, process::Command, sync::Arc};

struct OwnedGit(std::process::Child);
impl Drop for OwnedGit {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

fn publication_peer(fixture: &Fixture) -> (OwnedGit, String, std::path::PathBuf) {
    let remote = fixture.dir.path().join("remote.git");
    std::fs::create_dir(&remote).unwrap();
    let format = if fixture.base.len() == 64 {
        "sha256"
    } else {
        "sha1"
    };
    git(
        &remote,
        &[
            "init",
            "--bare",
            "--template=",
            &format!("--object-format={format}"),
        ],
    );
    git(
        &fixture.root(),
        &[
            "push",
            remote.to_str().unwrap(),
            &format!("{}:refs/heads/main", fixture.base),
        ],
    );
    let socket = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let address = socket.local_addr().unwrap();
    drop(socket);
    let mut daemon = OwnedGit(
        Command::new("/usr/bin/git")
            .env_clear()
            .env("PATH", "/usr/bin:/bin")
            .env("GIT_CONFIG_GLOBAL", "/dev/null")
            .env("GIT_CONFIG_SYSTEM", "/dev/null")
            .args([
                "daemon",
                "--export-all",
                "--enable=receive-pack",
                "--reuseaddr",
                "--listen=127.0.0.1",
                &format!("--port={}", address.port()),
                &format!("--base-path={}", fixture.dir.path().display()),
            ])
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()
            .unwrap(),
    );
    let mut ready = false;
    for _ in 0..100 {
        assert!(daemon.0.try_wait().unwrap().is_none());
        if std::net::TcpStream::connect(address).is_ok() {
            ready = true;
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    assert!(ready);
    (daemon, format!("git://{address}/remote.git"), remote)
}

fn configure_publication(fixture: &mut Fixture, repository: &str) {
    fixture.config.git_publication=vec![serde_json::from_value(json!({"repository":repository,"branch_prefixes":["refs/heads/anima/"],"config_file":null})).unwrap()];
    fixture.config.validate().unwrap();
    fixture.executor = Executor::new(
        Arc::new(fixture.config.clone()),
        fixture.executor.store.clone(),
    )
    .unwrap();
}

fn publication_request(
    fixture: &Fixture,
    id: &str,
    repository: &str,
    branch: &str,
) -> WorkspaceOperation {
    let mut request = fixture.request(id);
    request.operation = "git_publish".into();
    request.turn_id = "repository-publication".into();
    request.arguments = json!({"snapshot_operation_id":"publication-source","repository":repository,"branch_ref":branch});
    request
}

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

#[tokio::test]
async fn native_publication_is_controller_owned_scoped_and_recovers_its_exact_receipt() {
    use axum::{
        body::{Body, to_bytes},
        http::{Request, StatusCode},
    };
    use tower::ServiceExt;
    for format in ["sha1", "sha256"] {
        let mut fixture = Fixture::new(format);
        let (_daemon, repository, remote) = publication_peer(&fixture);
        fixture.capture("publication-source");
        configure_publication(&mut fixture, &repository);
        let token = fixture.dir.path().join("publisher-token");
        std::fs::write(&token, "owned-publisher-capability-123456").unwrap();
        let read_token = fixture.dir.path().join("reader-token");
        std::fs::write(&read_token, "owned-reader-capability-123456").unwrap();
        fixture.config.principals[0].token_file = token;
        fixture.config.principals[0].roles = vec!["session".into(), "repository_publish".into()];
        fixture.config.principals.push(
            serde_json::from_value(
                json!({"id":"reader","roles":["session"],"token_file":read_token}),
            )
            .unwrap(),
        );
        fixture.executor = Executor::new(
            Arc::new(fixture.config.clone()),
            fixture.executor.store.clone(),
        )
        .unwrap();
        let router = akh_session_services::server::executor_router(fixture.executor.clone());
        let request = publication_request(
            &fixture,
            "publication",
            &repository,
            "refs/heads/anima/work",
        );
        let call = |path: &str, capability: &str| {
            Request::builder()
                .method("POST")
                .uri(path)
                .header("authorization", format!("Bearer {capability}"))
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&request).unwrap()))
                .unwrap()
        };
        assert_eq!(
            router
                .clone()
                .oneshot(call("/v1/operations", "owned-publisher-capability-123456"))
                .await
                .unwrap()
                .status(),
            StatusCode::FORBIDDEN
        );
        assert_eq!(
            router
                .clone()
                .oneshot(call(
                    "/v1/repository-publications",
                    "owned-reader-capability-123456"
                ))
                .await
                .unwrap()
                .status(),
            StatusCode::FORBIDDEN
        );
        let response = router
            .oneshot(call(
                "/v1/repository-publications",
                "owned-publisher-capability-123456",
            ))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let receipt: OperationReceipt =
            serde_json::from_slice(&to_bytes(response.into_body(), 1024 * 1024).await.unwrap())
                .unwrap();
        assert!(
            receipt.result.as_ref().unwrap().success,
            "{:?}",
            receipt.result
        );
        let published: publication::GitPublication =
            serde_json::from_value(receipt.result.as_ref().unwrap().data.clone().unwrap()).unwrap();
        assert!(published.push_acknowledged);
        assert!(!published.already_present);
        assert_eq!(published.previous_revision, None);
        assert_eq!(published.observed_revision, fixture.head);
        assert_eq!(
            git(&remote, &["rev-parse", "refs/heads/anima/work"]),
            fixture.head
        );
        fixture.executor = Executor::new(
            Arc::new(fixture.config.clone()),
            fixture.executor.store.clone(),
        )
        .unwrap();
        assert_eq!(fixture.executor.operate(request).unwrap(), receipt);
        let another = fixture
            .executor
            .operate(publication_request(
                &fixture,
                "already-published",
                &repository,
                "refs/heads/anima/work",
            ))
            .unwrap();
        let observed: publication::GitPublication =
            serde_json::from_value(another.result.unwrap().data.unwrap()).unwrap();
        assert!(observed.already_present);
        assert!(!observed.push_acknowledged);
        for (id, branch) in [
            ("main", "refs/heads/main"),
            ("tag", "refs/tags/v1"),
            ("traversal", "refs/heads/anima/../main"),
        ] {
            assert!(
                !fixture
                    .executor
                    .operate(publication_request(&fixture, id, &repository, branch))
                    .unwrap()
                    .result
                    .unwrap()
                    .success
            );
        }
        let mut wrong = publication_request(
            &fixture,
            "wrong-snapshot",
            &repository,
            "refs/heads/anima/wrong",
        );
        wrong.arguments["snapshot_operation_id"] = json!("publication");
        assert!(
            !fixture
                .executor
                .operate(wrong)
                .unwrap()
                .result
                .unwrap()
                .success
        );
        assert_eq!(
            git(&remote, &["rev-parse", "refs/heads/main"]),
            fixture.base
        );
        assert!(git(&fixture.root(), &["status", "--porcelain"]).is_empty());
    }
}

#[test]
fn native_publication_preserves_a_racing_branch_and_fences_uncertain_writes_and_release() {
    use std::os::unix::fs::PermissionsExt;
    let mut fixture = Fixture::new("sha1");
    let (_daemon, repository, remote) = publication_peer(&fixture);
    fixture.capture("publication-source");
    configure_publication(&mut fixture, &repository);
    git(&fixture.root(), &["checkout", "--detach", &fixture.base]);
    std::fs::write(fixture.root().join("other.txt"), "another writer\n").unwrap();
    git(&fixture.root(), &["add", "."]);
    git(&fixture.root(), &["commit", "-m", "other writer"]);
    let other = git(&fixture.root(), &["rev-parse", "HEAD"]);
    git(
        &fixture.root(),
        &[
            "push",
            remote.to_str().unwrap(),
            &format!("{other}:refs/heads/other"),
        ],
    );
    git(&fixture.root(), &["checkout", "--detach", &fixture.head]);
    git(&remote, &["update-ref", "refs/heads/anima/ahead", &other]);
    let before = fixture
        .executor
        .operate(publication_request(
            &fixture,
            "ahead",
            &repository,
            "refs/heads/anima/ahead",
        ))
        .unwrap();
    assert_eq!(before.status, TurnStatus::Failed);
    assert!(!before.result.unwrap().success);
    assert_eq!(
        git(&remote, &["rev-parse", "refs/heads/anima/ahead"]),
        other
    );
    git(
        &remote,
        &["update-ref", "refs/heads/anima/race", &fixture.base],
    );
    std::fs::create_dir_all(remote.join("hooks")).unwrap();
    let hook = remote.join("hooks/pre-receive");
    std::fs::write(&hook,format!("#!/bin/sh\nunset GIT_QUARANTINE_PATH GIT_OBJECT_DIRECTORY GIT_ALTERNATE_OBJECT_DIRECTORIES\n/usr/bin/git update-ref refs/heads/anima/race {other} {}\n",fixture.base)).unwrap();
    std::fs::set_permissions(&hook, std::fs::Permissions::from_mode(0o700)).unwrap();
    let request = publication_request(&fixture, "raced", &repository, "refs/heads/anima/race");
    let receipt = fixture.executor.operate(request.clone()).unwrap();
    assert_eq!(receipt.status, TurnStatus::Unknown, "{:?}", receipt.result);
    assert_eq!(git(&remote, &["rev-parse", "refs/heads/anima/race"]), other);
    assert!(fixture.executor.release("workspace", "session", 1).is_err());
    assert!(
        fixture
            .executor
            .operate(fixture.request("later-write"))
            .is_ok(),
        "read snapshots remain available"
    );
    let mut write = fixture.request("write-after-unknown");
    write.operation = "write".into();
    write.arguments = json!({"path":"new.txt","content":"unconfirmed","create":true});
    assert!(fixture.executor.operate(write).is_err());
    fixture.executor = Executor::new(
        Arc::new(fixture.config.clone()),
        fixture.executor.store.clone(),
    )
    .unwrap();
    assert_eq!(fixture.executor.operate(request).unwrap(), receipt);
}
