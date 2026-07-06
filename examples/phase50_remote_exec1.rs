//! Phase 50 W2b: validate the Phase 48c `RemoteBackend` against a LIVE
//! OpenIndiana `exec1` guest (not the `src/backend/remote.rs` loopback
//! fixture). This is the proof that the whole Phase-50 feature-test
//! substrate — `vmctl` illumos guests + the russh `Backend` seam — actually
//! works end to end, not just against an in-process test server.
//!
//! ADR 082 (privilege-separated executor backend) + ADR 086 (feature-test VM
//! harness) + docs/ai/plans/2026-07-05-phase50-feature-test-benchmarks.md.
//!
//! ## Why an example, not a `#[test]`
//!
//! This drives a real network target (a specific, already-provisioned
//! illumos VM) that is not part of the normal test fixture set — there is no
//! way to stand exec1 up as a side effect of `cargo test`, and the guardrail
//! for this task is explicitly "do not reboot/rebake exec1". An example that
//! only runs on an explicit `cargo run` (and skips cleanly with no env var
//! set) keeps `cargo test` hermetic while still giving a real, re-runnable
//! validation path.
//!
//! ## Run it
//!
//! ```sh
//! export CARGO_TARGET_DIR=/home/toasty/ws/agi/akh-medu/target
//! AKH_EXEC1_SSH=akh-exec@127.0.0.1:10042 \
//! AKH_EXEC1_KEY=/path/to/akh-exec_ed25519 \
//! cargo run --quiet --features remote-executor --example phase50_remote_exec1
//! ```
//!
//! `AKH_EXEC1_SSH` is `user@host:port`. Without it set, the example prints a
//! SKIP message and exits 0 (safe to leave wired into any script). Without
//! `AKH_EXEC1_KEY`, it errors out with a clear message (key is required —
//! there's no sane default to fall back to).

#[cfg(feature = "remote-executor")]
fn main() {
    use akh_medu::backend::remote::{HostKeyPolicy, RemoteBackend, RemoteBackendConfig};
    use akh_medu::backend::{Backend, Confinement, ExecRequest};
    use akh_medu::fs::Filesystem;
    use std::path::PathBuf;
    use std::time::Duration;

    let Some(target) = std::env::var("AKH_EXEC1_SSH").ok() else {
        println!(
            "SKIP: AKH_EXEC1_SSH not set (expected user@host:port, e.g. \
             akh-exec@127.0.0.1:10042) — not pointed at a live guest, nothing to do."
        );
        return;
    };
    let Some((user, host_port)) = target.split_once('@') else {
        eprintln!("AKH_EXEC1_SSH must be user@host:port, got: {target}");
        std::process::exit(2);
    };
    let Some((host, port_str)) = host_port.rsplit_once(':') else {
        eprintln!("AKH_EXEC1_SSH must be user@host:port, got: {target}");
        std::process::exit(2);
    };
    let port: u16 = port_str.parse().unwrap_or_else(|_| {
        eprintln!("bad port in AKH_EXEC1_SSH: {port_str}");
        std::process::exit(2);
    });
    let key_path = std::env::var("AKH_EXEC1_KEY").unwrap_or_else(|_| {
        eprintln!("AKH_EXEC1_KEY not set — path to the staged akh-exec private key is required");
        std::process::exit(2);
    });

    println!("=== Phase 50 W2b: RemoteBackend -> live illumos exec1 ({user}@{host}:{port}) ===\n");

    // Lab/loopback posture (ADR 086 §1: plain provisioned keys, no cert/OIDC
    // in this build). exec1 is a disposable VM reachable only via a
    // localhost port-forward for this session, so `InsecureAcceptAny` is the
    // honest choice here rather than pretending a known_hosts pin means
    // anything for a guest that gets rebaked routinely; RemoteBackend logs a
    // loud warning on every connect under this policy by design.
    let config = RemoteBackendConfig::new(host, user, PathBuf::from(&key_path))
        .with_port(port)
        .with_host_key_policy(HostKeyPolicy::InsecureAcceptAny)
        .with_connect_timeout(Duration::from_secs(15))
        .with_roots(vec![PathBuf::from("/etc"), PathBuf::from("/tmp")], vec![PathBuf::from("/tmp")]);

    let backend = match RemoteBackend::connect(config) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("FAIL: could not connect RemoteBackend to exec1: {e}");
            std::process::exit(1);
        }
    };
    println!("PASS  connect: SSH session established to {user}@{host}:{port}\n");

    let mut results: Vec<(&'static str, bool, String)> = Vec::new();
    let mut record = |name: &'static str, ok: bool, detail: String| {
        let tag = if ok { "PASS" } else { "FAIL" };
        println!("{tag}  {name}");
        for line in detail.lines().take(6) {
            println!("      | {line}");
        }
        results.push((name, ok, detail));
    };

    // -----------------------------------------------------------------
    // exec: real illumos sysadmin commands over the SSH exec channel
    // -----------------------------------------------------------------

    match backend.exec(ExecRequest::shell("svcs -H svc:/network/ssh", Duration::from_secs(10))) {
        Ok(out) => {
            let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
            let ok = out.success && stdout.contains("online");
            record("exec: svcs -H svc:/network/ssh (expect online)", ok, stdout);
        }
        Err(e) => record("exec: svcs -H svc:/network/ssh", false, format!("error: {e}")),
    }

    match backend.exec(ExecRequest::shell("uname -s", Duration::from_secs(10))) {
        Ok(out) => {
            let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
            let ok = out.success && stdout.trim() == "SunOS";
            record("exec: uname -s (expect SunOS)", ok, stdout);
        }
        Err(e) => record("exec: uname -s", false, format!("error: {e}")),
    }

    match backend.exec(ExecRequest::shell(
        "kstat -p cpu_info:0:cpu_info0:brand || psrinfo -v",
        Duration::from_secs(10),
    )) {
        Ok(out) => {
            let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
            let ok = out.success && !stdout.trim().is_empty();
            record("exec: kstat cpu_info (or psrinfo -v) non-empty", ok, stdout);
        }
        Err(e) => record("exec: kstat/psrinfo", false, format!("error: {e}")),
    }

    match backend.exec(ExecRequest::shell("man -k svcs", Duration::from_secs(10))) {
        Ok(out) => {
            let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
            let ok = out.success && stdout.to_lowercase().contains("svcs");
            record("exec: man -k svcs (manpage-grounding path)", ok, stdout);
        }
        Err(e) => record("exec: man -k svcs", false, format!("error: {e}")),
    }

    match backend.exec(ExecRequest::shell("man svcs 2>&1 | head -3", Duration::from_secs(10))) {
        Ok(out) => {
            let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
            let ok = out.success && stdout.to_uppercase().contains("SVCS");
            record("exec: man svcs (page body)", ok, stdout);
        }
        Err(e) => record("exec: man svcs", false, format!("error: {e}")),
    }

    // -----------------------------------------------------------------
    // exec error path: a failing command must return a real ExitOutcome,
    // never a panic / Err from the transport itself.
    // -----------------------------------------------------------------

    match backend.exec(ExecRequest::shell("exit 42", Duration::from_secs(10))) {
        Ok(out) => {
            let ok = !out.success && out.exit_code == 42;
            record(
                "exec error path: `exit 42` -> ExecOutcome{success:false, exit_code:42}",
                ok,
                format!("exit_code={}, success={}", out.exit_code, out.success),
            );
        }
        Err(e) => record("exec error path: exit 42", false, format!("unexpected transport error: {e}")),
    }

    match backend.exec(ExecRequest::shell(
        "ls /definitely/does/not/exist/akh-w2b-probe",
        Duration::from_secs(10),
    )) {
        Ok(out) => {
            let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
            let ok = !out.success && out.exit_code != 0;
            record("exec error path: ls of missing path -> non-zero exit", ok, format!("exit_code={}, stderr={stderr}", out.exit_code));
        }
        Err(e) => record("exec error path: ls missing path", false, format!("unexpected transport error: {e}")),
    }

    // -----------------------------------------------------------------
    // file verbs over the RemoteBackend's shell-command Filesystem path
    // -----------------------------------------------------------------

    let probe_path = format!("/tmp/akh-probe-{}", std::process::id());
    let probe_path = std::path::Path::new(&probe_path);
    let payload = format!(
        "phase50-w2b remote-backend probe\npid={}\nline-with-'quotes'-and-$dollar\n",
        std::process::id()
    );

    match backend.write(probe_path, &payload) {
        Ok(()) => record("file: write(/tmp/akh-probe-<pid>)", true, "write returned Ok".into()),
        Err(e) => record("file: write(/tmp/akh-probe-<pid>)", false, format!("error: {e}")),
    }

    match backend.read_to_string(probe_path) {
        Ok(read_back) => {
            let ok = read_back == payload;
            record(
                "file: read_to_string round-trip matches written payload",
                ok,
                format!("read {} bytes back", read_back.len()),
            );
        }
        Err(e) => record("file: read_to_string round-trip", false, format!("error: {e}")),
    }

    let exists = backend.exists(probe_path);
    record("file: exists(probe) == true", exists, format!("exists = {exists}"));

    let is_symlink = backend.is_symlink(probe_path);
    record("file: is_symlink(probe) == false", !is_symlink, format!("is_symlink = {is_symlink}"));

    match backend.file_size(probe_path) {
        Ok(size) => {
            let ok = size as usize == payload.len();
            record(
                "file: file_size(probe) matches payload length",
                ok,
                format!("file_size={size}, payload.len()={}", payload.len()),
            );
        }
        Err(e) => record("file: file_size(probe)", false, format!("error: {e}")),
    }

    match backend.read_dir(std::path::Path::new("/etc")) {
        Ok(entries) => {
            let ok = !entries.is_empty();
            record(
                "file: read_dir(/etc) non-empty",
                ok,
                format!("{} entries, e.g. {:?}", entries.len(), entries.iter().take(3).collect::<Vec<_>>()),
            );
        }
        Err(e) => record("file: read_dir(/etc)", false, format!("error: {e}")),
    }

    let etc_passwd_symlink = backend.is_symlink(std::path::Path::new("/etc/passwd"));
    record(
        "file: is_symlink(/etc/passwd) == false (known non-symlink path)",
        !etc_passwd_symlink,
        format!("is_symlink = {etc_passwd_symlink}"),
    );

    // Clean up the probe file we wrote (best-effort — exec1 is disposable
    // anyway, but no reason to litter /tmp on a long-lived guest).
    let _ = backend.exec(ExecRequest::shell(
        format!("rm -f -- {}", probe_path.display()),
        Duration::from_secs(10),
    ));

    // -----------------------------------------------------------------
    // posture / handshake
    // -----------------------------------------------------------------

    let posture = backend.handshake();
    let ok = posture.confinement == Confinement::RemoteUnverified;
    record(
        "posture: handshake() reports Confinement::RemoteUnverified",
        ok,
        format!("{posture:?}"),
    );

    // -----------------------------------------------------------------
    // audit trail: honest gap report, not a fabricated pass
    // -----------------------------------------------------------------
    //
    // ADR 082 §7 describes an eventual `{ akh_id, session_id, agent_id,
    // backend, host, os_user, exit_code, correlation_id }` envelope on
    // `AuditKind::ToolInvocation`. As of this run that envelope does not
    // exist: `src/audit.rs`'s `AuditKind::ToolInvocation` carries only
    // `tool_name`/`params_summary`/`success`/`output_summary`/`duration_ms`
    // (no backend/host/os_user/correlation_id fields), and `RemoteBackend`
    // itself never calls into `src/audit.rs` at all — logging happens one
    // layer up, in `ToolRegistry::execute` (src/agent/tool.rs), which is
    // backend-agnostic by construction (it audits *tool* calls, not which
    // `Backend` served them). Driving `RemoteBackend` directly here (as
    // this example does, bypassing `ToolRegistry`) produces *no* ledger
    // entry at all. This is real, unbuilt 48f/48f-adjacent scope — flagged
    // here rather than papered over.
    println!(
        "\nNOTE  audit trail: no ADR-082 §7 correlation-index envelope exists yet (48f open \
         work) — RemoteBackend has no audit hook, and the generic ToolRegistry ledger has no \
         backend/host/os_user/correlation_id fields to distinguish this run from an in-process \
         one. The W3 runner cannot yet consume a per-command remote audit trail from this path."
    );

    // -----------------------------------------------------------------
    // summary
    // -----------------------------------------------------------------

    let total = results.len();
    let passed = results.iter().filter(|(_, ok, _)| *ok).count();
    println!("\n=== SUMMARY: {passed}/{total} checks passed ===");
    if passed != total {
        for (name, ok, detail) in &results {
            if !ok {
                println!("FAILED: {name} -- {detail}");
            }
        }
        std::process::exit(1);
    }
}

#[cfg(not(feature = "remote-executor"))]
fn main() {
    eprintln!("build with --features remote-executor");
}
