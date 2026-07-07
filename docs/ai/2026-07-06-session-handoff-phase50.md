# Session handoff — Phase 50 fleet run (2026-07-05 → 07-06)

**Purpose:** resume point after context compaction.

---

## UPDATE 2026-07-07 (session 2) — W3 runner LIVE, pipeline validated end-to-end

develop advanced `9eb6d15` → **`df270cb`** (+ user's own `4a1df0b` phase51 commit interleaved). New this session:
- **48f audit execution-envelope** (`8938356`) — `ExecLocus` on every `Backend` + `ExecEnvelope` stamped on each `ToolInvocation` audit record (backward-compat: JSON + bincode legacy). `df270cb` fixes a bincode round-trip bug (`skip_serializing_if` is incompatible with positional bincode).
- **`Agent::with_backend` seam** (`4239f6c`) — inject a caller-supplied executor `Backend`; `Agent::new` delegates `None` byte-identically.
- **W3 runner** (`650571a`, `examples/phase50_w3_runner.rs`) — drives a cloud model through the 48c executor against live exec1, runs the illumos-e2e corpus (setup→`run_agentic_turn`→verify), deterministic + manpage-grounding grade, emits `harness_core::RunRecord`. `--dry-run` validates plumbing without cloud spend.
- **★ RemoteBackend multi-KB write bug fixed** (in `650571a`) — exec1's `openssl 1.0.2u` truncates `base64 -d` of a single unwrapped line past ~500 B; W2b only probed 300 B so missed it. Fixed by MIME-wrapping at 76 cols. Real task scripts are multi-KB — this would have silently corrupted every setup.sh.
- **★ FIRST REAL RESULT**: glm-5.2, through the executor against live illumos, autonomously diagnosed + fixed a maintenance-stuck SMF service — **deterministic PASS, score 1.000, 79.5s, 11 requests**. The whole Phase-50 thesis (VM → executor → agentic akh → deterministic grade → RunRecord) is validated end-to-end.
- **★ FIRST REAL LEADERBOARD PUBLISHED** (`30c8326`): all 3 models ran the full 15-task illumos-e2e corpus through the executor vs live exec1 → **each 8/15 (53.3%, mean 0.533)**. The 53.3% is a **substrate cap** (guest has no cc/cargo → 6 c-\*/rust-\* compile tasks fail uniformly; rbac-03 = real shared miss; 8/9 on toolchain-free tasks). Inter-model signal = **manpage grounding** (deepseek least/parametric, qwen most). `benchmarks/results/2026-07-07-illumos-e2e-*` + `NOTES.md` + regenerated `LEADERBOARD.md`; site is gitignored (regen `bench-tool site`). Runs are sequential (shared guest); launch script `/tmp/w3-run.sh <model> <slug>`, `OLLAMA_API_KEY` from env-file `OLLAMA_CLOUD_KEY`. Design doc: `docs/ai/2026-07-07-phase50-w3-runner-design.md`.
- **Next for a fair full-15 v2**: provision `gcc`+`rust` on exec1 (DNS+egress work) or stand up the `builder` VM, then re-run the 3 models.
- exec1 keys restored to `tests/vm/.keys/` (from worktree `agent-a7eae87aa87ddccbd`); a temp copy at `/tmp/akh-exec_ed25519`.

Still open after this: 48b LocalDrop (needs akhomed on illumos — builder VM), W4 advisory, W6 orchestration bench, Terminal-Bench adapter. Push still blocked (locked signing key; all commits unsigned).

---

## 1. Repo state

- Branch **`develop`** @ `9eb6d15`, working tree clean, **full suite green incl doctests** (2812 lib + 105 integration + benchmark/harness suites).
- **Push is BLOCKED** all session (SSH signing key locked) → every commit is unsigned (`--no-gpg-sign`). Nothing has been pushed; it's all local on `develop`.
- Commits end with the session trailer `Claude-Session: https://claude.ai/code/session_014mtRPRbLyJoMf2ZXxEDNXb`.

## 2. ★ LIVE VM — exec1 (ephemeral, running now)

A booted OpenIndiana guest is **running right now** (detached qemu; survives conversation compaction but NOT a host reboot or `vmctl down`).

- **exec1**: `root@127.0.0.1:10042` and `akh-exec@127.0.0.1:10042` (uid100, non-root, the executor user). Has pfexec/man/svcs/dladm/ipadm/kstat; **no gcc** (no toolchain).
- **pg**: `127.0.0.1:10043` (up, provisioning-content incomplete).
- **Keys staged** (gitignored elsewhere) at:
  `/tmp/claude-1000/-home-toasty-ws-agi-akh-medu/522bb62f-31de-4b5f-b55d-1c4dd419692b/scratchpad/exec1-keys/{akh-exec_ed25519,root_ed25519}`
  (also in worktree `.claude/worktrees/agent-a7eae87aa87ddccbd/tests/vm/.keys/`).
- **SSH gotcha** — ssh-agent/config inject default keys and trip the guest's MaxAuthTries. Always:
  ```sh
  SSH_AUTH_SOCK= ssh -F /dev/null -o IdentitiesOnly=yes -i <key> \
    -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
    -p 10042 akh-exec@127.0.0.1
  ```
- **Run the executor validation** (proves 48c against the live guest):
  ```sh
  export CARGO_TARGET_DIR=/home/toasty/ws/agi/akh-medu/target
  AKH_EXEC1_SSH=akh-exec@127.0.0.1:10042 \
  AKH_EXEC1_KEY=<staged>/akh-exec_ed25519 \
    cargo run --features remote-executor --example phase50_remote_exec1
  ```

### Reproduce the VM from clean (if it dies)
```sh
cd tests/vm
bash scripts/gen-keys.sh                      # mints .keys/{root,akh-exec}_ed25519
scripts/bake-hipster-cloud-image.sh \         # offline qemu-nbd+zpool bake
  ~/.local/share/vmctl/images/openindiana-hipster-cloud.img \
  ~/.local/share/vmctl/images/openindiana-hipster-cloud-fixed.qcow2 \
  .keys/root_ed25519.pub
./smoke.sh                                     # PATH-shims q35→pc, vmctl up, checks
```
Teardown: `vmctl down --name pg --destroy` (keep exec1) / `vmctl down --name exec1 --destroy`.

### Why the VM was hard (root-cause chain — do not re-derive)
1. vmctl hardcodes qemu `-machine q35`; illumos vioblk panics mapping modern virtio-pci CAP → **q35→pc PATH shim** (`tests/vm/scripts/qemu-illumos-shim/`).
2. `openindiana-rust-ci` image = **botched build**: Linux-ELF sysconfig binary (SIGKILLed), no cdrom driver (seed unreadable), missing dladm/ipadm. Abandoned it.
3. `openindiana-hipster-cloud.img` (OI 2025.04) is unbroken but its root pool is named **`oirpool`** while illumos's metadata agent hardcodes **`rpool`** → agent aborts before configuring networking AND DNS.
4. Fix baked offline (`bake-hipster-cloud-image.sh`): declarative DHCP (`/etc/hostname.vioif0`+`/etc/dhcp.vioif0`) + DNS (`resolv.conf` 10.0.2.3/1.1.1.1 + `nsswitch.conf` `hosts: files dns mdns` — **OI doesn't wire DNS by default**) + static root key + PermitRootLogin + sudo→pfexec shim.
5. QEMU usernet DNS is `10.0.2.3`. Console gotcha: vmctl runs `-nodefaults` no-VGA, so serial console.log goes silent on a HEALTHY boot — use `-vga std` + QMP `screendump` to actually see the guest.

## 3. Phase 50 build status (ADR 086 + plan `docs/ai/plans/2026-07-05-phase50-feature-test-benchmarks.md`)

| Item | Status |
|---|---|
| **W7** results schema + LEADERBOARD.md + HTML site-gen | ✅ `benchmarks/harness-core/` |
| **W2a / 48c** russh RemoteBackend | ✅ `src/backend/remote.rs`, feature `remote-executor` (off by default) |
| **W3** 15-task illumos E2E corpus | ✅ `benchmarks/illumos-e2e/` (runner NOT built) |
| **W1** OpenIndiana VM substrate + live exec1 | ✅ boot cracked, reproducible; pg/builder provisioning-content deferred |
| **W2b** 48c validated vs live illumos | ✅ 15/15, fixed OI-no-`base64` write bug |
| **W5** public benchmark legs | ✅ 9 real rows (see §4) |
| **48f** ADR-082 §7 audit envelope | ✅ (2026-07-07) `8938356`+`df270cb` — full ExecLocus layer + ExecEnvelope on ToolInvocation |
| **Agent::with_backend seam** | ✅ (2026-07-07) `4239f6c` — binds RemoteBackend to an akh's executor tools |
| **W3 runner** (drive akh + grade) | ✅ (2026-07-07) `650571a` `examples/phase50_w3_runner.rs` — proven live, glm-5.2 solved smf-01 |
| **48b LocalDrop** (RBAC drop in exec1) | ❌ needs akhomed built for illumos |
| **W4** advisory mode / **W6** orchestration bench / Terminal-Bench adapter | ❌ |

## 4. Benchmark results (first real rows — `benchmarks/LEADERBOARD.md`)

Models: `deepseek-v4-pro`, `glm-5.2`, `qwen3.5:397b` (ollama.com cloud; key `OLLAMA_CLOUD_KEY` in `/home/toasty/env-file`). Driven via a dedicated akhomed:8301 + standalone PG (production akhomed:8201 untouched), budget-capped ~110 requests.

| Harness | deepseek | glm-5.2 | qwen |
|---|---|---|---|
| frontiercode (SWE-bench Diamond-style) | 0/2 | 0/2 | 0/2 (matches "frontier≈0 on Diamond") |
| LoCoMo | 0.20 | 0.20 | 0.00 |
| LongMemEval (oracle) | 50% | 50% | 67% |

Regenerate: `cargo run -p harness-core --bin bench-tool -- leaderboard …` / `… site …`.

## 5. Recommended next wave (highest-value order)

1. **48f audit envelope** — add backend/host/os_user/correlation_id to `AuditKind::ToolInvocation` (`src/audit.rs`) + an audit hook in `RemoteBackend`. Unblocks manpage-grounding grading.
2. **W3 runner** — per-task vmctl snapshot-restore → run `setup.sh` (root) → drive an akh (a cloud model) through the 48c executor at `akh-exec@exec1` → run `verify.sh` → scrape audit trail for `expected_manpages`. Emit RunRecords via `harness_core::store::write_run`. Corpus + executor are ready.
3. **48b LocalDrop** — needs an akhomed binary ON exec1 (no toolchain there → finish the `builder` VM's rust toolchain provisioning first, or cross-build).
4. **W4 advisory** (engineer-sim + rotation judge), **W6 orchestration bench**, **Terminal-Bench adapter**.

## 6. Items that need YOU (not the fleet)

- **`vm-manager` repo** (`/home/toasty/ws/nebula/vm-manager`): I reconstructed a clean **`salvage/virtio-transitional`** branch (commit `ec950dd`, builds + 19 tests) from the messy stash-pop conflict your pull created. Review + push it (and decide on the machine-type knob / OCI-in-RAM-buffering upstream fixes). `main` is untouched; a safety patch of the original messy state is in scratchpad `vmmgr-salvage/`.
- **Optional image rebuild**: refraction-forger recipe should base on hipster-cloud, name the pool `rpool` (or ship declarative DHCP+DNS), and bake the rust toolchain for `builder` — then the bake-workaround becomes unnecessary. Full recipe in `docs/ai/2026-07-06-phase50-w1c-image-defects.md`.
- **Push** develop once the signing key is unlocked (all session's commits are local).
- **Seshat/Anima**: ADR 086 + Phase 50 plan already ingested to Seshat. Anima work-item creation still needs you (MCP not reachable from here).

## 7. Working constraints (carry into next session)

- Fleet scheduling rules: `[[feedback-fleet-scheduling]]` in memory — sonnet implements, worktrees must `git reset --hard develop` first (they base on stale main), shared `CARGO_TARGET_DIR` (no doctests while agents build; gate on `--lib --tests`), `setsid`-detach long ops, ≤3 build-heavy agents, don't re-pull the 14GB OCI images (vmctl OOM bug).
- Don't touch `/home/toasty/ws/nebula/{vm-manager,sysconfig}` from fleet agents (user's repos; sysconfig is read-only reference).
- Key repos/paths: `docs/ai/2026-07-06-phase50-w1c-image-defects.md` (image forensics + rebuild recipe), `docs/ai/2026-07-06-phase50-w2b-remote-exec1-validation.md` (executor validation), `tests/vm/` (substrate).

## 8. This session's full commit arc on develop (most recent first)
`9eb6d15` W2b merge · `f9a2556` W1d follow-ups (exec1 LIVE) · `7a230df` W1d · `1cc2242` W5 · `667b90d` W1c · `2a2949b` W1 · `bf5dbce` W3 · `92ffc17` W1b · then the 7-phase run (Phase 41/40/34/29/48a/33a/ADR-081) + `66ab22c` ADR 086.
