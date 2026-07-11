# illumos-e2e **v2** — toolchain-provisioned compile sweep (2026-07-07)

The v1 run (`2026-07-07-illumos-e2e-NOTES.md`) hit a **substrate cap**: the guest
had no C compiler and no Rust toolchain, so all six compile-family tasks
(`c-01/02/03`, `rust-01/02/03`) failed uniformly across every model — a property
of the box, not the models. This v2 sweep re-runs exactly those six tasks on a
**toolchain-provisioned** `exec1`, so the compile families can be graded fairly.

## What changed on the guest (the afternoon's real work)

The provisioning was not "pkg install gcc". The minimal cloud image fought back:

1. **Disk** — the virtio disk was 1.95 GB, root pool 90 % full (155 MB free).
   Grew exec1's **per-VM overlay** (+12 GB, not the shared base) and expanded the
   zpool → 13.6 GB / 12.3 GB free.
2. **Boot** — the resize made OVMF default to the EFI shell, and `vmctl start`
   launched with `-machine q35`, whose **AHCI controller kernel-panics illumos**.
   Recovered by hand-launching with `-machine pc` (the documented shim) and
   driving the UEFI shell over the serial console to chain-load `bootx64.efi`.
3. **Version lock** — gcc-14 / rustc refused to install: the image was pinned by
   `userland-incorporation@2025.0.0.28617`, which excludes the 2026 binutils
   modern gcc needs. Required a **full `pkg update`** (190 install + 492 update,
   2.8 GB, new BE).
4. **OOM** — the first `pkg update` wedged the VM (sshd couldn't emit a banner):
   684 packages on 4 GB RAM with **no swap** OOM-thrashed. pkg is transactional,
   so the old BE stayed bootable — recovered with zero data loss. Added a **4 GB
   swap zvol + 6 GB RAM**; the retry completed.
5. **Toolchain** — booted the new BE (`illumos-c4b084c090`, was `-3caf57556b`),
   installed `developer/gcc-14` (14.3.0) + `developer/lang/rustc` (1.96.0) +
   `gnu-make`, added a `cc`→gcc symlink, and **sanity-compiled** libkstat +
   libscf C and a Rust binary before spending any cloud budget. All linked+ran.

**Methodology finding:** the illumos-e2e substrate needs a *toolchain-baked
image*, not runtime provisioning. That is now the recommended path for v3.

## Result — the compile families are genuinely hard

| Model | v1 compile | **v2 compile** | Passed (v2) |
|---|---|---|---|
| glm-5.2 | 0/6 (blocked) | **2/6** (0.333) | c-02, rust-03 |
| deepseek-v4-pro | 0/6 (blocked) | **1/6** (0.167) | rust-03 |
| qwen3.5:397b | 0/6 (blocked) | **0/6** (0.000) | — |

## The tie breaks — corrected full-15 (v1 9 non-compile + v2 6 compile)

v1 had **all three models tied at 8/15** — an artifact of the six blocked tasks
scoring zero for everyone. With the compile families fairly graded, the models
**differentiate**:

| Model | Corrected full-15 | Δ vs v1 |
|---|---|---|
| **glm-5.2** | **10/15 (66.7%)** | +2 |
| **deepseek-v4-pro** | **9/15 (60.0%)** | +1 |
| **qwen3.5:397b** | **8/15 (53.3%)** | +0 |

That is the headline the substrate cap was hiding: **the toolchain sweep turned a
flat three-way tie into a real ranking.** glm-5.2 is the strongest illumos systems
programmer of the three; qwen the weakest (note this is a different axis from W6
fleet-orchestration, where glm *regressed* — capability is task-shaped).

## Why they fail — API avoidance, not compile errors

The verify scripts are **anti-cheese**: they check the program actually *uses* the
required native API, not just that output looks right. The dominant failure mode
across all three models is **shelling out instead of doing the hard FFI**:

- `rust-01` (processor_info(2) via FFI): **all three** wrote
  `std::process::Command` shelling to `psrinfo` → all fail.
- `rust-02` (Doors IPC): **all three** shelled out instead of the Doors API → all fail.
- `c-01` / `c-03`: models produced C that doesn't call `kstat_open`/
  `scf_simple_prop_get` (avoided the library, parsed CLI output instead).
- deepseek/qwen `c-02`: compiled and ran, but read the **wrong** SMF property
  (printed a hardcoded/other greeting instead of the live `config/greeting` value).

Manpage grounding was **high** here (mostly 2/3–3/3) yet they still failed —
**grounding ≠ FFI competence.** A model can read `door_call(3C)` and still write
`Command::new("psrinfo")`. That gap — knowing the primitive exists vs. being able
to *call it natively* — is exactly what a compiler-provisioned grader exposes and
a pass/fail-on-output grader would miss.

## Caveats

- **Single trial per (model, task).** Directions are robust; exact scores are not
  multi-trial medians.
- **Substrate seam:** the v2 compile tasks ran on the updated OS
  (`illumos-c4b084c090`); the v1 nine non-compile results were on the pre-update
  OS (`-3caf57556b`). The non-compile tasks (SMF, kstat CLI, RBAC) are
  version-insensitive, so combining them for the corrected full-15 is reasonable,
  but a fully clean v3 should re-run all 15 on the single updated image.
- **Grading:** deterministic `verify.sh` (autonomous), no LLM judge. ollama.com
  cloud models, bare published tags. Recipe/host tier recorded in each RunRecord.

## Provenance

Harness `illumos-e2e`; slugs `glm-5.2-v2` / `deepseek-v2` / `qwen-v2`;
`HostTier::Vm`. Regenerate board/site: `bench-tool leaderboard` / `bench-tool site`.
