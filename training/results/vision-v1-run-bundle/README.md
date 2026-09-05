# V1 A40 run provenance bundle

This directory preserves the small inputs and interrupted-attempt logs needed
to interpret the final compact native-vision run. It contains no credentials
and no model weights.

- `Cargo.toml` is the deliberately reduced two-crate CUDA workspace used on
  the disposable Runpod host. It is evidence, not a replacement for the
  repository root workspace manifest.
- `Cargo.lock` is the lockfile resolved in that reduced workspace.
- `grug-data-Cargo.toml` and `grug-train-Cargo.toml` are the exact crate
  manifests paired with it (renamed only to keep this evidence directory flat).
- `vision_compact_overfit-run.rs` is the byte-exact training example used for
  the final run. The maintained example adds periodic model/Adam progress
  snapshots after the missing-checkpoint attempt exposed that operational gap.
- `attempt-01-no-checkpoint.log` was stopped when the missing persistence path
  was noticed.
- `attempt-02-context256-budget-stop.log` was stopped when 96 visual tokens
  were found to consume 37.5% of its 256-token context.
- `attempt-03-fixed-rate-plateau-stop.log` was stopped after the loss settled
  into a repeatable alternating floor, triggering the cosine-decay response.
- `attempt-04-cosine-pass.log` is the completed 4,000-step gate log.
- `periodic-checkpoint-smoke.{json,log}` validates the maintained source's
  post-run checkpoint hardening. Its one-case capability gate is expected to
  be false; the assertion is one complete progress snapshot plus one complete
  final snapshot.
- `local-offline-time.log` is `/usr/bin/time -l` output from the three-case
  network-denied CPU replay.

`SHA256SUMS` is generated after the completed run and includes the final log
and result as well.
