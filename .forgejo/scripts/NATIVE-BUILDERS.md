# Native release builds

`run-platform-build.py` connects Buildx to persistent AMD64 BuildKit on Archibald
and ARM64 BuildKit on armini. Explicit platform assignments select native workers.
The existing pipeline retains all image, revision, heap-profile and native-provider
proofs before promoting `main-ready`; an unavailable builder fails the job.
PRs retain the existing non-publishing validation path.

The two `AKH_BUILDKIT_*_SSH_KEY` Actions secrets are delivery copies of dedicated
OpenBao keys. The wrapper writes private temporary SSH files, verifies pinned host
keys, and removes key variables from subprocess environments. Server-side forced
commands allow only BuildKit stdio. Keys, operator credentials and host agents are
never forwarded into compilation containers. Client registrations and temporary
files are removed when the wrapper exits; daemon caches persist across jobs.

Each daemon serializes build stages. QEMU in the ephemeral CI VM is used only for
short ARM64 image smoke tests, not compilation. Builder installation, limits,
rotation and rollback are documented in the homelab repository's
`scripts/akh-ci-builders/README.md`.

Validation: `test_native_builders.py` checks credential isolation, unavailable-key
failure, PR behavior, architecture routing and cleanup. A live provisioning smoke
test additionally built each architecture and checked x86_64/aarch64 output, and
verified that both dedicated SSH keys reject arbitrary commands.
