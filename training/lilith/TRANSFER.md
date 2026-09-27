# Transferring Lilith to another machine

Check out `feature/lilith-runtime-training-runpod`. Start with
[the Astra Runpod handoff](../../docs/ai/plans/2026-09-22-lilith-runpod-handoff.md).

The branch includes Lilith source, tests, benchmark/task definitions, transcript
admission policies and hash manifests, and selected JSON/Markdown evidence. Run
results are historical evidence, not live services or proof that training finished.
Both local runtime training attempts ended before any optimizer update.

## Not included in Git

Transfer these privately when needed; do not expect a clone to contain them:

- Parent model, tokenizer and associated files from
  `~/.local/share/akh-medu/lilith-training/parent-5926caa4/`.
- Admitted transcript subset from
  `~/.local/share/akh-medu/lilith-training/transcripts-20260922/sft-subset-v1.jsonl`.
  Its committed manifest pins the hash. Raw harvested transcripts stay private.
- Ptah's model/tokenizer if doing reviews on the cloud instance.
- Built daemon binaries and the Python environment; rebuild or provision for CUDA.
- Private daemon configuration, credentials and VM access. Recreate a dedicated
  stack; do not transfer shared credentials by putting them in this repository.
- Generated model/optimizer tensors, raw trajectory dumps, numeric binary probes,
  SQLite databases, process markers, logs, and most JSONL run output.
- Credential incident/rotation maintenance files and the source host's SSH key.

Some historical reports reference those excluded artifacts. Preserve their hashes
and status as historical records; regenerate new artifacts in new output directories.
Historical SHA256SUMS files describe the original experiment, not the reduced Git
transfer subset. Do not rewrite them to hide omitted files.

The source machine retains excluded files. No source artifacts were deleted as
part of this commit. The user's shared-credential rotation remains deferred.

## Checks performed before the transfer commit

35 focused Python tests passed: runtime stack/worker (10), training engine/cache/
transport (11), preparation (10), and LoRA metadata safety (4). Real-daemon
`runtime-training-stack-20260922/fixture6` already passed missing-read, create,
read, revision-checked overwrite and readback. A full Rust/GPU rebuild was not
repeated for this packaging commit; run the appropriate build and CUDA controls
on the destination before starting the full experiment.
