# Lilith transcript admission — 2026-09-22

This directory defines the CPU-side admission boundary for using harvested
coding sessions to train Lilith's Gemma policy through the ADAM/Kaine reward
curriculum. It does not contain private transcript text.

## Availability

The supported `akh-transcripts-mcp` endpoint exported three complete sessions
for the Linux-host repository path on 2026-09-22: 17,218 rendered events. The
initial client failed because it did not preserve `Mcp-Session-Id` and send the
initialized notification; the repository exporter implements that lifecycle.
No database fallback was used.

The first Linux-host slice was quarantined because its three sessions are the
live Lilith experiment sessions and therefore contain evaluation/protocol
leakage. Policy v2 froze those session hashes as denied before exporting the
historical repository cohort. The preferred private v2 artifact contains 559
records from 12 complete sessions: 359 train and 200 validation, assigned by
whole session. All are `supervised_imitation`; zero have external verifier
evidence and therefore zero are eligible for RL reward. The public
[`admission-manifest-v2.json`](admission-manifest-v2.json) contains only counts,
policies, and hashes. This is a usable distillation/SFT slice, but it makes no
RL or quality-gain claim.

## Frozen boundary

[`admission-policy.json`](admission-policy.json) freezes held-out session hashes,
family tags, and content fingerprints before export. The admission tool accepts
only complete, rendered and redacted `get_session` results. It pairs prompts
with replies, applies another conservative redaction pass, deduplicates normalized
content globally, and assigns whole sessions to a deterministic 90/10
train/validation split.

Rows without external evidence are `supervised_imitation` and have no reward.
A `verified_outcome` row requires a nonempty `trajectory.steps`, a passed
external verifier, the verifier's SHA-256, and an outcome SHA-256 equal to the
canonical trajectory hash. The ADAM/Kaine reward adapter can consume only those
verified rows. Its independently frozen evaluation sessions and families must
be added to this policy before any export.

The intended bounded first slice is complete sessions associated with this
repository, capped during MCP export before inspection. Raw transcript exports
and admitted JSONL stay in the private training store; only aggregate manifests
and hashes belong here.

## Frozen SFT subset

[`sft-subset-v1-manifest.json`](sft-subset-v1-manifest.json) freezes an 8-train,
8-validation subset for the proven Gemma runner envelope. Selection hashes
opaque record IDs and applies only predeclared structural and token-length
filters. The pinned Gemma tokenizer and chat template reject any example over
512 total tokens and never truncate assistant output. Validation records remain
from the admission dataset's held-out sessions. The subset is an SFT control;
it has no reward-bearing rows.

Run the unit boundary (the project environment may require its test runner):

```sh
python -m pytest -q scripts/session-platform/test_lilith_transcript_admission.py
```
