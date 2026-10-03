# Lilith historical receipt pilot

Status: pilot completed after operator go, 2026-10-02 UTC. Next development work follows [the October 3 reconciled plan](2026-10-03-lilith-train-now-reconciliation.md).
Anima: WI-526 (`01a0f96b-54fc-7c00-b851-1025a945beae`), under the Lilith C-series plan.
This supplements `2026-10-01-lilith-c5-c7-dojo-and-training.md`. It does not
replace C7's guided-RL round, its exam floors, or the fresh recorder holdout.

## Completed pilot and next stage

PR #510 merged as `89b1e40b73ebfbf3349c0c5771d06a8344da4642`. Pod `784ptttwzqhho2` completed 200 optimizer steps. Parent exact receipt answers: 0/12 development and 0/8 historical; candidate: 12/12 and 8/8. Fresh-process reload matched probe logits/argmax with zero additional optimizer steps. The candidate is inactive. The pod was confirmed stopped/EXITED at 2026-10-02T22:46:48Z; artifacts were copied and readback-verified on NFS.

Candidate SHA-256: `d7e46c5905dbfa042810e23da7d75111acec539718840e8c3427cf9d1185b9b1`.
Private result: `/Users/toasty/lilith-private/history-20261001/launch/pilot-summary.json`.
Backup: `/Volumes/akh-medu-runs/lilith/history-pilot-20261001-prepared/pod-784ptttwzqhho2`.

The recipe and hold/go descriptions below are the original preparation record. They do not describe current launch authorization or current GPU capacity. The receipt pilot did not train the broader voices/CoS curriculum. That is the next development stage, with no fourteen-day training-start wait.

## Authorized scope and evidence

The operator requested historical evaluations from harvested transcripts,
Hephaistos wind-down, and preparation of a first historical-data training run.
Starting or restarting Runpod and GPU training requires a subsequent go. The
operator separately authorized graceful shutdown of QEMU `drmtest-r5a-run1`.

Local ADR 119 Claude/Codex transcript copies provide dated, UUID-bound tracker
tool results. Only complete outer JSON envelopes become facts. Truncated
results, failed reads, mismatched identities, missing timestamps, and unsupported
assistant prose are excluded. This is a bounded historical tracker-state
projection, not reconstructed complete C3 snapshots or project-wide truth.
Source-file hashes and call/result sequence numbers preserve provenance.

The cutoff is exclusive `2026-09-28T00:00:00Z`. The October 1 import scanned 545
files and admitted 338 receipts: 262 Codex and 76 Claude, observed August 22
through September 27. Targets cover recorded phase/park reason and correction
of a deliberately contradicted memory claim. Those memory seeds are labeled
synthetic; they are not attributed to past model failures. No original assistant
answer is a positive target. No historic probabilities become on-policy/KLPO
probabilities, and this pilot invents no preference labels.

Whole sessions connected through a shared WI remain in one split. Most receipts
form one large component; it is assigned to training, with the newest remaining
independent components held out. This is explicitly retrospective, not a
chronological release evaluation. Exact substantive fact hashes are also checked.

| Split | Receipts | Lessons | Sessions | WIs |
| --- | ---: | ---: | ---: | ---: |
| Training | 318 | 636 | 35 | 172 |
| Development | 16 | 32 | 6 | 9 |
| Historical evaluation | 4 | 8 | 2 | 3 |

The small historical evaluation is development evidence, not certification.
Fresh C3 recorder data stays outside the corpus. A later full-role exam still
needs its own sealed window and decontamination against whichever training
sources it actually uses. Earlier C5 transcript grades are not factual-grounding
certificates and are not used as targets here.

## Frozen first-run recipe

- Parent: `google/gemma-4-12B-it`, revision
  `5926caa4ec0cac5cbfadaf4077420520de1d5205`, existing model/config/tokenizer pins.
- Tokenization: Transformers 5.10.4, tokenizers 0.22.2, Jinja2 3.1.6; exact
  tokenizer, tokenizer-config and chat-template hashes in private manifest.
- Template: pinned no-thinking generation prefix, including its empty thought
  channel. Prefix/receipts are context only; loss covers answer and turn-end only.
- 8K context ceiling; actual examples at most 886 tokens. No truncation, packing,
  or padding to manufacture an 8K workload. All 636 training examples fit.
- Existing full-depth language projection trainer; frozen BF16 parent,
  rank/alpha 16, seed 73, learning rate `1e-4`, batch one, efficient SDPA with
  math fallback, CE chunk 1024, attention query chunk 512.
- Maximum 200 optimizer steps, three epochs, 5400 training-process seconds.
  Reserve 900 seconds inside that process for save and candidate evaluation.
  The 200-step pilot samples a bounded subset, not three complete corpus passes.
- Deterministically select at most 12 development and 12 historical-evaluation
  cases. Use identical cases and greedy generation before and after training;
  exact JSON fields, missing/extra fields, invalid/duplicate JSON keys are graded.
- Save only a candidate adapter. Perform the existing fresh-process tensor/file
  and logits/argmax reload control with zero optimizer steps. No activation,
  active-slot write, release selection or capability claim follows from passing.

The pod must be H100, CUDA >=13, driver >=580 and pass the existing bootstrap
check. Its payload is built only after the code lands on `origin/main`, using
`scripts/lilith/runpod/make_payload.sh`. Whole repository transfer is forbidden.
The private data archive is separate from the source payload.

## Launch and cost control

No resource is provisioned by these scripts. At go time, recheck stock, account,
volume, driver constraints and exact main/source/data hashes. The existing
100GB volume `c6kdciozex` is in EU-FR-1. The read-only October 1/2 catalog showed
H100 SXM at USD 3.49/hour, but no EU-FR-1 stock. Two GPU hours would be USD 6.98
at that observed rate, excluding storage; refresh the quote before provisioning.
Do not silently move regions or allocate another volume. Do not restart the old
serving pod: its startup command automatically starts the serving workload.

Create an attributed approval receipt only after actual operator go. It binds
run ID `lilith-history-r1-20261001`, data manifest/tokenization hashes, merged main
SHA, approved pod ID and observed pod creation timestamp, and expressly enables
stop at completion/deadline. The prepared receipt currently says `hold`.

Start the host supervisor before launching the workload; keep that host awake
(for example under `caffeinate`). `supervise.py` checks the completion marker,
copies and hashes artifacts, and stops the exact approved pod on completion,
failure or two hours after its observed creation time. It uses the caller's
enrolled `agent-tools runpod` delivery, with no 1Password fallback. A stop request
is not success until the control plane reports stopped/EXITED. This is a
host-enforced guard, not a native Runpod lifetime guarantee; host supervision
loss needs operator intervention. Network-volume storage continues after stop.

On the approved pod, after uploading and verifying the two archives:

```sh
/workspace/lilith/env/bin/python <payload>/scripts/lilith/history/run_pod.py \
  --data /workspace/lilith/data/history-r1 \
  --approval /workspace/lilith/data/history-r1/operator-approval.json \
  --main-sha <merged-main-sha>
```

`run_pod.py` verifies approval and main-only source before any GPU operation.
It checks the bootstrap, runs the bounded pilot, cold-reloads in a fresh process,
and writes a checksum inventory before the final `history-result.json` marker.
Failure and timeout evidence remain on the network volume.

## Wind-down and remaining gates

Final Hephaistos transcript harvest succeeded. The Mac parent-baseline runner was
stopped and its partial results retained locally and on NFS; its 385 selected
episodes are not a completed baseline. Lilith's llama-server exited; systemd
reported failed after its stop timeout, so this is not a clean service-exit claim.
QMP `system_powerdown` was accepted for the separately owned `drmtest-r5a-run1`
guest and the QEMU process subsequently exited. Its disk is preserved. Other
containers and services were left under their existing ownership. SSH later timed
out; power-off of the physical machine itself was not witnessed. The parent and
data copies on the independently accessible NFS share remain available for the
pilot, so prolonged Hephaistos operation is not a prerequisite.

Original preparation gates were subsequently passed for this pilot: merge, go, main payload, bootstrap, training/evaluation/reload and confirmed pod stop. Anima lifecycle acceptance remains distinct from those artifact receipts. The next mixed curriculum still needs its own export, launch preparation and actual execution evidence; see the reconciled plan.
