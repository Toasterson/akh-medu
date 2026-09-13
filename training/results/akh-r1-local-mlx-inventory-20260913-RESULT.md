# Akh R1 / BabyLLM local-MLX inventory (Mac), 2026-09-02..09-13

Read-only survey of everything run locally with MLX on this Mac. Nothing was
deleted or modified; this is a map of what exists and where.

## Base models and adapters

| item | path | size | on disk now |
|---|---|---|---|
| `Qwen/Qwen3-0.6B` (upstream BF16) | `~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B` | 1.4G | yes (mtime 09-11, pulled for the R1 exam) |
| `mlx-community/Qwen3-0.6B-4bit` | `~/.cache/huggingface/hub/models--mlx-community--Qwen3-0.6B-4bit` | 335M | yes (mtime 09-02, White-belt v0's trunk) |
| `white-belt-v0` LoRA adapter (rank 16, dropout 0.05, scale 20, 16 layers) | `/Users/toasty/ws/akh-medu/training/character/adapters/white-belt-v0/` | 132M (6 checkpoints @ 23MB: iters 200/400/600/800/1000 + final `adapters.safetensors`) | yes, local-only, gitignored |
| MLX venv (uv-managed) | `/Users/toasty/ws/akh-medu/training/character/.venv-mlx` | 331M — `mlx 0.32.2`, `mlx_lm 0.31.3` | yes, local-only |
| 8 `akh-r1-*` runs with real LoRA checkpoints (see below) | `runs/akh-r1-{binding-training,choice-loss,curriculum-20260911-v3,full-exposure,memory-action-profile,memory-action-training,registry-training,repair-lessons}-*` | ~24GB combined (adapters + optimizer state) | yes, local-only, gitignored |

No other base model or MLX env was found on the Mac (checked `~/models`, `~/.mlx`, `~/.venvs`, homebrew python site-packages — all empty/absent for MLX).

## Executed MLX experiments

Two distinct efforts, three days apart:

### White-belt v0 character-SFT (2026-09-02, commits b7bc2173/4ca307a9/053c6651, all on `main`)
One QLoRA fine-tune of `mlx-community/Qwen3-0.6B-4bit` on the dojo kata/sparring corpus (`training/character/`), plus a cross-model reasoning exam bridging to the Rust `grug-train` scorer (`dump_rule_eval` / `score_rule_jsonl`).
- **Belt exam** (3 held-out pressures, 3 languages): fr silence **PASS**; en yes-and half-formed; de redirect-refusal **FAIL** (trunk lacks German at 0.6B/4-bit; val-loss min 3.486@iter300 vs 4.116@iter1000 — overfit past 300).
- **Reasoning exam** (rule corpus, 64+64 holdout): 350M reference organ 0.928/0.494 (chain 0.935); generic 0.6B zero-shot 0.000; 0.6B few-shot K=3 lenient 0.375/0.266 (label-copying, below majority baseline); white-belt adapter few-shot 0.016/0.000 (persona voice overrides task — validates ADR 099 "adapters situationally linked, never resident").
- Fully **committed**: `training/results/character-white-belt-v0-EXAMS.md`, scripts under `training/character/` and `crates/grug-train/examples/{dump_rule_eval,score_rule_jsonl}.rs`. Only the adapter binary + venv are local-only (table above); the eval JSONL streams were `/tmp` scratch and are gone (regenerable in ~10 min from the committed harness).

### `akh-r1-*` recovery/episode harness (2026-09-11 → 2026-09-13, ~305 local run dirs, ~39 GB)
An automated agentic test-generation loop (branch `feat/akh-r1-episode-exam`, worktree `~/ws/akh-medu-r1-exam`) that spun up one uniquely-named run directory per condition/experiment — `recovery-*`, `perception-*`, `neutral-*`, `language-*`, `service-*`, `ownership-*`, `production-*`, `identity-*`, etc. (26/22/19/12/11/9/7/7 dirs respectively; full 40+-bucket breakdown in `/tmp/mlx-inventory/prefix_counts.txt`).

| what | count |
|---|---|
| total local run dirs (`runs/akh-r1-*`, `runs/r1-*-smoke-*`) | 305, spanning 2026-09-11 → 2026-09-13, 39.1 GB total |
| dirs with a `summary.json`/`results.json` receipt (harness actually executed, even if model-free) | 196 |
| dirs with real MLX LoRA checkpoints on disk (actual gradient-descent training, not just harness plumbing) | 8: `binding-training-20260912-v1` (5.3GB), `choice-loss-20260911-v1`, `curriculum-20260911-v3` (1.1GB), `full-exposure-20260912-v1` (731MB), `memory-action-profile-20260912-v1`, `memory-action-training-20260912-v1` (6.6GB, largest single dir), `registry-training-20260912-v1` (3.0GB), `repair-lessons-20260911-v1` (596MB) |
| dirs that are pure prep/review (plan/status JSON, no model call, `model_used` field is `false` everywhere it's present — 0 hits for `true`) | remainder (~230+) |
| corresponding RESULT.md docs already **committed and pushed** on `feat/akh-r1-episode-exam` (`training/results/akh-r1-*-RESULT.md`) | 286 |

Headline numbers for the harness itself live in those committed RESULT docs (e.g. `akh-r1-R0-reference-20260912-RESULT.md`: 3/3 worlds passed logic, 0/3 passed the identity/source audit — "receipt IDs collided across goals in one runtime session"; `akh-r1-exam-20260911` / `akh-r1-integration-20260911`: `learned_model_success: false`, `release_certification: false` across all 8 sampling configs of the cached 0.6B). This whole branch is **not merged to main** (per prior repo note: "the drift branch, never merge") but every RESULT doc is committed on it and the branch is pushed to `origin` (`e7818965`..`33afdda0`), so nothing here depends solely on this Mac's disk.

## Local-only artifacts worth preserving

`/tmp/mlx-inventory/preserve.txt` lists 56 small JSON receipt files (528,970 bytes total, well under the 2MB cap) — `summary.json`/`results.json`/`status.json` from the 24 run-dir families that have **no matching committed RESULT.md** on `feat/akh-r1-episode-exam` (diff in `/tmp/mlx-inventory/{local_dir_stems_raw,committed_result_stems}.txt`), plus the 7 `r1-local-mlx.lock.*` identity/allowance files from `runs/babyllm-recovery-20260909/` (pin the exact hashes of every `.py` module + pack JSON an in-flight recovery run depended on — useful provenance even though the runs themselves are receipts, not weights).//
Recommend copying these into `training/results/akh-r1-local-mlx-evidence-20260913/` before the local `runs/` dirs are ever cleaned up.

Not included in preserve.txt (too large / not receipts): the 132MB white-belt-v0 adapter, the 8 real-training run dirs' safetensors (~24GB), the 331MB `.venv-mlx`, and the HF model caches (1.7GB) — all reproducible from committed scripts + public model IDs, not one-of-a-kind data.

## Gaps

- **24 local run-dir families have zero committed RESULT.md** on the r1-exam branch (`akh-r1-{binding-kata-20260912-flat,binding-kata-20260912-untuned,binding-lessons,choice-loss,compact-probe,compact-registry,context-transfer,effect-native-integration,exact-support-design,exam-20260911,four-selector-integration,goal-handoff-observed,grounded-native-integration,memory-action-profile,neutral-ambiguity-integration,public-revision-integration,public-selector-lifecycle,recall-authority-20260912,registry-context,registry-lessons,registry-seeded,repair-lessons,source-validation-20260912,typed-selector-integration}`) — these are exactly the ones captured in `preserve.txt` above. Some (`exam-20260911`, `choice-loss`, `repair-lessons`) are the earliest/most foundational runs in the sequence and their only record is the local JSON.
- **31 committed RESULT.md docs cite a run-dir stem that isn't on disk** under a matching name (e.g. `akh-r1-compact-evidence-20260912`, `akh-r1-four-selector-freeze-20260913`, `akh-r1-grounded-native-20260913`, `akh-r1-neutral-*-native-20260913` family, `akh-r1-system-experiments-20260912-REPORT.md`). Spot-checking a few suggests most are naming drift (`-freeze`/`-native`/`-v2`/`-v3` suffixes not present verbatim in the dir name, or the receipt lives one level deeper than `<dir>/summary.json`) rather than genuinely missing data — worth a closer pass if any specific one matters, but not re-verified individually here given the ~300-dir volume.
- The White-belt v0 `/tmp/rule_eval_outputs.*.jsonl` scratch files referenced in `training/character/reasoning_exam.py` are gone (as expected for `/tmp`); regenerating them is a ~10-minute rerun of the committed harness against the still-present adapter + base model, not a loss.
- `model_used` is `false` in every `akh-r1-*` summary.json that has the field (0 hits for `true`) — the harness itself never live-invoked a resident model mid-condition; the actual model calls happen in the separate training/eval scripts (`akh_r1/*.py`) whose own receipts are the summary/results JSON already accounted for above.
