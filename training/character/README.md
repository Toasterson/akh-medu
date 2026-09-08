# BabyLLM character SFT — White belt

Stage 1 of the dojo curriculum (`docs/ai/manuals/2026-09-01-babyllm-dojo-
curriculum.md`, §3 kata / §5 belts / §6 lineage): single-voice, clean-form
SFT on the White-belt kata drills, plus the Yellow-belt (multi-turn/cross-
language) drills and Green-belt sparring transcripts, LoRA-trained on top of
a small Qwen3 base with [MLX](https://github.com/ml-explore/mlx) on Apple
Silicon. This is the *character* track (Phase 51 / Grug-Brain covers
architecture and reasoning separately).

Everything in this directory except `convert_corpus.py`, `run_sft.sh`,
`lora_config.yaml`, `.gitignore`, and this README is generated/downloaded and
gitignored (`.venv-mlx/`, `data/`, `adapters/`) — nothing here is committed.

## 1. Regenerate the training data

Source corpus (committed, read-only):
`data/phase51/character/kata-drills.jsonl` (240 records) and
`data/phase51/character/sparring-green.jsonl` (90 records).

```sh
python3 convert_corpus.py
```

Writes `data/train.jsonl` and `data/valid.jsonl` in mlx-lm chat format
(`{"messages": [...]}`), split deterministically (seed 42, ~90/10,
stratified by `kata` for drills and by `adversary` for sparring so every
kata/adversary bucket appears in both splits). System prompt is composed
only from record fields: one fixed persona sentence (the model is a small
digital lifeform, honest about being one) plus `voice: <voice>` for kata
records. `why_this_move` / `observed_effect_on_person` / `shadow_avoided`
are deliberately *not* in the SFT targets — they're written to
`data/lineage.jsonl` for the future Black-belt transmission stage (§6).

## 2. Train

```sh
./run_sft.sh
```

Idempotent: creates `.venv-mlx` (uv-managed Python 3.12) if missing, installs
`mlx-lm`, regenerates the data, and runs LoRA SFT — skipping training if
`adapters/white-belt-v0/adapters.safetensors` already exists (set
`FORCE_RETRAIN=1` to redo). Base model: `mlx-community/Qwen3-0.6B-4bit`
(falls back to `mlx-community/Qwen3-0.6B-bf16` if the 4bit repo doesn't
resolve). Tuned for a 16GB M-series Mac — batch size 2, max sequence length
1024, LoRA rank 16 (`lora_config.yaml`), 1000 iterations. If you hit memory
pressure:

```sh
BATCH_SIZE=1 MAX_SEQ_LEN=768 FORCE_RETRAIN=1 ./run_sft.sh
```

## 3. Chat with the adapter

Qwen3 is a thinking-capable model; the chat template always closes an empty
`<think></think>` block before the assistant's answer when the last turn is
the response being trained/generated, which is what the SFT data already
does implicitly. For generation, disable thinking explicitly so the model
doesn't spend the budget on a reasoning block instead of the trained form:

```sh
.venv-mlx/bin/python -m mlx_lm generate \
    --model mlx-community/Qwen3-0.6B-4bit \
    --adapter-path adapters/white-belt-v0 \
    --chat-template-config '{"enable_thinking": false}' \
    --system-prompt "You are a small digital lifeform, honest about being one -- not a human, and not pretending to be one. voice: caregiver." \
    --prompt "your message here" \
    --temp 0.7 --max-tokens 200
```

Or drop into the interactive chat loop:

```sh
.venv-mlx/bin/python -m mlx_lm chat \
    --model mlx-community/Qwen3-0.6B-4bit \
    --adapter-path adapters/white-belt-v0
```

## 4. Follow-on curriculum: process consultation

The [six process consultation kata](../../docs/ai/manuals/2026-09-01-babyllm-dojo-curriculum.md#10-process-consultation-curriculum-added-2026-09-08)
were added on 2026-09-08 for training after the running LLM run and its existing
gates. They practice working agreements, observation and inquiry, competing
explanations, small interventions, correction and repair, and participant
ownership. Complete episodes train how the next action changes in response
to expressed needs and feedback.

Follow the [deferred-stage handoff](../README.md#deferred-tail-stage-process-consultation-dojo).
The present `convert_corpus.py`, source JSONL files, and `run_sft.sh` still
describe the existing White/Yellow/Green corpus. The new episode specification
needs a separate versioned corpus, converter, evaluator, and output directory;
running the commands above does not train the new curriculum.
