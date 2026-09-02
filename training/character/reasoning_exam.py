#!/usr/bin/env python3
"""Phase 51 "reasoning exam" — run the MLX character model (and its base)
against the repo's deterministic Phase 51 P4 deductive-rule corpus.

Reads a JSONL prompt dump produced by
`cargo run -p grug-train --example dump_rule_eval -- --out <file> --n 64`
(rows: `{"id", "axis", "prompt"}`, where `prompt` is exactly
`format_rule_prompt()` output — the same text the 350M Burn model is shown),
generates a completion per row with `mlx_lm`, and writes back
`{"id", "axis", "output"}` rows for
`cargo run -p grug-train --example score_rule_jsonl -- --outputs <file>`.

Each prompt is wrapped as a single user turn (no system prompt — this is a
zero-shot capability probe, not exercising the dojo character persona the
adapters were actually SFT'd on) and rendered through the model's own chat
template with `enable_thinking=False` (Qwen3 is a thinking model by
default). Generation is greedy (`temp=0.0`), batches sequentially (mlx_lm's
plain `generate()` is single-sequence), `max_tokens=260` (matches
`train_rule.rs`'s `--eval-max-new-tokens` headroom for a depth-6 trace).

The raw completion text is written as-is — no post-hoc cleanup beyond what
the chat-template detokenizer already strips (BOS/EOS and other special
tokens are not part of `response.text` in mlx_lm's `stream_generate`). A
mangled/rambling/off-task completion that scores 0 on the replayer is a
valid, expected result for an out-of-domain zero-shot probe.

```text
source .venv-mlx/bin/activate
python reasoning_exam.py \
    --model mlx-community/Qwen3-0.6B-4bit \
    --prompts /tmp/rule_eval.jsonl \
    --out /tmp/rule_eval_outputs.base.jsonl

python reasoning_exam.py \
    --model mlx-community/Qwen3-0.6B-4bit \
    --adapter-path adapters/white-belt-v0 \
    --prompts /tmp/rule_eval.jsonl \
    --out /tmp/rule_eval_outputs.white-belt.jsonl
```
"""

import argparse
import json
import sys
import time
from pathlib import Path

from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="HF repo or local path (e.g. mlx-community/Qwen3-0.6B-4bit)")
    ap.add_argument("--adapter-path", default=None, help="Optional LoRA adapter directory")
    ap.add_argument("--prompts", required=True, type=Path, help="Input JSONL from dump_rule_eval")
    ap.add_argument("--out", required=True, type=Path, help="Output JSONL ({id, axis, output})")
    ap.add_argument("--max-tokens", type=int, default=260)
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--limit", type=int, default=0, help="Only run the first N rows (0 = all)")
    ap.add_argument("--progress-every", type=int, default=8)
    args = ap.parse_args()

    rows = read_jsonl(args.prompts)
    if args.limit > 0:
        rows = rows[: args.limit]
    print(f"reasoning_exam: {len(rows)} prompts from {args.prompts}", file=sys.stderr)

    t0 = time.time()
    model, tokenizer = load(args.model, adapter_path=args.adapter_path)
    print(f"reasoning_exam: loaded model={args.model!r} adapter={args.adapter_path!r} in {time.time() - t0:.1f}s", file=sys.stderr)

    sampler = make_sampler(temp=args.temp)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    run_t0 = time.time()
    with args.out.open("w", encoding="utf-8") as out_f:
        for i, row in enumerate(rows):
            messages = [{"role": "user", "content": row["prompt"]}]
            prompt_tokens = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                enable_thinking=False,
                tokenize=True,
            )
            gen_t0 = time.time()
            output = generate(
                model,
                tokenizer,
                prompt_tokens,
                max_tokens=args.max_tokens,
                sampler=sampler,
            )
            gen_dt = time.time() - gen_t0
            out_f.write(json.dumps({"id": row["id"], "axis": row["axis"], "output": output}) + "\n")
            out_f.flush()
            if (i + 1) % args.progress_every == 0 or (i + 1) == len(rows):
                elapsed = time.time() - run_t0
                rate = (i + 1) / elapsed if elapsed > 0 else 0.0
                eta = (len(rows) - (i + 1)) / rate if rate > 0 else float("nan")
                print(
                    f"reasoning_exam: {i + 1}/{len(rows)} id={row['id']} last_gen={gen_dt:.1f}s "
                    f"elapsed={elapsed:.1f}s eta={eta:.1f}s",
                    file=sys.stderr,
                )

    print(f"reasoning_exam: wrote {len(rows)} outputs -> {args.out} ({time.time() - run_t0:.1f}s total)", file=sys.stderr)


if __name__ == "__main__":
    main()
