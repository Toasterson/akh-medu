# Memory benchmarks — driving Tecton through LoCoMo & LongMemEval over ACP

Long-term / conversational memory evaluation for the akh `tecton`, driven **live
over ACP** through `scripts/acp_adapter.py`. Companion to the grounded
exercise (`scripts/exercise_memory.py`) and the agentic-coding harness
(`benchmarks/frontiercode/`, separate).

## Datasets (downloaded; not vendored beyond these files)
- `locomo/locomo10.json` — [snap-research/locomo](https://github.com/snap-research/locomo):
  10 very-long multi-session dialogues (≈19 sessions each) + QA with evidence turns.
- `longmemeval/longmemeval_s.json` — [xiaowu0162/longmemeval-cleaned](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned):
  500 questions, each with a 53-session haystack; types = info-extraction,
  multi-session, temporal, knowledge-update, abstention.

## Runner — `run_memory_bench.py`
Per item: opens ONE ACP session, replays the relevant dialogue sessions as
"remember this" user turns, then asks the question(s) and scores the answer
against gold (normalized **containment** OR token-**F1 ≥ 0.5**; raw answers saved
for review — short-factual judging is noisy, so containment is the headline).

Each injected session and each question is one cloud-GLM turn, so cost is real —
the runner **samples** by default.

```
# LongMemEval, oracle setting (inject only answer-bearing sessions — tests
# recall-from-context, the cheap/feasible mode):
python3 benchmarks/run_memory_bench.py --benchmark longmemeval --sample 6
# LongMemEval with a real haystack (first N sessions injected — stresses the
# context manager; expensive):
python3 benchmarks/run_memory_bench.py --benchmark longmemeval --sample 5 --haystack 10
# LoCoMo (cap sessions to bound cost; 0 = all ~19):
python3 benchmarks/run_memory_bench.py --benchmark locomo --sample 1 --max-sessions 8 --questions 5
```
Results land in `benchmarks/results_<label>.json`.

## Caveats / honest scope
- **Oracle ≠ full long-context.** The default LongMemEval mode injects only the
  answer sessions, so it measures *can Tecton answer given the right context*, not
  *can it find the needle in 53 sessions*. Use `--haystack N` for the harder test.
- **Truncation.** Injected sessions are capped at `--max-inject-chars` (6000);
  long sessions lose tail content, which can drop an answer (seen on 2/6 in the
  first LongMemEval feasibility run).
- **Judge is lexical**, not an LLM judge — verbose correct answers score low F1;
  containment is the more reliable signal. A `--judge llm` mode (ask Tecton or
  another akh to grade) is a clean future add.
- **Per-strategy sweep:** set the strategy via `~/.config/akh-medu/sweep.env`
  (`AKH_CONTEXT_STRATEGY=…`) + `kickstart`, then run, as in the grounded exercise.

## Feasibility results (2026-06-23, oracle / capped, RoundBudget manager)
- **LongMemEval oracle, sample 6 → 4/6** containment (2 misses = injected-session
  truncation). This is the meaningful feasibility number.
- **LoCoMo, 1 conv capped to 8/19 sessions → 0/5** — *config artifact, not a
  capability result*: these questions' answers live in the dropped sessions
  (sessions 9-19), so Tecton correctly answers "no mention." A fair LoCoMo score
  needs `--max-sessions 0` (all ~19 injected) — expensive, run separately.

Both validate the harness end-to-end (load → inject → ask → score). Full-set runs
(500 LongMemEval / 10 LoCoMo × ~19 sessions) are large and left as an explicit,
parameterized next step. Reminder: when running multiple drivers (e.g. a coding
harness) at once, they serialize on the single per-akh agent lock (ADR-027).
