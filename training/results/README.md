# `training/results/` — small, committable training artifacts

Model checkpoints and raw logs live in `runs/` (gitignored — gigabytes).
**This directory is for the small stuff that belongs in git**, so results
survive the cloud box being torn down:

- `*-eval.jsonl`     — the per-eval time series copied out of `runs/<run>/`
- `*-RESULT.md`      — the write-up for a finished run (tables + verdict)
- `*-eval_rule.json` — the definitive end-of-run `eval_rule --n 512` output

Convention: name by run (`p350m-eval.jsonl`, `p350m-RESULT.md`).

When a run finishes:
```bash
cp runs/p350m/eval_rule.jsonl training/results/p350m-eval.jsonl
# precision eval, ONLY once the GPU is free (never during training):
./target/release/examples/eval_rule --logic rule --model p1a_350m --typed-slots true \
    --seed 42 --preds-per-cell 6 --max-level 6 --guard-prob 0.5 \
    --distractor-facts 4 --distractor-rules 6 --yes-frac 0.5 \
    --train-depths 2,3,4 --eval-depths 5,6 --n 512 --max-seq-len 1024 \
    --artifacts-dir runs/p350m --resume runs/p350m/checkpoints/<BEST>
cp runs/p350m/eval_rule.json training/results/p350m-eval_rule.json
git add training/results && git commit -m "results(p350m): <verdict>"
```
Quote the **best** checkpoint, not the final one.
