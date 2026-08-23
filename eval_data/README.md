# Evaluation datasets

This directory holds the **LongMemEval** and **LoCoMo** benchmark datasets used
by the `heka eval` harness.  The datasets themselves are **not committed** to
source control; only the fetch script below is.

## Fetch script

Run from the repository root:

```bash
bash eval_data/fetch_datasets.sh
```

The helper downloads the official cleaned LongMemEval oracle split for a
bounded local smoke run. For a full retrieval benchmark, place
`longmemeval_s_cleaned.json` from the same Hugging Face dataset in
`eval_data/longmemeval/`; the evaluator prefers that file when both are
present.

### `eval_data/fetch_datasets.sh`

```bash
#!/usr/bin/env bash
# Fetches LongMemEval and LoCoMo evaluation datasets.
# Do NOT commit the downloaded files to git.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_DIR="$ROOT_DIR/eval_data"

mkdir -p "$DATA_DIR/longmemeval" "$DATA_DIR/locomo"

echo "Downloading LongMemEval oracle split (bounded retrieval smoke benchmark)..."
curl -fsSL -o "$DATA_DIR/longmemeval/longmemeval_oracle.json" \
    "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_oracle.json" \
    || { echo "LongMemEval download failed — dataset may require HuggingFace authentication"; exit 1; }

echo "Downloading LoCoMo (long conversational memory benchmark)..."
curl -fsSL -o "$DATA_DIR/locomo/locomo.tar.gz" \
    "https://github.com/alext234/locomo/raw/main/data/locomo.tar.gz" \
    || { echo "LoCoMo download failed"; exit 1; }
tar -xzf "$DATA_DIR/locomo/locomo.tar.gz" -C "$DATA_DIR/locomo/"
rm "$DATA_DIR/locomo/locomo.tar.gz"

echo "Datasets ready in $DATA_DIR"
```

## Directory layout after fetching

```
eval_data/
├── README.md
├── fetch_datasets.sh
├── longmemeval/
│   └── longmemeval_oracle.json
└── locomo/
    └── locomo.json
```

## Report format

`heka eval longmemeval --mode raw` and `heka eval locomo --top-k auto` write
their reports to `eval_reports/`:

```
eval_reports/
├── longmemeval_report.json
├── longmemeval_report.md
├── locomo_report.json
└── locomo_report.md
```

The JSON report contains:

- `benchmark`: `"longmemeval"` or `"locomo"`
- `mode`: retrieval mode / top-k strategy
- `timestamp`: ISO-8601 UTC timestamp
- `hardware`: OS, CPU brand, core count, total memory, hostname
- `models`: embedding model and optional LLM model identifiers
- `dataset_hash`: content fingerprint of the dataset directory
- `sessions`: number of sessions evaluated
- `metrics`: `total_questions`, benchmark-specific `scores`,
  `p50_latency_secs`, `p99_latency_secs`, and `retrieval_params`

For LongMemEval, `scores` reports the official evidence-session retrieval
metrics `recall_any@5` and `recall_all@5`; `recall_all@5` is the primary
acceptance metric. These are retrieval metrics, not answer accuracy. LoCoMo's
separate baseline score remains a generic binary `recall_at_k`.

The Markdown report is a human-readable rendering of the same data, including
an explicit warning when `--top-k-explicit` bypasses the auto-cap safety guard.

## Top-k auto-cap

LoCoMo uses `min(5, conv_len / 4)` when `--top-k auto` is selected.  This refuses
the MemPalace exploit of requesting `top_k = 50` on a 30-item conversation.  Use
`--top-k-explicit N` only when you deliberately want to override the guard; the
report will flag it loudly.
