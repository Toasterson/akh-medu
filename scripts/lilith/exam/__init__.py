"""Lilith exam harness (Anima WI 01a0e2db-ed79 "Lilith C4b", plan slice C4b).

Design: docs/ai/plans/2026-09-27-lilith-exam-and-evaluation-workflow.md §1–§6;
ADR 100 (adaptive exams), ADR 102 (certification), ADR 105 (run governance).
CPU-only, stdlib-only (plus the dojo package next to it).

- `manifest`        frozen exam manifest: floors + gates A–E, sentinel/contract/rubric pins,
                    sampling plan, statistical rule; four tiers + `ExamPlan` identity
- `controller`      Python port of `crates/grug-data/src/adaptive_exam.rs` decisions
                    (green/watch/steer/stop), Wilson + Benjamini–Hochberg, catastrophic vs
                    precursor classification, per-family matrix, stop latch across segments
- `certify`         Lilith candidate/active manifest; activation only by operator acceptance
- `panel`           judging-panel work orders + result ingestion, adjudication, Cohen's κ
- `result_doc`      RESULT.md skeleton in the 8k-lane / P6.4 template
- `decontam`        P6.3 decontamination hook against the frozen sentinels
- `belts`           held-out belt drills, sparring, receipt comparisons, behavior spots, coding pack:
                    sentinel rows, deterministic scoring, the exporter's exclusion rows
- `freeze`          writes the frozen exam (manifest, sentinels, rubric, exclusions) before any candidate
                    is scored; `verify`; `decontam` against the training archives
- `run_acceptance`  the acceptance driver: workflow §4 steps 6-9 for one frozen candidate

Modules import each other as the top-level packages `exam` and `dojo`; entry points put
`scripts/lilith` on `sys.path` first.
"""

EXAM_HARNESS_VERSION = "lilith/exam-harness@1"
