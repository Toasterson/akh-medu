# akh-medu — top-level `just` recipes.
#
# Phase 50 (W7, ADR 086 §4/§5): benchmark results pipeline entry points.
# `benchmarks/harness-core` owns the RunRecord schema, storage layout,
# leaderboard, and HTML site generator; this file just wires them up as
# discoverable commands.

export CARGO_TARGET_DIR := env_var_or_default("CARGO_TARGET_DIR", justfile_directory() / "target")

# Default target directory for benchmark artifacts.
results_dir := "benchmarks/results"
leaderboard_out := "benchmarks/LEADERBOARD.md"
site_out := "benchmarks/site/index.html"

# List available recipes.
default:
    @just --list

# Run one benchmark harness against one model and record the result.
#
# TODO(W3 — illumos E2E corpus + runner): wire `illumos-e2e` to the
#   `(setup.sh, prompt, verify.sh)` runner over the vmctl exec1 VM, autonomous
#   + advisory grading, manpage-grounding audit-trail check.
# DONE(W5 — public legs, partial): `frontiercode` / `locomo` / `longmemeval`
#   are real now — drive the harness directly against a benchmark akhomed
#   instance (see `benchmarks/frontiercode/harness/runner.py --workspace
#   <model> --port <port>` and `benchmarks/run_memory_bench.py --workspace
#   <model> --host <host> --port <port> --max-requests <cap>`), then convert
#   with `benchmarks/emit_run_record.py {frontiercode,memory} ...`, which
#   shells out to `bench-tool record-run` (below) to do the actual
#   schema-validated write. `terminal-bench` has no adapter yet — still a
#   TODO. This `bench` recipe itself still only records a synthetic run
#   (no live akhomed lifecycle management here); real runs are driven by
#   hand/script per the above until that's wired into one command.
# TODO(W6 — fleet orchestration): wire `fleet` to the orchestrator-vs-single-akh
#   control comparison over the VM fleet.
#
# Every harness name here still falls through to the synthetic recorder so
# the results→leaderboard→site pipeline stays demonstrably end-to-end
# runnable with zero setup. Real runners should call
# `harness_core::store::write_run` directly — either via `bench-tool
# record-run` (a JSON-array-of-task-rows seam any harness, Rust or Python,
# can target; see `record_run` in `bench_tool.rs`) or by constructing a
# `RunRecord` in-process (see `record_synthetic` for the shape).
bench harness model *seed="42":
    #!/usr/bin/env bash
    set -euo pipefail
    case "{{harness}}" in
      illumos-e2e)
        echo "TODO(W3): illumos E2E corpus runner not wired yet — recording a synthetic run instead" >&2 ;;
      frontiercode|locomo|longmemeval|terminal-bench)
        echo "TODO(W5): public benchmark leg runner not wired yet — recording a synthetic run instead" >&2 ;;
      fleet)
        echo "TODO(W6): fleet-orchestration runner not wired yet — recording a synthetic run instead" >&2 ;;
    esac
    cargo run --quiet -p harness-core --bin bench-tool -- record-synthetic \
        --harness "{{harness}}" --model "{{model}}" --seed "{{seed}}" \
        --results-dir "{{results_dir}}"

# Regenerate benchmarks/LEADERBOARD.md from every committed run.
bench-leaderboard:
    cargo run --quiet -p harness-core --bin bench-tool -- leaderboard \
        --results-dir "{{results_dir}}" --out "{{leaderboard_out}}"

# Regenerate the self-contained HTML benchmark site.
bench-site:
    cargo run --quiet -p harness-core --bin bench-tool -- site \
        --results-dir "{{results_dir}}" --out "{{site_out}}"

# Run a harness, then regenerate both the leaderboard and the site.
bench-and-publish harness model *seed="42":
    just bench "{{harness}}" "{{model}}" "{{seed}}"
    just bench-leaderboard
    just bench-site
