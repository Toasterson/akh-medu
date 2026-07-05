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
# TODO(W5 — public legs): wire `frontiercode` / `locomo` / `longmemeval` /
#   `terminal-bench` to their existing harnesses (benchmarks/frontiercode/,
#   run_memory_bench.py, the Terminal-Bench adapter) per-model.
# TODO(W6 — fleet orchestration): wire `fleet` to the orchestrator-vs-single-akh
#   control comparison over the VM fleet.
#
# Until those runners exist, every harness name falls through to the
# synthetic recorder below so the results→leaderboard→site pipeline stays
# demonstrably end-to-end runnable today. Real runners should call
# `harness_core::store::write_run` directly (see
# `benchmarks/harness-core/src/bin/bench_tool.rs`'s `record_synthetic` for
# the shape) rather than shelling out to `bench-tool record-synthetic`.
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
