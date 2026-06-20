#!/usr/bin/env bash
# Reasoning benchmark for akh-medu chat + inference.
#
# Each case teaches facts in a FRESH workspace, asks a question, and checks the
# answer contains an expected keyword. Tests recall, inference (syllogism,
# transitivity), properties, and negation — i.e. whether the engine can
# "figure things out", not just echo.
#
# Grammar answers (AKH_LLM_VERBALIZE=0) keep scoring deterministic and fast;
# the reasoning lives in the KG, the verbalizer only rephrases.
#
# Usage: scripts/reason_bench.sh [path-to-akh-binary]
set -uo pipefail

AKH="${1:-./target/release/akh}"
[ -x "$AKH" ] || { echo "akh binary not found at $AKH"; exit 1; }

# Cases: "facts (;-separated) || question || expected-keyword || label"
CASES=(
  "Eldoria is a kingdom. || What is Eldoria? || kingdom || recall"
  "Mira is a wizard. || Who is Mira? || wizard || type-query"
  "Dragons are dangerous. || Are dragons dangerous? || ^. *Yes || property"
  "Dogs are mammals.;Mammals are animals. || Are dogs animals? || ^. *Yes || transitive-yesno"
  "Socrates is a man.;All men are mortal. || Is Socrates mortal? || ^. *Yes || syllogism"
  "Paris is in France.;France is in Europe. || What is Paris in? || Europe || transitivity-wh"
  "Dogs are not cats. || Are dogs cats? || not || negation"
)

pass=0; fail=0; i=0
echo "== akh-medu reasoning benchmark =="
for case in "${CASES[@]}"; do
  i=$((i+1))
  facts="${case%%||*}";    rest="${case#*||}"
  question="${rest%%||*}"; rest="${rest#*||}"
  expect="${rest%%||*}";   label="${rest##*||}"
  facts="$(echo "$facts" | sed 's/^ *//;s/ *$//')"
  question="$(echo "$question" | sed 's/^ *//;s/ *$//')"
  expect="$(echo "$expect" | sed 's/^ *//;s/ *$//')"
  label="$(echo "$label" | sed 's/^ *//;s/ *$//')"

  # Build stdin: each fact on its own line, then the question, then quit.
  input="$(echo "$facts" | tr ';' '\n')"$'\n'"$question"$'\n'quit

  # Use a clean workspace per case (--fresh clears session/goals but NOT the KG,
  # so reused workspaces would carry stale facts and corrupt the result).
  ws="bench_${i}_$$"
  rm -rf "$HOME/.local/share/akh-medu/workspaces/$ws" 2>/dev/null
  out="$(printf '%s\n' "$input" | \
    AKH_LLM_VERBALIZE=0 timeout 200 "$AKH" --workspace "$ws" chat --headless --fresh 2>/dev/null \
    | sed 's/\x1b\[[0-9;]*m//g' \
    | grep -vE '^\[audit:|^\s+[0-9]+\.|similarity:|episode:|learnings|nlu_assert|nlu_parse|^> *$|^akh chat|Session saved|INFO |WARN |DEBUG |ERROR ' \
    | grep -vE '^\s*$' | tail -1)"

  if echo "$out" | grep -qiE "$expect"; then
    echo "  PASS [$label] q=\"$question\" -> \"$out\""
    pass=$((pass+1))
  else
    echo "  FAIL [$label] q=\"$question\" expected=\"$expect\" got=\"$out\""
    fail=$((fail+1))
  fi
done

echo "== score: $pass/$((pass+fail)) =="
