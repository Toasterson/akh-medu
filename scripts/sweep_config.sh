#!/usr/bin/env bash
# Run the memory exercise under one (strategy, max_rounds) config.
# Writes the sweep env the launch wrapper sources, restarts the daemon clean
# (fresh agent history), waits for the strategy to attach, then runs the probes.
#   usage: sweep_config.sh <compacting|passive|virtual> <max_rounds>
set -uo pipefail
STRAT="${1:?strategy}"; ROUNDS="${2:?max_rounds}"
ENVF="$HOME/.config/akh-medu/sweep.env"
LOG="$HOME/Library/Logs/akhomed.out.log"

cat > "$ENVF" <<EOF
export AKH_CONTEXT_STRATEGY="$STRAT"
export AKH_AGENT_MAX_ROUNDS="$ROUNDS"
EOF
echo ">>> config: strategy=$STRAT rounds=$ROUNDS — restarting daemon"
launchctl kickstart -k gui/$(id -u)/dev.akh-medu.akhomed >/dev/null 2>&1

# wait for fresh init
i=0; until tail -8 "$LOG" 2>/dev/null | grep -qE "akhomed initialized"; do i=$((i+1)); [ $i -gt 45 ] && break; sleep 1; done
sleep 2
# touch tecton so the agent loads with this strategy, then confirm
"$HOME/akh-medu/target/release/akh" -w tecton acp status >/dev/null 2>&1
echo ">>> attached: $(grep 'context manager attached' "$LOG" | tail -1 | grep -oE 'strategy=[a-z]+')"

cd "$HOME/akh-medu"
python3 -u scripts/exercise_memory.py --strategy-label "${STRAT}-r${ROUNDS}"
