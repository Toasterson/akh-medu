# Manual Autonomy Testing Guide

Procedure for verifying that akh-medu operates autonomously before unattended deployment (e.g., leaving it running on a Mac Mini M2).

## Pre-Departure Checklist

Before leaving, complete all 7 test scenarios below and verify:

- [ ] Workspace has knowledge (not empty KG)
- [ ] `./scripts/smoke-test.sh` passes all endpoints
- [ ] Daemon is running and cycles are incrementing
- [ ] Crash recovery works (kill -9 → auto-restart)
- [ ] Session persistence survives stop/start
- [ ] Continuous learning logs show activity (or explain why not)
- [ ] `./scripts/stability-monitor.sh` has been running for at least 2 hours with no alarming RSS growth

## Prerequisites

```bash
# Build release binaries
cargo build --release --bin akh --bin akhomed --features server

# Ensure workspace exists and has knowledge
akh init
akh seed apply core-ontology
akh awaken bootstrap   # if identity/domain is configured
```

## Test Scenarios

### 1. Cold Start

Install and start the service from scratch.

```bash
akh service install
akh service start
sleep 5
akh service status
# Expected: Loaded=true, Running=true, PID shown

curl http://127.0.0.1:8200/health
# Expected: 200 OK

akh agent daemon-status
# Expected: Running=true, Started at shown
```

### 2. Smoke Test

Run the automated endpoint checks.

```bash
./scripts/smoke-test.sh
# Expected: All endpoints PASS (20/20)
```

### 3. Goal Generation

Verify the daemon generates goals from existing knowledge.

```bash
# Wait 5 minutes (default goal generation interval) or check immediately:
akh agent daemon-status
# Expected: Active goals > 0 after goal generation fires

# If goals stay at 0: the KG may be too sparse. Seed more knowledge:
akh seed apply core-ontology
```

### 4. OODA Cycles

Verify cycles increment over time.

```bash
akh agent daemon-status
# Note the cycle count

sleep 60

akh agent daemon-status
# Expected: Cycles should have increased (if active goals exist)
```

### 5. Continuous Learning

Check that the learning pipeline runs. Default interval is 2 hours.

```bash
# Check logs for learning activity:
tail -f ~/Library/Logs/akh-medu/akhomed.stderr.log | grep -i learning

# Or check last_learning_at in daemon status:
akh agent daemon-status
# Expected: "Last learning:" shows a recent timestamp

# If learning is "skipped": verify network access:
akh pref suggest
# This calls external APIs; if it returns URLs, network is fine.
```

### 6. Session Persistence (Stop/Start)

Verify goals and knowledge survive a restart.

```bash
# Record current state
akh agent daemon-status
# Note: active goals, KG symbols, KG triples

# Stop and restart
akh service stop
sleep 3
akh service start
sleep 10

akh agent daemon-status
# Expected: KG symbols/triples match pre-stop values
# Goals should be restored from persisted session
```

### 7. Crash Recovery

Verify launchd restarts akhomed after a crash (SIGKILL).

```bash
# Get PID
akh service status
# Note the PID

# Simulate crash
kill -9 $(pgrep akhomed)

# Wait for launchd to restart (ThrottleInterval = 10s)
sleep 12

akh service status
# Expected: Running=true, NEW PID (different from before)

# Verify health
curl http://127.0.0.1:8200/health
# Expected: 200 OK
```

## Overnight Stability Monitor

Run before going to sleep the night before departure:

```bash
./scripts/stability-monitor.sh 8200 300 &
# Logs to stability-YYYY-MM-DD.csv every 5 minutes
```

Next morning, check the CSV:

```bash
# Quick check: did RSS grow unboundedly?
awk -F',' 'NR>1 {print $1, $2"KB"}' stability-*.csv | tail -20

# Plot if desired (requires gnuplot):
# gnuplot -e "set datafile separator ','; plot 'stability-*.csv' using 2 with lines title 'RSS KB'"
```

**Warning signs:**
- RSS growing monotonically without plateauing
- Cycles stuck at 0 (no active goals → daemon is idle)
- Running=false (daemon crashed and didn't restart)

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| Service not loaded | Plist not installed | `akh service install` |
| Running but 0 cycles | No active goals | Seed knowledge, wait for goal gen |
| Learning always "skipped" | No curiosity targets or no network | `akh pref suggest` to test APIs |
| Crash loop (restarts every 10s) | Bad config or missing data dir | Check `~/Library/Logs/akh-medu/akhomed.stderr.log` |
| RSS growing fast | Memory leak in KG or VSA | File an issue; restart as workaround |

## Log Locations

- stdout: `~/Library/Logs/akh-medu/akhomed.stdout.log`
- stderr: `~/Library/Logs/akh-medu/akhomed.stderr.log`
- PID file: `~/.local/state/akh-medu/run/akhomed.pid`
- Plist: `~/Library/LaunchAgents/dev.akh-medu.akhomed.plist`
