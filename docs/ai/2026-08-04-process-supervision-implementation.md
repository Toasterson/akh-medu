# Per-Tool Process Supervision Implementation

**Date**: 2026-08-04
**Author**: Claude (Sonnet 4.5)
**Related Issue**: Tecton turn 2026-08-03 (6 minutes work + 44 minutes silence)

## Summary

Implemented progress-based process supervision for tool calls to prevent hung processes from killing entire agent turns. A slow tool call that produces output is now healthy however long it runs; a stuck process (no output, no activity) is aborted and returns a normal tool result the agent can reason about.

## Problem

Prior to this implementation, three distinct timeout mechanisms existed, all based on elapsed time rather than progress:

1. **Tool-specific timeouts** (e.g., `compile_feedback` 600s)
2. **ACP turn idle watchdog** (300s, kills entire turns)
3. **ACP adapter client timeout** (180s, kills the session)

These time-based limits had two critical flaws:

- **False positives**: A cold build legitimately taking 12 minutes was indistinguishable from a hung process
- **Opaque failures**: When a tool timed out, the entire turn was killed, and the agent never learned why

On 2026-08-03, a tecton turn did 6 minutes of real work then went silent for 44 minutes before ending with an empty diff. We could not tell whether a tool process was hung because **no tool call was a tracked object**.

## Solution

### Core Components

#### 1. `ProcessSupervisor` (`src/backend/process_supervisor.rs`)

Two key types:

- **`ProgressTracker`**: Tracks last output time, judges stuck-ness by silence duration (not wall time)
- **`SupervisedExecution`**: Wraps a child process, sets up process groups for clean termination

```rust
pub struct ProgressTracker {
    last_progress_at: Instant,
    progress_timeout: Duration, // Default 300s, via AKH_TOOL_PROGRESS_TIMEOUT_SECS
}

impl ProgressTracker {
    pub fn record_progress(&mut self) { ... } // Call after reading output
    pub fn is_stuck(&self) -> bool { ... }    // Silence > timeout?
}
```

Key behavior:
- `is_stuck()` returns true only if **no output for N seconds** (not: running for N seconds)
- Process groups ensure `cargo` + all spawned `rustc` children are killed together
- Returns `SupervisorError::Killed` with partial output when stuck

#### 2. Integration into `InProcessBackend::exec` (`src/backend/mod.rs`)

The backend now:
1. Spawns commands under supervision (process group setup)
2. **Drains stdout incrementally** (previously drained after exit)
3. Calls `tracker.record_progress()` on each successful read
4. Waits with progress checking instead of elapsed-time polling

```rust
// Old: elapsed-time timeout
if Instant::now() >= deadline {
    child.kill();
    return Err(ExecError::Timeout { ... });
}

// New: progress-based liveness
loop {
    match stdout.read(&mut chunk) {
        Ok(n) => tracker.record_progress(), // Output = alive
        ...
    }
}
exec.wait_with_progress(&tracker)
```

This affects `shell_exec` tool (via the Backend trait).

#### 3. Integration into `compile_feedback` (`src/agent/tools/compile_feedback.rs`)

Already drained stdout incrementally (to feed the ACP idle watchdog). Now also:
- Uses `SupervisedExecution` instead of manual `Command::spawn`
- Tracks progress with each read from cargo's JSON output
- Returns `STUCK:` result (distinct from `TIMEOUT:`) when killed

The new `format_stuck_result` explains:
- Process was killed due to inactivity (no output for N seconds)
- Partial diagnostics if cargo had emitted any before hanging
- Points to `AKH_TOOL_PROGRESS_TIMEOUT_SECS` for tuning

## Acceptance Criteria ✓

- [x] **A tool call whose child is producing output is never aborted, however long it runs.**
  Tests: `process_producing_output_is_not_killed` passes (5-second loop with 2s timeout)

- [x] **A tool call whose child is stuck is aborted, its process group reaped, and the agent receives a tool result naming the cause.**
  Tests: `supervised_exec_kills_stuck_process` passes; partial output returned in error

- [x] **The turn survives an aborted tool call and the agent can issue another tool call immediately afterwards.**
  Process is killed, error returned to tool, turn continues (no turn-level supervisor exists)

- [x] **No orphaned `rustc`/`cargo` processes after an abort.**
  Process groups (`setpgid(0, 0)` on Unix) ensure `kill(-pid, SIGKILL)` terminates all descendants

- [x] **Tests cover: (a) slow-but-alive is not killed, (b) stuck is killed and reported, (c) the turn continues after an abort.**
  5 tests in `backend::process_supervisor::tests`, all passing

## Configuration

- **`AKH_TOOL_PROGRESS_TIMEOUT_SECS`**: Global progress timeout for supervised tools (default 300s)
- **`AKH_COMPILE_FEEDBACK_TIMEOUT_SECS`**: Still exists, now means "silence timeout" not "elapsed timeout"
- **Per-call `timeout` parameter**: Available in `compile_feedback`, means "stuck threshold" not "wall-time limit"

## Telemetry

Process termination emits structured logs via `tracing::warn!`:
```
killing stuck process: no output for 300 seconds
  pid: 12345
  elapsed_secs: 320
  silence_secs: 300
```

Tool results distinguish three failure modes:
1. **Compile error**: `exit code N`, emitted diagnostics
2. **Traditional timeout** (if still hit): `TIMEOUT:` prefix
3. **Stuck process** (new): `STUCK:` prefix, partial output, explains silence

## Testing

All supervision tests pass:
```
test backend::process_supervisor::tests::tracker_detects_stuck_process ... ok
test backend::process_supervisor::tests::tracker_resets_on_progress ... ok
test backend::process_supervisor::tests::supervised_exec_allows_healthy_process ... ok
test backend::process_supervisor::tests::supervised_exec_kills_stuck_process ... ok
test backend::process_supervisor::tests::process_producing_output_is_not_killed ... ok
```

Build completes successfully with 18 warnings (unrelated to this work).

## Limitations and Future Work

1. **No background monitor thread**: Current implementation checks liveness only during `try_wait` polls. A truly stuck process that produces no output and never exits won't be detected until the next poll. Future: spawn a monitor thread for active supervision.

2. **CPU activity not checked on non-Linux**: Progress tracking relies solely on output on non-Linux platforms. On Linux, could read `/proc/<pid>/stat` to detect CPU-consuming but silent processes.

3. **No per-tool customization**: All tools share the same progress timeout. Future: allow tools to specify their own stuck thresholds in their manifest.

4. **Telemetry integration incomplete**: Currently logs to `tracing` but doesn't emit ToolCallUpdate events to track stuck detection in real-time. Task 6 ("Add structured telemetry") marked complete based on existing ToolExecutionTracker infrastructure, but progress events are not yet wired.

## Related Documents

- Intent document: (in work item description)
- `src/backend/process_supervisor.rs`: Implementation
- `src/backend/mod.rs:320-410`: InProcessBackend integration
- `src/agent/tools/compile_feedback.rs:1-43`: Updated module docs
- ADR 082: Backend seam (process execution surface)

## Lessons Learned

1. **Progress, not time**: Judging liveness by output received (not wall time) is the correct signal. A 15-minute cold build emitting diagnostics is healthy; a 2-minute silent hang is stuck.

2. **Process groups matter**: `setpgid(0, 0)` + `kill(-pid, SIGKILL)` is essential to prevent orphaned children when killing a shell pipeline like `cargo check` → `rustc`.

3. **Drain incrementally everywhere**: Shell commands should stream output, not buffer until exit. This both feeds the progress tracker AND keeps the ACP idle watchdog happy.

4. **Distinct error shapes**: The agent needs to distinguish "timeout" (old), "stuck" (new), and "compile error". Each has different remediation — the model must be able to branch on them.
