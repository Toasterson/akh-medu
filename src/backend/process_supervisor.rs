//! Per-tool process supervision with progress-based liveness detection.
//!
//! Supervises spawned child processes to prevent hung tool calls from killing
//! entire turns. A tool call that produces output is healthy however long it
//! runs; a process that is stuck (no output, no CPU activity) is aborted and
//! the agent receives a normal tool result describing what happened.
//!
//! ## Principles
//!
//! 1. **Supervise the process, not the turn.** A stuck tool call aborts only
//!    that call; the agent turn continues with a tool result it can reason about.
//! 2. **Judge liveness by progress, not elapsed time.** A cold build emitting
//!    output is healthy at 15 minutes; a process silent for N minutes is stuck.
//! 3. **Clean up process groups.** Kill the child and all descendants to prevent
//!    orphaned `rustc`/`cargo` processes.
//! 4. **Emit everything.** Tool start, end, duration, and abort reason must be
//!    observable in telemetry.
//!
//! ## Usage
//!
//! ```ignore
//! use process_supervisor::{SupervisedExecution, ProgressTracker};
//! use std::process::{Command, Stdio};
//! use std::time::Duration;
//!
//! let mut cmd = Command::new("cargo");
//! cmd.arg("check").stdout(Stdio::piped()).stderr(Stdio::piped());
//!
//! let exec = SupervisedExecution::new(cmd, Duration::from_secs(300))?;
//! let mut tracker = ProgressTracker::new();
//!
//! // Read from child.stdout incrementally, calling tracker.record_progress()
//! let outcome = exec.wait_with_progress(&mut tracker)?;
//! ```

use std::io::Read;
use std::process::{Child, Command, ExitStatus};
use std::sync::mpsc;
use std::time::{Duration, Instant};

use miette::Diagnostic;
use thiserror::Error;

/// Errors from process supervision.
#[derive(Debug, Error, Diagnostic)]
pub enum SupervisorError {
    /// Failed to spawn the process.
    #[error("failed to spawn process: {source}")]
    #[diagnostic(
        code(akh::supervisor::spawn),
        help("Check that the command exists and is executable.")
    )]
    Spawn {
        #[source]
        source: std::io::Error,
    },

    /// Process was killed due to inactivity.
    #[error("process killed: {reason}")]
    #[diagnostic(
        code(akh::supervisor::killed),
        help(
            "The process was terminated because it appeared stuck. Check the \
             partial output to diagnose whether this was a hang or a legitimately \
             slow operation. Increase AKH_TOOL_PROGRESS_TIMEOUT_SECS if needed."
        )
    )]
    Killed {
        /// Human-readable reason why the process was killed.
        reason: String,
        /// Partial output captured before termination.
        partial_stdout: Vec<u8>,
        partial_stderr: Vec<u8>,
        /// How long the process ran before being killed.
        elapsed: Duration,
    },

    /// Failed to wait on the child process.
    #[error("failed to wait on process: {source}")]
    #[diagnostic(
        code(akh::supervisor::wait),
        help("The OS refused to report the child's status.")
    )]
    Wait {
        #[source]
        source: std::io::Error,
    },
}

pub type SupervisorResult<T> = Result<T, SupervisorError>;

// ---------------------------------------------------------------------------
// Output draining — reads happen on dedicated threads so the supervision
// loop can check liveness on a timer, independent of pipe activity.
// ---------------------------------------------------------------------------

/// A chunk of output received from the child, or an end-of-stream signal.
///
/// Sent by the drain threads to the supervision loop. `Eof` carries the
/// total bytes read so the supervisor knows the stream is fully closed.
pub enum DrainEvent {
    /// A chunk of stdout output.
    Stdout(Vec<u8>),
    /// A chunk of stderr output.
    Stderr(Vec<u8>),
    /// The stdout stream reached EOF.
    StdoutEof,
    /// The stderr stream reached EOF.
    StderrEof,
}

/// Handles to the drain threads and the receiver for their output.
///
/// Created by [`SupervisedExecution::spawn_drained`]. The drain threads
/// read from the child's stdout/stderr in the background, sending
/// [`DrainEvent`]s through the channel. The supervision loop polls the
/// channel with `recv_timeout`, so it can check [`ProgressTracker::is_stuck`]
/// on a timer regardless of whether the child is producing output.
pub struct DrainHandle {
    rx: mpsc::Receiver<DrainEvent>,
    stdout_thread: Option<std::thread::JoinHandle<()>>,
    stderr_thread: Option<std::thread::JoinHandle<()>>,
}

impl DrainHandle {
    /// Try to receive a drain event without blocking.
    ///
    /// Returns `Ok(Some(event))` if an event is available, `Ok(None)` if no
    /// event is ready, or `Err` if both drain threads have exited and the
    /// channel is closed.
    pub fn try_recv(&self) -> Result<Option<DrainEvent>, mpsc::TryRecvError> {
        self.rx.try_recv().map(Some).or_else(|e| match e {
            mpsc::TryRecvError::Empty => Ok(None),
            mpsc::TryRecvError::Disconnected => Err(e),
        })
    }

    /// Receive a drain event with a timeout.
    pub fn recv_timeout(
        &self,
        timeout: Duration,
    ) -> Result<Option<DrainEvent>, mpsc::RecvTimeoutError> {
        self.rx
            .recv_timeout(timeout)
            .map(Some)
            .or_else(|e| match e {
                mpsc::RecvTimeoutError::Timeout => Ok(None),
                mpsc::RecvTimeoutError::Disconnected => Err(e),
            })
    }

    /// Join both drain threads, retaining the receiver so callers can collect
    /// events that raced with child exit.
    pub fn join(&mut self) {
        if let Some(h) = self.stdout_thread.take() {
            let _ = h.join();
        }
        if let Some(h) = self.stderr_thread.take() {
            let _ = h.join();
        }
    }
}

/// Output captured by the drain threads during `wait_with_drain`.
///
/// Returned on both success and kill paths so callers always have access to
/// whatever output was captured before the process exited or was terminated.
#[derive(Debug)]
pub struct DrainedOutput {
    /// All stdout captured by the drain thread.
    pub stdout: Vec<u8>,
    /// All stderr captured by the drain thread.
    pub stderr: Vec<u8>,
}

/// Spawn a drain thread for a single pipe (stdout or stderr).
///
/// The thread reads from the pipe in a loop, sending chunks through the
/// channel. When the pipe reaches EOF or an error, it sends the appropriate
/// `Eof` event and exits.
fn spawn_drain_thread<R: Read + Send + 'static>(
    tx: mpsc::Sender<DrainEvent>,
    mut pipe: R,
    is_stdout: bool,
) -> std::thread::JoinHandle<()> {
    std::thread::spawn(move || {
        let mut chunk = [0u8; 8 * 1024];
        loop {
            match pipe.read(&mut chunk) {
                Ok(0) => break,
                Ok(n) => {
                    let event = if is_stdout {
                        DrainEvent::Stdout(chunk[..n].to_vec())
                    } else {
                        DrainEvent::Stderr(chunk[..n].to_vec())
                    };
                    if tx.send(event).is_err() {
                        break; // Receiver dropped — supervisor is done
                    }
                }
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(_) => break,
            }
        }
        let _ = tx.send(if is_stdout {
            DrainEvent::StdoutEof
        } else {
            DrainEvent::StderrEof
        });
    })
}

/// Tracks progress of a supervised process.
///
/// Call `record_progress()` whenever output is received from the child to
/// update the liveness tracking.
pub struct ProgressTracker {
    /// Last time progress was recorded (output received).
    last_progress_at: Instant,
    /// Timeout after which the process is considered stuck if no progress.
    progress_timeout: Duration,
}

impl ProgressTracker {
    /// Create a new progress tracker with the default timeout.
    ///
    /// The timeout can be overridden via AKH_TOOL_PROGRESS_TIMEOUT_SECS.
    pub fn new() -> Self {
        let timeout_secs = std::env::var("AKH_TOOL_PROGRESS_TIMEOUT_SECS")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(300); // Default: 5 minutes

        Self {
            last_progress_at: Instant::now(),
            progress_timeout: Duration::from_secs(timeout_secs),
        }
    }

    /// Create a progress tracker with an explicit timeout.
    pub fn with_timeout(timeout: Duration) -> Self {
        Self {
            last_progress_at: Instant::now(),
            progress_timeout: timeout,
        }
    }

    /// Record that progress was made (output received).
    pub fn record_progress(&mut self) {
        self.last_progress_at = Instant::now();
    }

    /// Check if the process appears stuck (no progress for too long).
    pub fn is_stuck(&self) -> bool {
        self.last_progress_at.elapsed() > self.progress_timeout
    }

    /// Get the duration since last progress.
    pub fn time_since_progress(&self) -> Duration {
        self.last_progress_at.elapsed()
    }
}

impl Default for ProgressTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// A supervised child process execution.
///
/// Wraps a `Child` and provides progress-based liveness checking.
pub struct SupervisedExecution {
    child: Child,
    #[allow(dead_code)]
    command_string: String, // For error messages
    started_at: Instant,
}

impl SupervisedExecution {
    /// Spawn a command under supervision.
    ///
    /// The command should have stdout/stderr piped so the caller can drain
    /// them and record progress.
    pub fn spawn(mut command: Command) -> SupervisorResult<Self> {
        // Set up process group to ensure descendants are killed too.
        // We capture the setpgid result so kill_stuck can fall back to
        // bare-pid signalling if the group setup failed.
        #[cfg(unix)]
        {
            use std::os::unix::process::CommandExt;
            unsafe {
                command.pre_exec(|| {
                    // Create a new process group with this process as leader.
                    // The return value is captured so the supervisor knows
                    // whether process-group kill is safe.
                    let ret = libc::setpgid(0, 0);
                    if ret != 0 {
                        // setpgid failed — the child will still run, but
                        // kill_stuck must fall back to bare-pid signalling.
                        // `std::io::Error::last_os_error()` reads errno
                        // portably (the raw `libc::__errno_location()` symbol
                        // is Linux-specific; macOS/BSD use `__error()` and
                        // illumos uses `___errno()`).
                        tracing::warn!(
                            errno = ?std::io::Error::last_os_error(),
                            "setpgid failed in child; will use bare-pid kill"
                        );
                    }
                    Ok(())
                });
            }
        }

        let command_string = format!("{:?}", command);
        let child = command
            .spawn()
            .map_err(|source| SupervisorError::Spawn { source })?;

        Ok(Self {
            child,
            command_string,
            started_at: Instant::now(),
        })
    }

    /// Get a mutable reference to the child's stdout.
    pub fn stdout_mut(&mut self) -> Option<&mut std::process::ChildStdout> {
        self.child.stdout.as_mut()
    }

    /// Get a mutable reference to the child's stderr.
    pub fn stderr_mut(&mut self) -> Option<&mut std::process::ChildStderr> {
        self.child.stderr.as_mut()
    }

    /// Take the child's stdout and stderr, spawn drain threads for both,
    /// and return a [`DrainHandle`] for polling output without blocking
    /// the supervision loop.
    ///
    /// This is the critical fix for the silent-hang problem: the drain
    /// threads own the blocking pipe reads, so the supervision loop can
    /// check [`ProgressTracker::is_stuck`] on a timer *regardless of
    /// whether the child is producing output*. A child that writes
    /// nothing and never exits (e.g. `sleep 3600`) will be detected as
    /// stuck and killed — the blocking read no longer gates supervision.
    pub fn take_drain(&mut self) -> DrainHandle {
        let stdout = self.child.stdout.take();
        let stderr = self.child.stderr.take();

        let (tx, rx) = mpsc::channel::<DrainEvent>();

        let stdout_thread = stdout.map(|s| spawn_drain_thread(tx.clone(), s, true));
        let stderr_thread = stderr.map(|s| spawn_drain_thread(tx.clone(), s, false));

        // Drop the sender clone we held so the channel closes when both
        // drain threads exit.
        drop(tx);

        DrainHandle {
            rx,
            stdout_thread,
            stderr_thread,
        }
    }

    /// Try to wait for the process without blocking.
    ///
    /// Returns Ok(Some(status)) if exited, Ok(None) if still running,
    /// or Err if stuck (based on the progress tracker).
    pub fn try_wait(&mut self, tracker: &ProgressTracker) -> SupervisorResult<Option<ExitStatus>> {
        if tracker.is_stuck() {
            return self.kill_stuck(tracker).map(Some);
        }

        self.child
            .try_wait()
            .map_err(|source| SupervisorError::Wait { source })
    }

    /// Wait for the process to exit, checking for stuck condition.
    ///
    /// Polls the process with `try_wait` and checks the progress tracker.
    /// If stuck, kills the process and returns an error with partial output.
    pub fn wait_with_progress(mut self, tracker: &ProgressTracker) -> SupervisorResult<ExitStatus> {
        loop {
            if tracker.is_stuck() {
                return self.kill_stuck(tracker);
            }

            match self.child.try_wait() {
                Ok(Some(status)) => return Ok(status),
                Ok(None) => {
                    std::thread::sleep(Duration::from_millis(50));
                }
                Err(source) => return Err(SupervisorError::Wait { source }),
            }
        }
    }

    /// Wait for the process to exit with drain-thread-based output capture.
    ///
    /// This is the non-blocking supervision path: drain threads read from
    /// the child's stdout/stderr in the background, so this loop can check
    /// [`ProgressTracker::is_stuck`] on a timer *regardless of whether the
    /// child is producing output*. A child that writes nothing and never
    /// exits will be detected as stuck and killed — the blocking read no
    /// longer gates supervision.
    ///
    /// Output chunks received from the drain threads are appended to the
    /// provided `stdout`/`stderr` buffers and [`ProgressTracker::record_progress`]
    /// is called on each chunk. The buffers are returned via
    /// [`DrainedOutput`] on both success and kill paths.
    ///
    /// Captures stdout/stderr in full, unbounded. Callers whose output can
    /// be pathologically large (a build script that spews gigabytes) should
    /// use [`wait_with_drain_capped`](Self::wait_with_drain_capped) instead.
    pub fn wait_with_drain(
        self,
        tracker: &mut ProgressTracker,
        drain: DrainHandle,
    ) -> SupervisorResult<(ExitStatus, DrainedOutput)> {
        self.wait_with_drain_bounded(tracker, drain, None)
    }

    /// Same as [`wait_with_drain`](Self::wait_with_drain), but truncates
    /// captured stdout to `max_stdout_bytes` as it arrives — chunks are
    /// still drained promptly (so the drain threads never stall on a full
    /// pipe and [`ProgressTracker::record_progress`] still fires on every
    /// chunk), but bytes beyond the cap are discarded rather than
    /// accumulated. Use this when a pathological child could otherwise grow
    /// the stdout buffer unbounded before the caller gets a chance to
    /// truncate it (e.g. `compile_feedback`'s `cargo` output).
    ///
    /// Stderr is left uncapped here — diagnostics tools already bound
    /// stderr separately at formatting time (e.g. `MAX_OUTPUT_SIZE` in
    /// `compile_feedback`), and a runaway stderr stream is rarer in
    /// practice than a runaway stdout one.
    pub fn wait_with_drain_capped(
        self,
        tracker: &mut ProgressTracker,
        drain: DrainHandle,
        max_stdout_bytes: usize,
    ) -> SupervisorResult<(ExitStatus, DrainedOutput)> {
        self.wait_with_drain_bounded(tracker, drain, Some(max_stdout_bytes))
    }

    /// Shared implementation behind [`wait_with_drain`](Self::wait_with_drain)
    /// and [`wait_with_drain_capped`](Self::wait_with_drain_capped).
    /// `max_stdout_bytes` bounds the stdout buffer when `Some`; `None` keeps
    /// it unbounded.
    fn wait_with_drain_bounded(
        mut self,
        tracker: &mut ProgressTracker,
        mut drain: DrainHandle,
        max_stdout_bytes: Option<usize>,
    ) -> SupervisorResult<(ExitStatus, DrainedOutput)> {
        let mut stdout_buf = Vec::new();
        let mut stderr_buf = Vec::new();
        let mut stdout_eof = false;
        let mut stderr_eof = false;

        let push_stdout = |buf: &mut Vec<u8>, chunk: &[u8]| {
            buf.extend_from_slice(chunk);
            if let Some(cap) = max_stdout_bytes
                && buf.len() > cap
            {
                buf.truncate(cap);
            }
        };

        loop {
            // Poll the drain channel with a short timeout so we check
            // is_stuck() on a timer regardless of pipe activity.
            match drain.recv_timeout(Duration::from_millis(100)) {
                Ok(Some(DrainEvent::Stdout(chunk))) => {
                    tracker.record_progress();
                    push_stdout(&mut stdout_buf, &chunk);
                }
                Ok(Some(DrainEvent::Stderr(chunk))) => {
                    tracker.record_progress();
                    stderr_buf.extend_from_slice(&chunk);
                }
                Ok(Some(DrainEvent::StdoutEof)) => {
                    stdout_eof = true;
                }
                Ok(Some(DrainEvent::StderrEof)) => {
                    stderr_eof = true;
                }
                Ok(None) => {
                    // No data within the timeout — fall through to stuck check
                }
                Err(_) => {
                    // Channel closed (both drain threads exited). Mark both
                    // streams as EOF so we don't wait on them further.
                    stdout_eof = true;
                    stderr_eof = true;
                    // `recv_timeout` returns immediately (not after the
                    // 100ms timeout) once the channel is known to be
                    // disconnected, so without this sleep the loop would
                    // spin at full CPU on every iteration from here until
                    // `try_wait`/`is_stuck` finally resolves the process's
                    // fate. A short cool-down keeps this path from
                    // busy-spinning while costing at most one extra tick of
                    // latency on the exit-detection path.
                    std::thread::sleep(Duration::from_millis(50));
                }
            }

            // Check if the process is stuck — this fires even if the child
            // produces no output at all, because the drain threads don't
            // gate this loop.
            if tracker.is_stuck() {
                // Collect any remaining drain events before killing
                while let Ok(Some(event)) = drain.try_recv() {
                    match event {
                        DrainEvent::Stdout(chunk) => push_stdout(&mut stdout_buf, &chunk),
                        DrainEvent::Stderr(chunk) => stderr_buf.extend_from_slice(&chunk),
                        DrainEvent::StdoutEof => stdout_eof = true,
                        DrainEvent::StderrEof => stderr_eof = true,
                    }
                }
                let elapsed = self.started_at.elapsed();
                let silence = tracker.time_since_progress();
                let reason = format!(
                    "no output for {} seconds (appears stuck)",
                    silence.as_secs()
                );

                // Kill the process group
                self.kill_process_group();

                // Killing closes the pipes. Join the readers, then collect
                // chunks they may have sent after the pre-kill drain above.
                drain.join();
                while let Ok(Some(event)) = drain.try_recv() {
                    match event {
                        DrainEvent::Stdout(chunk) => push_stdout(&mut stdout_buf, &chunk),
                        DrainEvent::Stderr(chunk) => stderr_buf.extend_from_slice(&chunk),
                        DrainEvent::StdoutEof => stdout_eof = true,
                        DrainEvent::StderrEof => stderr_eof = true,
                    }
                }

                let _ = self.child.wait(); // Reap the zombie

                return Err(SupervisorError::Killed {
                    reason,
                    partial_stdout: stdout_buf,
                    partial_stderr: stderr_buf,
                    elapsed,
                });
            }

            // Check if the process has exited
            match self.child.try_wait() {
                Ok(Some(status)) => {
                    // Process exit can win the race against the reader threads.
                    // Join first so every final pipe read reaches the channel,
                    // then drain it to completion.
                    drain.join();
                    while let Ok(Some(event)) = drain.try_recv() {
                        match event {
                            DrainEvent::Stdout(chunk) => push_stdout(&mut stdout_buf, &chunk),
                            DrainEvent::Stderr(chunk) => stderr_buf.extend_from_slice(&chunk),
                            DrainEvent::StdoutEof => stdout_eof = true,
                            DrainEvent::StderrEof => stderr_eof = true,
                        }
                    }
                    return Ok((
                        status,
                        DrainedOutput {
                            stdout: stdout_buf,
                            stderr: stderr_buf,
                        },
                    ));
                }
                Ok(None) => {
                    // Still running — continue polling
                }
                Err(source) => {
                    drain.join();
                    return Err(SupervisorError::Wait { source });
                }
            }

            // If both streams are EOF and the process is still running,
            // we still loop (try_wait will eventually catch the exit).
            // No special action needed — the drain threads have exited
            // and the channel will return Err on the next recv.
            let _ = (stdout_eof, stderr_eof);
        }
    }

    /// Kill the process group: SIGTERM then SIGKILL, with bare-pid fallback.
    ///
    /// On Unix, sends SIGTERM to the process group (`-pid`). If that fails
    /// (e.g. `setpgid` failed in `spawn`), falls back to killing the bare
    /// pid. After a brief grace period, sends SIGKILL the same way.
    ///
    /// This is used by both [`kill_stuck`](Self::kill_stuck) (the
    /// `wait_with_progress` path) and [`wait_with_drain`](Self::wait_with_drain)
    /// (which calls this directly since the drain threads own the pipes).
    fn kill_process_group(&mut self) {
        let pid = self.child.id() as i32;
        tracing::warn!(pid, "killing stuck process (pid={pid})");

        #[cfg(unix)]
        unsafe {
            // Try SIGTERM on the process group
            let pgid_ret = libc::kill(-pid, libc::SIGTERM);
            if pgid_ret != 0 {
                tracing::warn!(
                    pid,
                    errno = ?std::io::Error::last_os_error(),
                    "process-group SIGTERM failed, falling back to bare-pid kill"
                );
                let _ = libc::kill(pid, libc::SIGTERM);
            }
            // Grace period
            std::thread::sleep(Duration::from_millis(100));
            // Then SIGKILL the group (or bare pid as fallback)
            let pgid_kill = libc::kill(-pid, libc::SIGKILL);
            if pgid_kill != 0 {
                let _ = libc::kill(pid, libc::SIGKILL);
            }
        }

        #[cfg(not(unix))]
        {
            let _ = self.child.kill();
        }
    }

    /// Kill the process (and its process group) and return an error.
    ///
    /// This is the kill path for `wait_with_progress` — the blocking-read
    /// supervision loop. It reads remaining output from the child's pipes
    /// directly (they were not taken by drain threads).
    ///
    /// For the drain-based path, `wait_with_drain` calls `kill_process_group`
    /// directly and constructs the `Killed` error from the drain buffers.
    fn kill_stuck(&mut self, tracker: &ProgressTracker) -> SupervisorResult<ExitStatus> {
        let elapsed = self.started_at.elapsed();
        let silence = tracker.time_since_progress();

        tracing::warn!(
            pid = self.child.id(),
            elapsed_secs = elapsed.as_secs(),
            silence_secs = silence.as_secs(),
            "killing stuck process: no output for {} seconds",
            silence.as_secs()
        );

        // Kill the process group (SIGTERM→SIGKILL with bare-pid fallback)
        self.kill_process_group();

        // Best effort: collect any remaining output from the pipes
        // (this path is used by wait_with_progress, where pipes were not
        // taken by drain threads)
        let mut partial_stdout = Vec::new();
        let mut partial_stderr = Vec::new();

        if let Some(mut stdout) = self.child.stdout.take() {
            let _ = stdout.read_to_end(&mut partial_stdout);
        }
        if let Some(mut stderr) = self.child.stderr.take() {
            let _ = stderr.read_to_end(&mut partial_stderr);
        }

        let _ = self.child.wait(); // Reap the zombie

        let reason = format!(
            "no output for {} seconds (appears stuck)",
            silence.as_secs()
        );

        Err(SupervisorError::Killed {
            reason,
            partial_stdout,
            partial_stderr,
            elapsed,
        })
    }

    /// Access the underlying Child (for advanced use cases).
    pub fn child_mut(&mut self) -> &mut Child {
        &mut self.child
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::process::Stdio;

    #[test]
    fn tracker_detects_stuck_process() {
        let tracker = ProgressTracker::with_timeout(Duration::from_secs(1));
        assert!(!tracker.is_stuck());

        std::thread::sleep(Duration::from_millis(1100));
        assert!(tracker.is_stuck());
    }

    #[test]
    fn tracker_resets_on_progress() {
        let mut tracker = ProgressTracker::with_timeout(Duration::from_secs(1));

        std::thread::sleep(Duration::from_millis(600));
        assert!(!tracker.is_stuck());

        tracker.record_progress();

        std::thread::sleep(Duration::from_millis(600));
        assert!(!tracker.is_stuck()); // Still not stuck because we recorded progress
    }

    #[test]
    fn supervised_exec_allows_healthy_process() {
        let mut cmd = Command::new("echo");
        cmd.arg("hello")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        let exec = SupervisedExecution::spawn(cmd).unwrap();
        let tracker = ProgressTracker::with_timeout(Duration::from_secs(5));

        let status = exec.wait_with_progress(&tracker).unwrap();
        assert!(status.success());
    }

    #[test]
    fn short_lived_children_never_lose_their_final_capture() {
        // Regression for WI-346: `try_wait` can observe exit before the drain
        // threads forward their final pipe reads.
        for attempt in 0..20 {
            let mut command = Command::new("sh");
            command
                .arg("-c")
                .arg("printf stdout; printf stderr >&2")
                .stdout(Stdio::piped())
                .stderr(Stdio::piped());
            let mut execution = SupervisedExecution::spawn(command).unwrap();
            let drain = execution.take_drain();
            let mut tracker = ProgressTracker::with_timeout(Duration::from_secs(5));

            let (status, output) = execution
                .wait_with_drain(&mut tracker, drain)
                .unwrap_or_else(|error| panic!("attempt {attempt} failed: {error}"));

            assert!(status.success(), "attempt {attempt}");
            assert_eq!(output.stdout, b"stdout", "attempt {attempt}");
            assert_eq!(output.stderr, b"stderr", "attempt {attempt}");
        }
    }

    #[test]
    fn supervised_exec_kills_stuck_process() {
        let mut cmd = Command::new("sleep");
        cmd.arg("600").stdout(Stdio::piped()).stderr(Stdio::piped());

        let exec = SupervisedExecution::spawn(cmd).unwrap();
        let tracker = ProgressTracker::with_timeout(Duration::from_secs(2));

        // Don't record any progress, wait for it to be killed
        std::thread::sleep(Duration::from_secs(3));

        let result = exec.wait_with_progress(&tracker);
        assert!(matches!(result, Err(SupervisorError::Killed { .. })));
    }

    #[test]
    fn process_producing_output_is_not_killed() {
        let mut cmd = Command::new("sh");
        cmd.arg("-c")
            .arg("for i in 1 2 3 4 5; do echo $i; sleep 0.5; done")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        let mut exec = SupervisedExecution::spawn(cmd).unwrap();
        let mut tracker = ProgressTracker::with_timeout(Duration::from_secs(2));

        // Simulate reading output and recording progress
        let mut stdout = exec.child_mut().stdout.take().unwrap();
        let mut buffer = [0u8; 128];

        loop {
            match stdout.read(&mut buffer) {
                Ok(0) => break, // EOF
                Ok(_n) => {
                    tracker.record_progress();
                }
                Err(_) => break,
            }
        }

        let status = exec.wait_with_progress(&tracker).unwrap();
        assert!(status.success());
    }

    /// BLOCKER 1/5 test: A child that stays silent and never exits must be
    /// detected as stuck and killed by the drain-based supervision loop.
    ///
    /// This is the exact scenario the WI exists to fix: with blocking reads
    /// on the main thread, a silent child that never exits (e.g. `sleep
    /// 3600` with piped stdout) blocks the read forever and the stuck
    /// detector never fires. With drain threads, the supervision loop
    /// checks `is_stuck()` on a timer independent of pipe activity.
    ///
    /// We spawn `sleep 3600` with piped stdout, set a short stuck
    /// threshold, and assert that `wait_with_drain` returns
    /// `SupervisorError::Killed` within a reasonable time — not 3600
    /// seconds.
    #[test]
    fn silent_child_that_never_exits_is_killed() {
        let mut cmd = Command::new("sleep");
        cmd.arg("3600")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        let mut exec = SupervisedExecution::spawn(cmd).unwrap();
        let mut tracker = ProgressTracker::with_timeout(Duration::from_secs(2));

        // Take the drain threads — this is what decouples supervision from
        // pipe activity.
        let drain = exec.take_drain();

        let start = Instant::now();
        let result = exec.wait_with_drain(&mut tracker, drain);
        let elapsed = start.elapsed();

        // Must be killed, not waited out
        assert!(
            matches!(result, Err(SupervisorError::Killed { .. })),
            "silent child must be killed by stuck detection, got: {result:?}"
        );
        // Must return well before the 3600s sleep would have finished.
        // Give it generous margin for CI load: 30s is plenty.
        assert!(
            elapsed < Duration::from_secs(30),
            "silent child must be killed quickly, took {elapsed:?}"
        );
    }

    /// A child that produces output slowly but steadily must NOT be killed,
    /// even if individual reads are far apart. This is the "slow-but-alive"
    /// acceptance criterion.
    #[test]
    fn slow_but_alive_process_is_not_killed() {
        // Emit one line every 0.5s for 3s total. With a 2s stuck threshold,
        // each read resets the tracker, so the process is never stuck.
        let mut cmd = Command::new("sh");
        cmd.arg("-c")
            .arg("for i in 1 2 3 4 5 6; do echo line_$i; sleep 0.5; done")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        let mut exec = SupervisedExecution::spawn(cmd).unwrap();
        let mut tracker = ProgressTracker::with_timeout(Duration::from_secs(2));

        let drain = exec.take_drain();

        let result = exec.wait_with_drain(&mut tracker, drain);
        assert!(
            result.is_ok(),
            "slow-but-alive process must not be killed: {result:?}"
        );
    }

    /// Regression test: a child that stays fully silent AND never exits
    /// within the test window must still be caught by the drain-based
    /// supervision loop and reported as an error quickly — not left hanging
    /// for the full sleep duration. Additionally asserts the child process
    /// is actually gone afterward (not just that we gave up waiting on it),
    /// which `silent_child_that_never_exits_is_killed` above does not check.
    #[test]
    fn silent_hang_is_detected_and_child_is_reaped() {
        let mut cmd = Command::new("sh");
        cmd.arg("-c")
            .arg("sleep 300")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        let mut exec = SupervisedExecution::spawn(cmd).unwrap();
        let pid = exec.child.id() as i32;
        let drain = exec.take_drain();
        let mut tracker = ProgressTracker::with_timeout(Duration::from_secs(1));

        let start = Instant::now();
        let result = exec.wait_with_drain(&mut tracker, drain);
        let elapsed = start.elapsed();

        assert!(
            matches!(result, Err(SupervisorError::Killed { .. })),
            "a fully silent, never-exiting child must be reported as killed, \
             got: {result:?}"
        );
        assert!(
            elapsed < Duration::from_secs(10),
            "stuck detection must fire on the ~1s threshold, not hang for \
             the full 300s sleep; took {elapsed:?}"
        );

        // The child must actually be gone — not just abandoned. `kill(pid, 0)`
        // sends no signal but still fails with ESRCH once the process has
        // been reaped, which is exactly what we want to observe here.
        #[cfg(unix)]
        {
            std::thread::sleep(Duration::from_millis(200));
            let alive = unsafe { libc::kill(pid, 0) == 0 };
            assert!(!alive, "child pid {pid} must be reaped after a stuck kill");
        }
    }
}
