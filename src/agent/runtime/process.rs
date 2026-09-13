//! Bounded NDJSON process transport with cancellation and process-tree cleanup.
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, Command, Stdio};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    mpsc,
};
use std::time::{Duration, Instant};

use serde_json::Value;

const MAX_FRAME: u64 = 8 * 1024 * 1024;

pub(super) struct Process {
    child: Child,
    input: Option<mpsc::SyncSender<Vec<u8>>>,
    frames: mpsc::Receiver<Result<Value, String>>,
    deadline: Instant,
}

impl Process {
    pub fn spawn(mut command: Command, timeout_secs: u64) -> Result<Self, String> {
        let deadline = Instant::now()
            .checked_add(Duration::from_secs(timeout_secs))
            .ok_or("runtime timeout is too large")?;
        command
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        #[cfg(unix)]
        {
            use std::os::unix::process::CommandExt;
            command.process_group(0);
        }
        let mut child = command
            .spawn()
            .map_err(|e| format!("could not start coding runtime: {e}"))?;
        let mut stdin = child.stdin.take().expect("piped stdin");
        let stdout = child.stdout.take().expect("piped stdout");
        let mut stderr = child.stderr.take().expect("piped stderr");
        // Provider stderr can contain host configuration and credentials. Drain
        // it without retaining it; protocol errors use structured stdout.
        std::thread::spawn(move || {
            let _ = std::io::copy(&mut stderr, &mut std::io::sink());
        });
        let (tx, frames) = mpsc::sync_channel(64);
        let (input, writes) = mpsc::sync_channel::<Vec<u8>>(64);
        let write_errors = tx.clone();
        // A provider that stops reading must not block the turn's watchdog on
        // write_all (tool catalogs and results can exceed the OS pipe buffer).
        std::thread::spawn(move || {
            while let Ok(bytes) = writes.recv() {
                if let Err(e) = stdin.write_all(&bytes).and_then(|_| stdin.flush()) {
                    let _ = write_errors.send(Err(format!("runtime input failed: {e}")));
                    break;
                }
            }
        });
        std::thread::spawn(move || {
            let mut reader = BufReader::new(stdout);
            loop {
                let mut line = Vec::new();
                let read =
                    std::io::Read::take(&mut reader, MAX_FRAME + 1).read_until(b'\n', &mut line);
                let result = match read {
                    Ok(0) => {
                        let _ = tx.send(Err("coding runtime closed before completion; check its installation, login and configuration".into()));
                        break;
                    }
                    Ok(n) if n as u64 > MAX_FRAME => {
                        Err("runtime protocol frame exceeds 8 MiB".into())
                    }
                    Ok(_) => serde_json::from_slice(&line)
                        .map_err(|_| "runtime emitted invalid JSON".into()),
                    Err(e) => Err(format!("runtime output read failed: {e}")),
                };
                let failed = result.is_err();
                if tx.send(result).is_err() || failed {
                    break;
                }
            }
        });
        Ok(Self {
            child,
            input: Some(input),
            frames,
            deadline,
        })
    }

    pub fn send(&mut self, value: &Value) -> Result<(), String> {
        let mut bytes = serde_json::to_vec(value).map_err(|e| e.to_string())?;
        if bytes.len() as u64 > MAX_FRAME {
            return Err("runtime input exceeds 8 MiB".into());
        }
        bytes.push(b'\n');
        self.input
            .as_ref()
            .ok_or("runtime input is closed")?
            .try_send(bytes)
            .map_err(|_| "runtime input queue is full or disconnected".into())
    }

    pub fn recv(&self, cancel: &AtomicBool) -> Result<Value, String> {
        loop {
            if cancel.load(Ordering::Relaxed) {
                return Err("coding-agent turn cancelled".into());
            }
            if Instant::now() >= self.deadline {
                return Err("coding-agent turn timed out".into());
            }
            match self.frames.recv_timeout(Duration::from_millis(50)) {
                Ok(frame) => return frame,
                Err(mpsc::RecvTimeoutError::Timeout) => {},
                Err(mpsc::RecvTimeoutError::Disconnected) => return Err(
                    "coding runtime closed before completion; check its installation, login and configuration".into()),
            }
        }
    }
}

impl Drop for Process {
    fn drop(&mut self) {
        // Drain queued interrupt/cancel messages, then close stdin. Give the
        // official runtime a short opportunity to persist its final status.
        self.input.take();
        let deadline = Instant::now() + Duration::from_millis(200);
        while Instant::now() < deadline {
            if self.child.try_wait().ok().flatten().is_some() {
                break;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        // The provider is in its own process group. Never leave its shell/tool
        // descendants behind after timeout, protocol error, or normal completion.
        #[cfg(unix)]
        unsafe {
            libc::kill(-(self.child.id() as i32), libc::SIGKILL);
        }
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}
