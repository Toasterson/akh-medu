//! Bounded administrative subprocesses. No shell and no surviving child group.
use crate::types::{Fault, Result};
use std::{process::Stdio, time::Duration};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    process::Command,
};

struct Group(i32);
impl Drop for Group {
    fn drop(&mut self) {
        unsafe {
            libc::kill(-self.0, libc::SIGKILL);
        }
    }
}

pub async fn output(command: Command, timeout: Duration, limit: usize) -> Result<Vec<u8>> {
    output_with_input(command, None, timeout, limit).await
}
pub async fn output_with_input(
    mut command: Command,
    input: Option<Vec<u8>>,
    timeout: Duration,
    limit: usize,
) -> Result<Vec<u8>> {
    command
        .stdin(if input.is_some() {
            Stdio::piped()
        } else {
            Stdio::null()
        })
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .process_group(0)
        .kill_on_drop(true);
    let mut child = command
        .spawn()
        .map_err(|_| Fault::new("helper_spawn", "upstream action could not start"))?;
    let group = Group(child.id().expect("spawned child has an ID") as i32);
    let stdout = child.stdout.take().expect("piped stdout");
    let mut stdin = child.stdin.take();
    let work = async {
        let write = async {
            if let (Some(bytes), Some(pipe)) = (input, stdin.as_mut()) {
                pipe.write_all(&bytes).await.map_err(|_| {
                    Fault::new("upstream_action", "upstream input could not be written")
                })?;
                pipe.shutdown().await.map_err(|_| {
                    Fault::new("upstream_action", "upstream input could not be closed")
                })?;
            }
            drop(stdin);
            Ok(())
        };
        let read = async {
            let mut bytes = Vec::new();
            stdout
                .take(limit as u64 + 1)
                .read_to_end(&mut bytes)
                .await
                .map_err(|_| Fault::new("upstream_action", "upstream output could not be read"))?;
            if bytes.len() > limit {
                return Err(Fault::new("capacity", "upstream output exceeded limit"));
            }
            Ok(bytes)
        };
        let wait = async {
            let status = child.wait().await.map_err(|_| {
                Fault::new("upstream_action", "upstream action could not be waited for")
            })?;
            if !status.success() {
                return Err(Fault::new("upstream_action", "upstream action failed"));
            }
            Ok(())
        };
        tokio::try_join!(read, wait, write).map(|(bytes, (), ())| bytes)
    };
    let result = tokio::time::timeout(timeout, work)
        .await
        .unwrap_or_else(|_| Err(Fault::new("helper_timeout", "upstream action timed out")));
    drop(group);
    let _ = child.wait().await;
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    #[tokio::test]
    async fn bounds_output_and_kills_descendants_holding_stdout() {
        let mut command = Command::new("/bin/sh");
        command.args(["-c", "while true; do printf xxxxxxxxxxxxxxxxxxxx; done"]);
        assert_eq!(
            output(command, Duration::from_secs(2), 128)
                .await
                .unwrap_err()
                .code,
            "capacity"
        );
        let mut command = Command::new("/bin/sh");
        command.args(["-c", "sleep 30 & wait"]);
        let start = std::time::Instant::now();
        assert_eq!(
            output(command, Duration::from_millis(50), 128)
                .await
                .unwrap_err()
                .code,
            "helper_timeout"
        );
        assert!(start.elapsed() < Duration::from_secs(2));
    }
}
