"""Owned process-group deadlines for detached local experiments."""
import os
from contextlib import nullcontext
import signal
import subprocess
import time


class ExperimentStop(Exception):
    pass


def terminate_owned(process):
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        process.wait()
        return
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def run_bounded(command, log, timeout, heartbeat, interval=15):
    """Return the child's exit code; heartbeat may raise to stop owned work."""
    with (open(log, 'w') if log is not None else nullcontext(None)) as output:
        process = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        deadline = time.monotonic() + timeout
        try:
            while True:
                heartbeat(process.pid)
                left = deadline-time.monotonic()
                if left <= 0:
                    raise TimeoutError('Stage wall-time budget exhausted')
                try:
                    return process.wait(timeout=min(interval, left))
                except subprocess.TimeoutExpired:
                    pass
        except BaseException:
            terminate_owned(process)
            raise
