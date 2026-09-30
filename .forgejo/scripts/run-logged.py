#!/usr/bin/env python3
"""Keep long commands observable without flooding the per-line CI transport."""
import math
import os
import signal
import subprocess
import sys
import tempfile
import time


def main():
    label, *command = sys.argv[1:]
    interval = float(os.environ.get('AKH_CI_HEARTBEAT_SECONDS', '30'))
    if not command or not math.isfinite(interval) or not 0 < interval <= 300:
        raise ValueError('command required; heartbeat must be between 0 and 300 seconds')
    started = time.monotonic()
    with tempfile.TemporaryFile(mode='w+t', encoding='utf-8', errors='replace') as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True)

        def forward(signum, _frame):
            try:
                os.killpg(child.pid, signum)
            except ProcessLookupError:
                pass

        for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            signal.signal(signum, forward)
        while True:
            try:
                status = child.wait(timeout=interval)
                break
            except subprocess.TimeoutExpired:
                print(f'{label} still running ({int(time.monotonic() - started)}s)', flush=True)
        status = status if status >= 0 else 128 - status
        log.seek(0)
        if status:
            for line in log:
                print(line, end='')
            print(f'{label} failed (exit {status})', file=sys.stderr, flush=True)
        else:
            print(f'{label} passed in {int(time.monotonic() - started)}s', flush=True)
            for line in log:
                if line.startswith('test result:'):
                    print(line, end='')
        return status


if __name__ == '__main__':
    sys.exit(main())
