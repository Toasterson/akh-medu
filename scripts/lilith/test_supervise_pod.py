"""supervise_pod.sh against fake ssh / 1Password / curl doubles (no network, no pod)."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
import time
import unittest

SCRIPT = Path(__file__).resolve().parent / 'supervise_pod.sh'
FAKE_KEY = 'fake-runpod-key-7f3a'

FAKE_SSH = r"""#!/usr/bin/env bash
# The "remote" is this host: run the last argument locally unless the host is marked down.
[[ -e "$FAKE_STATE/ssh-down" ]] && exit 255
exec bash -c "${@: -1}"
"""
FAKE_OP = f"""#!/usr/bin/env bash
[[ "$1 $2" == "read op://Homelab/Runpod API Key/credential" ]] || exit 9
echo "{FAKE_KEY}"
"""
FAKE_CURL = r"""#!/usr/bin/env bash
printf '%s\n' "$@" > "$FAKE_STATE/curl-args"
cat > "$FAKE_STATE/curl-stdin"
printf '%s' "${FAKE_HTTP_STATUS:-200}"
"""


def write_run(run_dir: Path, status: str = 'passed') -> None:
    run_dir.mkdir(parents=True)
    (run_dir / 'smoke.log').write_text('finished rc=0\n')
    (run_dir / 'smoke-result.json').write_text(json.dumps({'status': status}) + '\n')
    lines = [f"{hashlib.sha256((run_dir / name).read_bytes()).hexdigest()}  ./{name}"
             for name in ('smoke.log', 'smoke-result.json')]
    (run_dir / 'SHA256SUMS').write_text('\n'.join(lines) + '\n')


class SupervisePodTests(unittest.TestCase):
    def setUp(self):
        self.base = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.base, True)
        self.state = self.base / 'state'
        self.state.mkdir()
        bin_dir = self.base / 'bin'
        bin_dir.mkdir()
        for name, text in (('ssh', FAKE_SSH), ('op-service-account', FAKE_OP), ('curl', FAKE_CURL)):
            (bin_dir / name).write_text(text)
            (bin_dir / name).chmod(0o755)
        self.root = self.base / 'pod-root'
        (self.root / 'runs').mkdir(parents=True)
        self.dest = self.base / 'dest'
        self.environ = {**os.environ, 'PATH': f'{bin_dir}{os.pathsep}{os.environ["PATH"]}',
                        'FAKE_STATE': str(self.state)}

    def supervise(self, *extra, run_id='c6-smoke-x', timeout=60):
        return subprocess.run(
            ['bash', str(SCRIPT), '--pod-id', 'abc123xyz', '--ssh', 'root@203.0.113.7', '--ssh-port', '22022',
             '--run-id', run_id, '--root', str(self.root), '--dest', str(self.dest), '--poll-seconds', '1',
             *extra], capture_output=True, text=True, env=self.environ, timeout=timeout)

    def test_existing_marker_is_copied_verified_and_printed(self):
        write_run(self.root / 'runs/c6-smoke-x', 'failed')
        result = self.supervise()
        self.assertEqual(result.returncode, 1, result.stderr)   # status failed
        self.assertEqual(json.loads(result.stdout)['status'], 'failed')
        self.assertTrue((self.dest / 'c6-smoke-x/SHA256SUMS').is_file())
        self.assertIn('SHA256SUMS verified', result.stderr)
        self.assertFalse((self.state / 'curl-args').exists())

    def test_marker_that_appears_later_ends_the_wait(self):
        (self.root / 'runs/c6-smoke-x').mkdir()
        timer = threading.Timer(1.5, lambda: (shutil.rmtree(self.root / 'runs/c6-smoke-x'),
                                              write_run(self.root / 'runs/c6-smoke-x')))
        timer.start()
        self.addCleanup(timer.cancel)
        started = time.monotonic()
        result = self.supervise()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('running (no smoke-result.json yet', result.stderr)
        self.assertLess(time.monotonic() - started, 30)

    def test_unreachable_host_gives_up_without_terminating(self):
        (self.state / 'ssh-down').touch()
        result = self.supervise('--max-unreachable', '2')
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn('NOT terminated', result.stderr)
        self.assertFalse((self.state / 'curl-args').exists())

    def test_hard_cap_terminates_through_the_rest_api_with_the_key_on_stdin(self):
        (self.root / 'runs/c6-smoke-x').mkdir()
        (self.root / 'runs/c6-smoke-x/smoke.log').write_text('still running\n')
        result = self.supervise('--hard-cap-seconds', '2')
        self.assertEqual(result.returncode, 3, result.stderr)
        args = (self.state / 'curl-args').read_text()
        self.assertIn('DELETE', args)
        self.assertIn('https://rest.runpod.io/v1/pods/abc123xyz', args)
        self.assertNotIn(FAKE_KEY, args)                     # never on a command line
        self.assertEqual((self.state / 'curl-stdin').read_text(), f'Authorization: Bearer {FAKE_KEY}\n')
        self.assertNotIn(FAKE_KEY, result.stdout + result.stderr)
        self.assertTrue((self.dest / 'c6-smoke-x/smoke.log').is_file())   # best-effort copy first
        self.assertFalse(any(FAKE_KEY in path.read_text(errors='ignore')
                             for path in self.dest.rglob('*') if path.is_file()))

    def test_failed_terminate_is_its_own_exit_code(self):
        (self.root / 'runs/c6-smoke-x').mkdir()
        self.environ['FAKE_HTTP_STATUS'] = '404'
        result = self.supervise('--hard-cap-seconds', '1')
        self.assertEqual(result.returncode, 4, result.stderr)
        self.assertIn('terminate it in the Runpod console', result.stderr)

    def test_existing_destination_and_bad_arguments_are_refused(self):
        (self.dest / 'c6-smoke-x').mkdir(parents=True)
        result = self.supervise()
        self.assertEqual(result.returncode, 64)
        self.assertIn('lilith::supervise::dest_exists', result.stderr)
        for bad in ('../x', 'a b'):
            result = self.supervise(run_id=bad)
            self.assertEqual(result.returncode, 64)
            self.assertIn('lilith::supervise::usage', result.stderr)

    def test_never_uses_process_names(self):
        text = SCRIPT.read_text()
        code = '\n'.join(line for line in text.splitlines() if not line.lstrip().startswith('#'))
        self.assertNotRegex(code, r'\bpgrep\b|\bpkill\b|\bps\s+-')
        self.assertNotRegex(text, r'set -x')


if __name__ == '__main__':
    unittest.main()
