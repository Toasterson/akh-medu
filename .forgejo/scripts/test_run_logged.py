import os
from pathlib import Path
import subprocess
import select
import sys
import unittest

WRAPPER = Path(__file__).with_name('run-logged.sh')


class LoggedCommandTests(unittest.TestCase):
    def run_command(self, code):
        return subprocess.run(['sh', str(WRAPPER), 'quiet-test', sys.executable, '-c', code],
                              capture_output=True, text=True,
                              env={**os.environ, 'AKH_CI_HEARTBEAT_SECONDS': '0.1'}, timeout=5)

    def test_quiet_command_reports_liveness_before_completion(self):
        result = self.run_command('import time; time.sleep(0.35)')
        self.assertEqual(result.returncode, 0)
        lines = result.stdout.splitlines()
        self.assertTrue(any('still running' in line for line in lines[:-1]), result.stdout)
        self.assertIn('passed in', lines[-1])

    def test_termination_reaches_the_command(self):
        child = subprocess.Popen(['sh', str(WRAPPER), 'terminate-test', sys.executable,
                                  '-c', 'import time; time.sleep(30)'],
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                 env={**os.environ, 'AKH_CI_HEARTBEAT_SECONDS': '0.1'})
        try:
            self.assertTrue(select.select([child.stdout], [], [], 3)[0])
            self.assertIn('still running', child.stdout.readline())
            child.terminate()
            _, stderr = child.communicate(timeout=3)
            self.assertEqual(child.returncode, 143)
            self.assertIn('failed (exit 143)', stderr)
        finally:
            if child.poll() is None:
                child.kill()
                child.wait()

    def test_failure_preserves_diagnostics_and_exit_status(self):
        result = self.run_command('import sys; print("diagnostic"); sys.exit(7)')
        self.assertEqual(result.returncode, 7)
        self.assertIn('diagnostic', result.stdout)
        self.assertIn('failed (exit 7)', result.stderr)

    def test_success_keeps_test_totals_without_compiler_noise(self):
        result = self.run_command('print("compiler chatter"); print("test result: ok. 3 passed")')
        self.assertEqual(result.returncode, 0)
        self.assertNotIn('compiler chatter', result.stdout)
        self.assertIn('test result: ok. 3 passed', result.stdout)


if __name__ == '__main__':
    unittest.main()
