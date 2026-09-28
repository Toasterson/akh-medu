"""Fail-closed behaviour of the pod shell scripts that runs without a GPU or network."""
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent


def run(script, *args, root=None, allow_non_workspace=True):
    environ = dict(os.environ)
    environ.pop('HF_TOKEN', None)
    if root is not None:
        environ['LILITH_ROOT'] = str(root)
    if allow_non_workspace:
        environ['LILITH_ALLOW_NON_WORKSPACE'] = '1'
    else:
        environ.pop('LILITH_ALLOW_NON_WORKSPACE', None)
    return subprocess.run(['bash', str(HERE / script), *args], capture_output=True, text=True,
                          env=environ, timeout=60)


class ScriptTests(unittest.TestCase):
    def test_scripts_parse(self):
        for script in ('bootstrap.sh', 'run_smoke.sh'):
            result = subprocess.run(['bash', '-n', str(HERE / script)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_smoke_refuses_an_existing_run_directory_without_touching_it(self):
        with tempfile.TemporaryDirectory() as directory:
            existing = Path(directory) / 'runs/c6-smoke-a'
            existing.mkdir(parents=True)
            (existing / 'evidence.json').write_text('{}')
            result = run('run_smoke.sh', 'c6-smoke-a', root=directory)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('lilith::smoke::exists', result.stderr)
            self.assertEqual(sorted(p.name for p in existing.iterdir()), ['evidence.json'])

    def test_smoke_refuses_invalid_run_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            for run_id in ('', '../escape', 'a/b', '-flag'):
                result = run('run_smoke.sh', run_id, root=directory)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('lilith::smoke::usage', result.stderr)
            self.assertFalse((Path(directory) / 'runs').exists())

    def test_scripts_refuse_roots_outside_the_volume(self):
        with tempfile.TemporaryDirectory() as directory:
            result = run('bootstrap.sh', '--check', root=directory, allow_non_workspace=False)
            self.assertIn('lilith::bootstrap::root', result.stderr)
            result = run('run_smoke.sh', 'c6-x', root=directory, allow_non_workspace=False)
            self.assertIn('lilith::smoke::root', result.stderr)

    def test_bootstrap_check_fails_closed_without_an_env(self):
        with tempfile.TemporaryDirectory() as directory:
            result = run('bootstrap.sh', '--check', root=directory)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('lilith::bootstrap::env_missing', result.stderr)
            self.assertEqual(list(Path(directory).iterdir()), [])   # --check creates nothing

    def test_bootstrap_rejects_unknown_arguments(self):
        result = run('bootstrap.sh', '--resume', root='/workspace/lilith')
        self.assertIn('lilith::bootstrap::usage', result.stderr)

    def test_token_never_appears_in_scripts_as_a_literal_write(self):
        for script in ('bootstrap.sh', 'run_smoke.sh'):
            text = (HERE / script).read_text()
            self.assertNotIn('set -x', text)
            self.assertNotIn('huggingface-cli login', text)
            self.assertNotIn('hf auth login', text)
            self.assertNotRegex(text, r'HF_TOKEN[^\n]*>')


if __name__ == '__main__':
    unittest.main()
