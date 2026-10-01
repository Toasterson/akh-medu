"""Native builder credentials never reach compiler children or console output."""
import importlib.util
import os
from pathlib import Path
import stat
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('native', Path(__file__).with_name('run-platform-build.py'))
native = importlib.util.module_from_spec(spec)
spec.loader.exec_module(native)


class NativeBuilderTests(unittest.TestCase):
    def environment(self):
        return {variable: '-----BEGIN OPENSSH PRIVATE KEY-----\nfixture-' + arch
                for arch, (_, _, variable) in native.BUILDERS.items()} | {'PATH': '/usr/bin', 'DOCKER_CONFIG': '/runner/auth'}

    def test_credentials_are_private_and_removed_from_child_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            clean = native.prepare_ssh(directory, self.environment())
            self.assertEqual(clean['DOCKER_CONFIG'], '/runner/auth')
            for arch, (_, _, variable) in native.BUILDERS.items():
                self.assertNotIn(variable, clean)
                self.assertEqual(stat.S_IMODE((Path(directory) / arch).stat().st_mode), 0o600)
            config = (Path(directory) / 'config').read_text()
            self.assertEqual(config.count('StrictHostKeyChecking yes'), 2)
            self.assertEqual(config.count('IdentityAgent none'), 2)
            self.assertNotIn('fixture-', config)

    def test_workflow_uses_solstice_direct_secret_delivery(self):
        workflow = (native.SCRIPTS.parent / 'workflows' / 'ci.yml').read_text()
        # Solstice injects task secrets directly, then overlays literal step env.
        # An Actions expression here would overwrite the valid delivered key.
        for _, _, variable in native.BUILDERS.values():
            self.assertNotIn(variable + ':', workflow)
        self.assertEqual(workflow.count('scripts/run-platform-build.py '), 5)

    def test_missing_credentials_fail_without_emulated_build_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, 'Missing native builder credential'):
                native.prepare_ssh(directory, {'PATH': '/usr/bin'})

    def test_pr_path_never_prepares_builder_credentials(self):
        with patch.object(native.subprocess, 'check_output', return_value='feature/test\n'), \
             patch.object(native.subprocess, 'run') as run, patch.object(native, 'prepare_ssh') as prepare:
            native.build('home')
            prepare.assert_not_called()
            self.assertEqual(len(run.call_args_list), 1)
            self.assertIn('build-platform-images.py', run.call_args.args[0][1])

    def test_native_platform_routing_and_cleanup_survive_build_failure(self):
        calls = []
        def run(args, **kwargs):
            calls.append((args, kwargs))
            if 'build-platform-images.py' in str(args[1] if len(args) > 1 else ''):
                raise native.subprocess.CalledProcessError(7, args)
        with patch.dict(os.environ, self.environment()), \
             patch.object(native.subprocess, 'check_output', return_value='main\n'), \
             patch.object(native.subprocess, 'run', side_effect=run):
            with self.assertRaises(native.subprocess.CalledProcessError):
                native.build('home')
        creates = [args for args, _ in calls if args[:3] == ('docker', 'buildx', 'create')]
        self.assertEqual(len(creates), 2)
        self.assertIn('linux/amd64', creates[0])
        self.assertEqual(creates[0][-1], 'ssh://akh-ci-amd64')
        self.assertIn('linux/arm64', creates[1])
        self.assertEqual(creates[1][-1], 'ssh://akh-ci-arm64')
        self.assertEqual(calls[-1][0][:3], ['docker', 'buildx', 'rm'])
        for _, kwargs in calls:
            self.assertFalse(any(variable in kwargs['env'] for _, _, variable in native.BUILDERS.values()))


if __name__ == '__main__':
    unittest.main()
