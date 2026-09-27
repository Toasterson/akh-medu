"""Anima token resolution for the Roman core LaunchDaemon (ADR 113, 53f)."""
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
import unittest.mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import daemon_credentials as dc  # noqa: E402


class AnimaTokenTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / 'secrets').mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def test_secrets_file_wins_and_bearer_prefix_is_stripped(self):
        path = self.root / 'secrets' / 'anima-token'
        path.write_text('Bearer tok-from-file\n')
        os.chmod(path, 0o600)
        with unittest.mock.patch.object(dc, 'capture', side_effect=AssertionError('keychain not consulted')):
            self.assertEqual(dc.anima_token(self.root), 'tok-from-file')

    def test_falls_back_to_anima_pat(self):
        header = '{"Authorization": "Bearer tok-from-keychain"}'
        with unittest.mock.patch.object(dc, 'capture', return_value=header):
            self.assertEqual(dc.anima_token(self.root), 'tok-from-keychain')

    def test_unavailable_keychain_yields_none_instead_of_raising(self):
        failure = RuntimeError('credential source unavailable: anima-pat')
        with unittest.mock.patch.object(dc, 'capture', side_effect=failure):
            self.assertIsNone(dc.anima_token(self.root))

    def test_empty_secrets_file_falls_through(self):
        (self.root / 'secrets' / 'anima-token').write_text('\n')
        with unittest.mock.patch.object(dc, 'capture', side_effect=RuntimeError('no keychain')):
            self.assertIsNone(dc.anima_token(self.root))


class CoreSecretTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / 'secrets').mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def test_secrets_file_means_no_1password_call(self):
        (self.root / 'secrets' / 'ollama-key').write_text('key-from-file\n')
        with unittest.mock.patch.object(dc, 'capture', side_effect=AssertionError('1Password called')):
            self.assertEqual(dc.core_secret(self.root, 'ollama-key', 'op://x/y/z'), 'key-from-file')

    def test_falls_back_to_1password(self):
        with unittest.mock.patch.object(dc, 'capture', return_value='key-from-op') as capture:
            self.assertEqual(dc.core_secret(self.root, 'ollama-key', 'op://x/y/z'), 'key-from-op')
        capture.assert_called_once_with([dc.OP, 'read', 'op://x/y/z'])

    def test_rate_limited_1password_yields_none(self):
        failure = RuntimeError('credential source unavailable: op-service-account')
        with unittest.mock.patch.object(dc, 'capture', side_effect=failure):
            self.assertIsNone(dc.core_secret(self.root, 'archive-token', 'op://x/y/z'))


class SessionsRoleTests(unittest.TestCase):
    def test_sessions_role_reads_files_without_1password(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'secrets').mkdir()
            (root / 'secrets' / 'archive-token').write_text('arch\n')
            (root / 'secrets' / 'anima-token').write_text('anima\n')
            for akh in ('janus', 'minerva', 'vesta'):
                (root / 'secrets' / f'board-{akh}.json').write_text('{"bearer_token": "b"}')
            captured = {}
            def fake_execve(path, argv, env):
                captured.update(env=env, path=str(path))
                raise SystemExit(0)
            with unittest.mock.patch.object(dc, 'capture', side_effect=AssertionError('1Password called')), \
                 unittest.mock.patch.object(dc.os, 'execve', side_effect=fake_execve), \
                 unittest.mock.patch.object(dc.sys, 'argv', ['x', 'sessions', str(root)]):
                with self.assertRaises(SystemExit):
                    dc.main()
            self.assertEqual(captured['env']['ROMAN_ARCHIVE_TOKEN'], 'arch')
            self.assertEqual(captured['env']['ROMAN_ANIMA_TOKEN'], 'anima')
            self.assertTrue(captured['path'].endswith('akhsessd'))


class CoreRoleTests(unittest.TestCase):
    """Role `core`'s binary provenance gate (operator decision 2026-09-27:
    stable signed `bin/akhomed`, sha256-checked against core-release.json
    instead of the old filename-only regex)."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / 'secrets').mkdir()
        (self.root / 'config').mkdir()
        (self.root / 'bin').mkdir()
        (self.root / 'secrets/ollama-key').write_text('ok\n')
        (self.root / 'secrets/archive-token').write_text('at\n')
        (self.root / 'secrets/anima-token').write_text('an\n')
        for akh in ('janus', 'minerva', 'vesta'):
            (self.root / f'secrets/board-{akh}.json').write_text('{"bearer_token": "b"}')

    def tearDown(self):
        self.tmp.cleanup()

    def release(self, binary_name, content):
        path = self.root / 'bin' / binary_name
        path.write_bytes(content)
        digest = hashlib.sha256(content).hexdigest()
        (self.root / 'config/core-release.json').write_text(json.dumps({"binary": str(path), "sha256": digest}))
        return path

    def run_main(self):
        with unittest.mock.patch.object(dc, 'capture', side_effect=AssertionError('1Password called')), \
             unittest.mock.patch.object(dc.sys, 'argv', ['x', 'core', str(self.root)]):
            dc.main()

    def test_matching_sha256_execs_the_stable_path(self):
        path = self.release('akhomed', b'binary-bytes')
        captured = {}

        def fake_execve(p, argv, env):
            captured['path'] = str(p)
            raise SystemExit(0)

        with unittest.mock.patch.object(dc.os, 'execve', side_effect=fake_execve):
            with self.assertRaises(SystemExit):
                self.run_main()
        self.assertEqual(captured['path'], str(path))

    def test_legacy_versioned_path_is_accepted_for_rollback(self):
        self.release('akhomed-main-' + 'c' * 12, b'binary-bytes')
        with unittest.mock.patch.object(dc.os, 'execve', side_effect=lambda *a: (_ for _ in ()).throw(SystemExit(0))):
            with self.assertRaises(SystemExit):
                self.run_main()

    def test_sha256_mismatch_refuses_to_start(self):
        path = self.release('akhomed', b'binary-bytes')
        path.write_bytes(b'tampered-bytes')  # content changed after core-release.json recorded its sha256
        with unittest.mock.patch.object(dc.os, 'execve', side_effect=AssertionError('must not exec')):
            with self.assertRaisesRegex(RuntimeError, 'sha256'):
                self.run_main()

    def test_unrecognized_filename_is_refused_before_the_sha256_check(self):
        self.release('akhomed-evil', b'binary-bytes')
        with unittest.mock.patch.object(dc.os, 'execve', side_effect=AssertionError('must not exec')):
            with self.assertRaisesRegex(RuntimeError, 'bin/akhomed'):
                self.run_main()


if __name__ == '__main__':
    unittest.main()
