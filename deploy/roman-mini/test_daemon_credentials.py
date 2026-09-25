"""Anima token resolution for the Roman core LaunchDaemon (ADR 113, 53f)."""
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


if __name__ == '__main__':
    unittest.main()
