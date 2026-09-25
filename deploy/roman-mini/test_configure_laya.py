import plistlib
from pathlib import Path
import tempfile
import unittest

from configure_laya import ENVIRONMENT, LABEL, render


class ConfigureLayaTests(unittest.TestCase):
    def test_renders_on_system_python_without_installing_or_a_venv(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            path = render(root)
            self.assertEqual(path, root / 'launchd' / f'{LABEL}.plist')
            self.assertTrue(path.exists())
            self.assertEqual(path.stat().st_mode & 0o777, 0o644)
            doc = plistlib.loads(path.read_bytes())
            self.assertEqual(doc['Label'], LABEL)
            interpreter, script, arg_root = doc['ProgramArguments']
            self.assertEqual(interpreter, '/usr/bin/python3')
            self.assertNotIn('venv', interpreter)
            self.assertNotIn('experiments', interpreter)
            self.assertEqual(script, str(root / 'bin/laya_controller.py'))
            self.assertEqual(arg_root, str(root))
            self.assertEqual(doc['EnvironmentVariables'], ENVIRONMENT)
            self.assertTrue(doc['RunAtLoad'])
            self.assertTrue(doc['KeepAlive'])
            self.assertEqual(doc['StandardOutPath'], str(root / 'logs/laya-controller.out.log'))
            self.assertEqual(doc['StandardErrorPath'], str(root / 'logs/laya-controller.err.log'))
            # configure_laya.py copies its own laya_controller.py into bin/,
            # mirroring configure_vesta.py / configure_janus_escalation.py.
            self.assertTrue((root / 'bin/laya_controller.py').exists())

    def test_does_not_write_outside_the_deployment_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            render(root)
            agents = root / 'Library/LaunchAgents'
            self.assertFalse(agents.exists())


if __name__ == '__main__':
    unittest.main()
