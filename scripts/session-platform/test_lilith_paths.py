import json
import tempfile
import unittest
from pathlib import Path

import lilith_paths
from lilith_runtime_training_worker import TRAINING_SOURCES, configured_cache_offloading


class LilithPathsTests(unittest.TestCase):
    def test_hephaistos_defaults_are_only_fallbacks(self):
        paths = lilith_paths.resolve(environ={}, home=Path('/home/toasty'))
        self.assertEqual(paths.parent_dir,
                         Path('/home/toasty/.local/share/akh-medu/lilith-training/parent-5926caa4'))
        self.assertEqual(paths.templates_dir, Path('/home/toasty/.local/share/akh-medu/lilith-dev-vulkan-20260921'))
        self.assertEqual(paths.venv_python,
                         Path('/home/toasty/.local/share/akh-medu/lilith-training/venv310/bin/python'))
        self.assertEqual(paths.runtime_binary('akhsessd'), lilith_paths.REPOSITORY_ROOT / 'target/release/akhsessd')
        self.assertTrue(paths.cache_offloading)
        self.assertEqual(set(paths.provenance.values()), {'default:hephaistos'})

    def test_environment_overrides_defaults(self):
        environ = {'LILITH_PARENT_DIR': '/workspace/lilith/parent', 'LILITH_CACHE_OFFLOADING': 'false',
                   'LILITH_VENV_PYTHON': '/workspace/lilith/env/bin/python'}
        paths = lilith_paths.resolve(environ=environ, home=Path('/home/x'))
        self.assertEqual(paths.parent_dir, Path('/workspace/lilith/parent'))
        self.assertEqual(paths.venv_python, Path('/workspace/lilith/env/bin/python'))
        self.assertFalse(paths.cache_offloading)
        self.assertEqual(paths.provenance['parent_dir'], 'env:LILITH_PARENT_DIR')
        self.assertEqual(paths.provenance['templates_dir'], 'default:hephaistos')

    def test_config_file_wins_over_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / 'paths.json'
            config.write_text(json.dumps({'schema': 'lilith/paths@1', 'parent_dir': '/workspace/lilith/parent',
                                          'cache_offloading': False}))
            paths = lilith_paths.resolve(environ={'LILITH_PATHS_CONFIG': str(config),
                                                  'LILITH_PARENT_DIR': '/elsewhere'}, home=Path('/home/x'))
            self.assertEqual(paths.parent_dir, Path('/workspace/lilith/parent'))
            self.assertFalse(paths.cache_offloading)
            self.assertTrue(paths.provenance['parent_dir'].startswith('config:'))
            record = paths.to_record()
            self.assertEqual(record['schema'], 'lilith/paths@1')
            self.assertEqual(record['parent_dir'], '/workspace/lilith/parent')
            json.dumps(record)

    def test_invalid_configurations_are_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / 'paths.json'
            config.write_text(json.dumps({'parent': '/typo'}))
            with self.assertRaises(lilith_paths.LilithPathsError) as raised:
                lilith_paths.resolve(config, environ={}, home=Path('/home/x'))
            self.assertEqual(raised.exception.code, 'lilith::paths::unknown_key')
            config.write_text(json.dumps({'parent_dir': 'relative/parent'}))
            with self.assertRaises(lilith_paths.LilithPathsError):
                lilith_paths.resolve(config, environ={}, home=Path('/home/x'))
        with self.assertRaises(lilith_paths.LilithPathsError):
            lilith_paths.resolve(environ={'LILITH_CACHE_OFFLOADING': 'maybe'}, home=Path('/home/x'))
        with self.assertRaises(lilith_paths.LilithPathsError):
            lilith_paths.resolve('/nonexistent/paths.json', environ={}, home=Path('/home/x'))

    def test_worker_cache_placement_is_explicit_and_typed(self):
        self.assertTrue(configured_cache_offloading({}))   # historical configs: offload
        self.assertFalse(configured_cache_offloading({'cache_offloading': False}))
        with self.assertRaises(ValueError):
            configured_cache_offloading({'cache_offloading': 'false'})
        self.assertIn('lilith_paths.py', TRAINING_SOURCES)


if __name__ == '__main__':
    unittest.main()
