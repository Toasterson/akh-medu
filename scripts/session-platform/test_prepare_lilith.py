import hashlib
import importlib.util
import json
import pathlib
import stat
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('prepare_lilith', pathlib.Path(__file__).with_name('prepare-lilith.py'))
prepare_lilith = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare_lilith)


class PreparationTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = pathlib.Path(self.temp.name)
        self.model = self.base / 'model.gguf'
        self.model.write_bytes(b'model-fixture')
        self.tokenizer = self.base / 'tokenizer.json'
        self.tokenizer.write_bytes(b'tokenizer-fixture')
        self.root = self.base / 'stack'

    def prepare(self, ptah=None):
        return prepare_lilith.prepare(self.root, self.base / 'bin', self.model, self.tokenizer, ptah)

    def test_write_catalog_exposes_revision_required_for_overwrite(self):
        self.prepare()
        config = json.loads((self.root / 'akhsessd.json').read_text())
        tool = next(t for t in config['akhs']['lilith']['tools'] if t['name'] == 'write_file')
        self.assertEqual(tool['input_schema']['properties']['expected_revision']['type'], 'string')
        self.assertIn('expected_revision', tool['description'])

    def test_credentials_are_private_and_receivers_only_get_verifiers(self):
        self.prepare()
        for p in (self.root / 'credentials').iterdir():
            self.assertEqual(stat.S_IMODE(p.stat().st_mode), 0o600)
        for p in self.root.glob('akh*.json'):
            config = json.loads(p.read_text())
            self.assertTrue(config['listen'].startswith('127.0.0.1:'))
            self.assertTrue(pathlib.Path(config['data_dir']).is_relative_to(self.root))
            for principal in config['principals']:
                self.assertTrue(principal['token_sha256'])
                verifier = pathlib.Path(principal['token_file'])
                token = verifier.with_suffix('.token').read_text().strip()
                self.assertEqual(verifier.read_text().strip(), hashlib.sha256(token.encode()).hexdigest())
                self.assertNotIn(token, p.read_text())

    def test_existing_root_is_never_overwritten(self):
        self.prepare()
        token = self.root / 'credentials/operator.token'
        original = token.read_bytes()
        with self.assertRaises(FileExistsError):
            self.prepare()
        self.assertEqual(token.read_bytes(), original)

    def test_coordination_is_explicit_and_reviewer_keeps_read_only_grants(self):
        prepare_lilith.prepare(self.root, self.base / 'bin', self.model, self.tokenizer,
                              self.ptah_config(), coordination=True)
        akhs = json.loads((self.root / 'akhsessd.json').read_text())['akhs']
        tools = {t['name']:t for t in akhs['lilith']['tools']}
        self.assertEqual(tools['request_review']['destination'], {'kind':'client'})
        self.assertEqual(tools['request_review']['input_schema']['properties']['path']['enum'], ['clamp_u8.py', 'range_within_u64.py'])
        inline = akhs['ptah-inline-reviewer']
        self.assertEqual(inline['tools'], [])
        self.assertEqual(akhs['lilith-learner']['tools'], [])
        self.assertEqual(akhs['lilith-learner']['runtime'], akhs['lilith']['runtime'])
        self.assertEqual(inline['runtime'], akhs['ptah-validator']['runtime'])
        for path in self.root.glob('akh*.json'):
            for principal in json.loads(path.read_text())['principals']:
                self.assertIn('ptah-inline-reviewer', principal['akhs'])
        self.assertTrue(all(t['read_only'] for t in akhs['ptah-validator']['tools']))

    def test_coordination_requires_a_reviewer_before_creating_any_state(self):
        with self.assertRaisesRegex(ValueError, 'Ptah'):
            prepare_lilith.prepare(self.root, self.base / 'bin', self.model, self.tokenizer,
                                  coordination=True)
        self.assertFalse(self.root.exists())

    def test_workspace_binding_names_the_actual_executor(self):
        self.prepare()
        sandbox = json.loads((self.root / 'akhsbxd.json').read_text())
        executor = json.loads((self.root / 'akhexecd.json').read_text())
        binding_executor = sandbox['workspaces']['local-dev']['provisioner']['executor']
        self.assertEqual(binding_executor, executor['service_id'])
        self.assertIn(binding_executor, sandbox['endpoints'])

    def test_runtime_binding_names_the_actual_runtime(self):
        self.prepare(self.ptah_config())
        session = json.loads((self.root / 'akhsessd.json').read_text())
        runtime = json.loads((self.root / 'akhrtd.json').read_text())
        for akh in session['akhs'].values():
            self.assertEqual(akh['runtime']['runtime'], runtime['service_id'])
            self.assertIn(akh['runtime']['runtime'], session['endpoints'])

    def test_vulkan_selection_keeps_reference_device_independent(self):
        prepare_lilith.prepare(self.root, self.base / 'bin', self.model, self.tokenizer,
                              self.ptah_config(), device='vulkan')
        runtimes = json.loads((self.root / 'akhrtd.json').read_text())['runtimes']
        self.assertEqual(runtimes['lilith']['native_candle']['device']['kind'], 'vulkan')
        self.assertEqual(runtimes['ptah']['native_candle']['device']['kind'], 'hip')
        self.assertEqual(runtimes['lilith']['native_candle']['max_output_tokens'], 512)
        self.assertEqual(runtimes['ptah']['native_candle']['max_output_tokens'], 256)
        self.assertIn('AKH_VULKAN_PIPELINE_CACHE_ENTRIES=16384',
                      (self.root / 'runtime.env').read_text())

    def ptah_config(self, valid=True):
        path = self.base / 'ptah.json'
        path.write_text(json.dumps({'runtimes': {'ptah': {'native_candle': {
            'model': 'reference-fixture', 'model_path': str(self.model),
            'model_sha256': prepare_lilith.digest(self.model) if valid else '0' * 64,
            'tokenizer_path': str(self.tokenizer),
            'tokenizer_sha256': prepare_lilith.digest(self.tokenizer),
            'device': {'kind': 'hip', 'ordinal': 0}, 'max_context_tokens': 2048,
            'max_output_tokens': 256, 'max_iterations': 4}}}}))
        return path

    def test_reference_hash_mismatch_leaves_no_stack(self):
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            self.prepare(self.ptah_config(valid=False))
        self.assertFalse(self.root.exists())

    def test_two_identities_share_runtime_but_not_profiles_or_grants(self):
        self.prepare(self.ptah_config())
        runtime = json.loads((self.root / 'akhrtd.json').read_text())['runtimes']
        self.assertEqual(set(runtime), {'lilith', 'ptah'})
        self.assertNotEqual(runtime['lilith']['isolation'], runtime['ptah']['isolation'])
        akhs = json.loads((self.root / 'akhsessd.json').read_text())['akhs']
        self.assertEqual(akhs['lilith']['runtime']['runtime'], akhs['ptah-validator']['runtime']['runtime'])
        self.assertTrue(all(t['read_only'] for t in akhs['ptah-validator']['tools']))
        self.assertEqual(akhs['lilith']['approval_policy'], 'allow_granted')
        self.assertFalse((self.root / 'akhrtd-ptah.json').exists())


if __name__ == '__main__':
    unittest.main()
