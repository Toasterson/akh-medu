import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import lilith_adapter_preflight as preflight


class ReadinessTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.corpus = self.root / 'corpus.jsonl'
        self.corpus.write_text('{"messages": []}\n')
        self.manifest = self.root / 'manifest.json'
        self.meta = {'corpus_sha256': preflight.sha256(self.corpus), 'examples': 1,
                     'split': 'development_candidate', 'eligible_for_training': False}
        self.write_manifest()

    def write_manifest(self):
        self.manifest.write_text(json.dumps(self.meta))

    def test_preserves_ineligibility(self):
        result = preflight.inspect_corpus(self.manifest, self.corpus)
        self.assertEqual(result['errors'], ['corpus_not_admitted_for_training'])

    def test_tampered_data_rejected(self):
        self.corpus.write_text('{"messages": ["tampered"]}\n')
        self.assertIn('corpus_hash_mismatch', preflight.inspect_corpus(self.manifest, self.corpus)['errors'])

    def test_mismatched_count_and_sealed_split_rejected(self):
        self.meta.update(examples=2, split='sealed_evaluation', eligible_for_training=True)
        self.write_manifest()
        errors = preflight.inspect_corpus(self.manifest, self.corpus)['errors']
        self.assertIn('example_count_mismatch', errors)
        self.assertIn('not_development_candidate', errors)

    def test_dependencies_do_not_imply_train_or_deploy_readiness(self):
        self.meta['eligible_for_training'] = True
        self.write_manifest()
        for source in preflight.SOURCES:
            target = self.root / source
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text('placeholder')
        model = self.root / 'model.gguf'
        model.write_bytes(b'GGUF1234')
        with patch.object(preflight.importlib.util, 'find_spec', return_value=object()):
            result = preflight.audit(self.root, model, self.manifest, self.corpus, True)
        self.assertFalse(result['ready_to_train'])
        self.assertFalse(result['ready_to_deploy'])
        self.assertEqual(result['model']['sha256'], hashlib.sha256(b'GGUF1234').hexdigest())
        self.assertIn('deployment_adapter_not_certified', [b['id'] for b in result['blockers']])


if __name__ == '__main__':
    unittest.main()
