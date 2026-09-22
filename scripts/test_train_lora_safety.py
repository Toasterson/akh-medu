import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import train_lora


class CheckpointSafetyTests(unittest.TestCase):
    def test_opaque_binary_never_copied_as_safetensors(self):
        with tempfile.TemporaryDirectory() as tmp:
            src, dst = Path(tmp) / 'adapter.bin', Path(tmp) / 'adapter.safetensors'
            src.write_bytes(b'not safetensors')
            dst.write_bytes(b'existing checkpoint')
            with self.assertRaises(RuntimeError):
                train_lora._convert_bin_to_safetensors(src, dst)
            self.assertEqual(dst.read_bytes(), b'existing checkpoint')

    def test_unsupported_backend_fails_before_process_launch(self):
        with patch.object(train_lora.subprocess, 'Popen') as launch:
            with self.assertRaises(RuntimeError):
                train_lora.train_llamacpp('thoth', Path('train'), Path('eval'), Path('out'), train_lora.LoraConfig())
            launch.assert_not_called()

    def test_crash_after_checkpoint_replace_cannot_keep_old_certificate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, destination = root / 'new.safetensors', root / 'adapter.safetensors'
            source.write_bytes(b'new checkpoint')
            destination.write_bytes(b'old checkpoint')
            (root / 'meta.json').write_text('{"eval_accuracy":1.0}')
            real_replace = train_lora.os.replace

            def crash_after_replace(src, dst):
                real_replace(src, dst)
                raise OSError('simulated process failure after checkpoint replacement')

            with patch.object(train_lora.os, 'replace', side_effect=crash_after_replace):
                with self.assertRaises(OSError):
                    train_lora._atomic_install_checkpoint(source, destination)
            self.assertEqual(destination.read_bytes(), b'new checkpoint')
            self.assertFalse((root / 'meta.json').exists())
            self.assertEqual(len(list(root.glob('previous-meta-*.json'))), 1)

    def test_unknown_accuracy_cannot_create_registry_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data, adapter = root / 'eval.jsonl', root / 'adapter.safetensors'
            data.write_text('{"input":"hello","output":"world"}\n')
            adapter.write_bytes(b'test checkpoint')
            (root / 'meta.json').write_text('{"old":true}')
            accuracy = train_lora._quick_eval('thoth', data)
            self.assertIsNone(accuracy)
            train_lora.write_meta('thoth', root, adapter, train_lora.LoraConfig(), 1, accuracy, 1.0)
            self.assertFalse((root / 'meta.json').exists())
            self.assertEqual(len(list(root.glob('previous-meta-*.json'))), 1)
            meta = json.loads((root / 'candidate-meta.json').read_text())
            self.assertNotIn('eval_accuracy', meta)
            self.assertFalse(meta['eligible_for_deployment'])


if __name__ == '__main__':
    unittest.main()
