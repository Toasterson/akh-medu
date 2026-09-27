import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
try:
    import torch
    from safetensors.torch import save_file
except ImportError:
    torch = None
import lilith_vulkan_export as exporter

@unittest.skipIf(torch is None, 'requires isolated training environment')
class ExportTest(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.source=self.root/'source';self.source.mkdir()
        self.config=dict(peft_type='LORA',target_modules=[exporter.TARGET],r=2,lora_alpha=2,bias='none')
        self.a=torch.arange(2*3840,dtype=torch.float32).reshape(2,3840)/10000
        self.b=torch.zeros((8192,2));self.b[0,0]=0.01
    def write(self):
        (self.source/'adapter_config.json').write_text(json.dumps(self.config))
        prefix='base_model.model.'+exporter.TARGET
        p=self.source/'adapter_model.safetensors'
        save_file({prefix+'.lora_A.weight':self.a,prefix+'.lora_B.weight':self.b},str(p))
        sha=exporter.digest(p)
        self.receipt={'status':'passed','initial_policy':{'model_sha256':exporter.PARENT_SHA256},'sft':{'adapter_file_sha256':sha}}
        return sha
    def test_export_preserves_values_and_is_not_certified(self):
        r=exporter.export(self.source,self.root/'output',self.write(),self.receipt)
        self.assertTrue(r['exact_tensor_roundtrip']);self.assertTrue(r['candidate_only'])
        self.assertFalse(r['quality_certified'])
    def test_wrong_hash_cannot_create_candidate(self):
        self.write()
        with self.assertRaises(ValueError):exporter.export(self.source,self.root/'output','0'*64,self.receipt)
        self.assertFalse((self.root/'output').exists())
    def test_unsupported_scaling_rejected(self):
        self.config['use_rslora']=True
        with self.assertRaises(ValueError):exporter.export(self.source,self.root/'output',self.write(),self.receipt)
    def test_nonfinite_rejected(self):
        self.b[0,0]=float('nan')
        with self.assertRaises(ValueError):exporter.export(self.source,self.root/'output',self.write(),self.receipt)
    def test_wrong_parent_rejected(self):
        sha=self.write();self.receipt['initial_policy']['model_sha256']='0'*64
        with self.assertRaises(ValueError):exporter.export(self.source,self.root/'output',sha,self.receipt)
    def test_unsupported_dtype_rejected(self):
        self.a=self.a.double()
        with self.assertRaises(ValueError):exporter.export(self.source,self.root/'output',self.write(),self.receipt)
    def test_wrong_geometry_rejected(self):
        self.a=self.a[:,:4]
        with self.assertRaises(ValueError):exporter.export(self.source,self.root/'output',self.write(),self.receipt)
    def test_save_failure_leaves_no_partial_candidate(self):
        sha=self.write()
        with patch('safetensors.torch.save_file', side_effect=OSError('disk full')):
            with self.assertRaises(OSError):
                exporter.export(self.source,self.root/'output',sha,self.receipt)
        self.assertEqual(list(self.root.iterdir()), [self.source])

if __name__=='__main__':unittest.main()
