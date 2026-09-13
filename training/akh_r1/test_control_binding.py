import copy,unittest
from pathlib import Path
from control_binding import bind,COMMON,FAMILY,ROOT
from episode_pack import file_sha
RUNS=Path('/Users/toasty/ws/akh-medu/runs')
class Tests(unittest.TestCase):
 def binding(self,pack,episode,adapter):
  files=[pack/'manifest.json',pack/'registry.json',*sorted((pack/'examiner').glob('*.json')),*[ROOT/n for n in COMMON+FAMILY[adapter]]]
  return dict(schema='akh-control-binding-v1',adapter=adapter,episode_id=episode,files={str(p.resolve()):file_sha(p) for p in files})
 def test_retained_packs_and_mismatch_controls(self):
  cases=[('akh-r1-control-preparation-20260913-v1/ownership-pack','ownership-owner_correction','ownership-v1'),('akh-r1-language-pack-20260913-v1','language-development','language-v1')]
  for directory,episode,adapter in cases:
   with self.subTest(adapter=adapter):
    pack=RUNS/directory;spec=self.binding(pack,episode,adapter);p=bind(pack,episode,adapter,spec)
    self.assertFalse(p['executed']);self.assertFalse(p['R1_ready']);self.assertNotIn('expected_target',str(p['public_start']))
    for mutation in ['hash','missing','alias']:
     bad=copy.deepcopy(spec)
     if mutation=='hash':bad['files'][next(iter(bad['files']))]='0'*64
     elif mutation=='missing':bad['files'].pop(next(iter(bad['files'])))
     else:bad['episode_id']='alias'
     with self.assertRaises(ValueError):bind(pack,episode,adapter,bad)
if __name__=='__main__':unittest.main()
