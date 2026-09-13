import json,pathlib,unittest
from native_selector_contract import Binding,compile_selector
from sourced_native_rule_contract import digest
ROOT=pathlib.Path('/Users/toasty/ws/akh-medu/runs/akh-r1-multifamily-source-binding-20260913-v1')
def retained(f,s,phase='acquire'):
 d=json.loads((ROOT/(f+'-'+s+'.json')).read_text());h=d['host_config'][phase]
 return {x['predicate']:json.loads(x['object']) for x in h['documents']['observed-state']['facts']},d['public_start']['goal'],h['live_witness']
class Tests(unittest.TestCase):
 def test_four_retained_mechanisms(self):
  for f in ['acquisition','persistent_recall']:
   for s in ['development','validation']:
    for phase in ['acquire','correction']:
     d,g,w=retained(f,s,phase);p=compile_selector(f,s,d,g,w);self.assertFalse(p['execution_ready']);self.assertNotIn('selected_answer',p);self.assertTrue(p['native_rules'])
 def test_attribute_conjunction(self):
  d,g,w=retained('acquisition','development');p=compile_selector('acquisition','development',d,g);self.assertEqual(len(p['native_rules'][0]['antecedents']),4)
 def test_capability_conjunction(self):
  d,g,w=retained('persistent_recall','development');p=compile_selector('persistent_recall','development',d,g);self.assertEqual(len(p['native_rules'][0]['antecedents']),4)
 def test_missing_witness(self):
  d,g,w=retained('persistent_recall','validation');p=compile_selector('persistent_recall','validation',d,g);self.assertEqual(p['witness_facts'],[]);self.assertIn('(current version ?v)',p['native_rules'][0]['antecedents'])
 def test_witness_scope(self):
  d,g,w=retained('persistent_recall','validation')
  with self.assertRaises(ValueError):compile_selector('persistent_recall','validation',d,g,dict(scope='wrong',version=2))
 def test_event_conflict(self):
  d,g,w=retained('acquisition','validation');d['events'].append(dict(d['events'][0],location='wrong'))
  with self.assertRaises(ValueError):compile_selector('acquisition','validation',d,g,w)
 def test_missing_predecessor_not_resolved(self):
  d,g,w=retained('acquisition','validation');d['events']=d['events'][1:];p=compile_selector('acquisition','validation',d,g,w);self.assertEqual(len(p['native_rules']),2);self.assertEqual(sum(x['predicate']=='reachable' for x in p['observed_facts']),2)
 def test_private_field(self):
  d,g,w=retained('acquisition','development');d['expected']=['entity-0']
  with self.assertRaises(ValueError):compile_selector('acquisition','development',d,g)
 def test_receipt_binding(self):
  b=Binding();d,g,w=retained('persistent_recall','validation');b.observe('source',d);b.observe('witness',w)
  self.assertEqual(b.request('persistent_recall','validation','source',digest(d),g,'witness',digest(w))['witness_receipt'],'witness')
  with self.assertRaises(ValueError):b.request('persistent_recall','validation','source','wrong',g)
 def test_rule_bound(self):
  d,g,w=retained('persistent_recall','development');g['required']=['c'+str(i) for i in range(8)]
  with self.assertRaises(ValueError):compile_selector('persistent_recall','development',d,g)
if __name__=='__main__':unittest.main()
