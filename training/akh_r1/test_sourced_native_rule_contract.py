import copy,unittest
from sourced_native_rule_contract import Contract,compile_document,digest
class Tests(unittest.TestCase):
 def d(self):return dict(supports=[dict(id='support-a',source='source-a',active=True),dict(id='support-b',source='source-b',active=True)],rules=[dict(target='target-a',requires=['support-a','support-b'])])
 def test_conjunction_compiles_without_inference(self):
  p=compile_document(self.d());self.assertEqual(len(p['native_rules'][0]['antecedents']),2);self.assertFalse(p['execution_ready']);self.assertNotIn('derived',p)
 def test_alternatives_remain_distinct(self):
  d=self.d();d['rules']=[dict(target='target-a',requires=[s['id']]) for s in d['supports']];self.assertEqual(len(compile_document(d)['native_rules']),2)
 def test_revocation_is_observation_not_answer(self):
  d=self.d();d['supports'][0]['active']=False;p=compile_document(d);self.assertEqual(p['observed_facts'][0]['object'],'false');self.assertIn('(support-a support-active true)',p['native_rules'][0]['antecedents'])
 def test_unknown_receipt_rejected(self):
  with self.assertRaises(ValueError):Contract().request('missing',digest(self.d()))
 def test_receipt_hash_and_immutability(self):
  c=Contract();d=self.d();h=digest(d);c.observe('receipt',d);d['supports'][0]['active']=False;self.assertEqual(c.request('receipt',h)['observed_facts'][0]['object'],'true')
  with self.assertRaises(ValueError):c.request('receipt',digest(d))
 def test_private_expected_rejected(self):
  d=self.d();d['expected']=['target-a']
  with self.assertRaises(ValueError):compile_document(d)
 def test_pattern_injection_rejected(self):
  d=self.d();d['rules'][0]['target']='target-a) (?x p q'
  with self.assertRaises(ValueError):compile_document(d)
 def test_unobserved_support_rejected(self):
  d=self.d();d['rules'][0]['requires']=['hidden']
  with self.assertRaises(ValueError):compile_document(d)
if __name__=='__main__':unittest.main()
