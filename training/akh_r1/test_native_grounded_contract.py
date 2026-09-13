import copy,json,pathlib,tempfile,unittest
from native_grounded_contract import compile_graph,GroundedContract,GroundedHost,GroundedEffectEpisode
from sourced_native_rule_contract import digest
D={'facts':[['root','edge','node'],['node','active','true']],'clauses':[['root','edge','?answer'],['?answer','active','true']]}
class Graph:
 def call(self,*args):return {'success':True}
class Tests(unittest.TestCase):
 def test_all_retained_structures(self):
  worlds=json.loads(pathlib.Path('/Users/toasty/ws/akh-medu/runs/akh-r1-stratified-incident-20260913-v2/worlds.json').read_text())
  for w in worlds:
   p=compile_graph({'facts':w['facts'],'clauses':w['clauses']});self.assertEqual(len(p['native_rules'][0]['antecedents']),len(w['clauses']));self.assertNotIn('expected',p)
 def test_unbound_answer(self):
  d=copy.deepcopy(D);d['clauses']=[['root','edge','node']]
  with self.assertRaises(ValueError):compile_graph(d)
 def test_unsafe_tokens(self):
  for token in ['?answer) (evil','bad name']:
   d=copy.deepcopy(D);d['clauses'][0][2]=token
   with self.assertRaises(ValueError):compile_graph(d)
 def test_private_fields(self):
  d=dict(D,expected=['node'])
  with self.assertRaises(ValueError):compile_graph(d)
 def test_reserved_fact(self):
  d=copy.deepcopy(D);d['facts'][0][1]='supported'
  with self.assertRaises(ValueError):compile_graph(d)
 def test_duplicates_bounds(self):
  for facts in [D['facts']*2,D['facts']*20]:
   with self.assertRaises(ValueError):compile_graph(dict(D,facts=facts))
 def test_source_query_hash(self):
  c=GroundedContract();c.observe('r',D);self.assertIn('native_rules',c.request('r',digest(D)))
  d=copy.deepcopy(D);d['clauses'][0][0]='elsewhere'
  with self.assertRaises(ValueError):c.request('r',digest(d))
 def test_prior_receipt_retained(self):
  c=GroundedContract();c.observe('base',D);d=copy.deepcopy(D);d['facts'][1][2]='false';c.observe('correction',d);self.assertEqual(c.request('base',digest(D))['observed_facts'][1]['object'],'true');self.assertEqual(c.request('correction',digest(d))['observed_facts'][1]['object'],'false')
 def test_host_cold_contract(self):
  with tempfile.TemporaryDirectory() as p:
   h=GroundedHost(Graph(),p,'identity');h.observe('base',D);h.close();h=GroundedHost(Graph(),p,'identity');self.assertEqual(h.contract.request('base',digest(D)),compile_graph(D));h.close()
 def test_public_schema_and_tools(self):
  with tempfile.TemporaryDirectory() as p:
   h=GroundedHost(Graph(),p,'identity');e=GroundedEffectEpisode(h,D,None);self.assertNotIn('phase',e.start()['goal']);self.assertFalse(e.call(dict(tool='exam_run_rules',arguments={}))['success']);h.close()
if __name__=='__main__':unittest.main()
