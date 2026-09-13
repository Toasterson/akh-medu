import json,unittest
from public_episode import PublicEpisode
from ownership_worlds import generate_distribution,DistributionOwnershipWorld
class G:
 def call(self,*args):return dict(success=True,triples=[])
class Boundary(unittest.TestCase):
 def test_witness_has_no_hidden_answer(self):
  w=DistributionOwnershipWorld(generate_distribution('development',19));r=w.call('epoch_witness',None);self.assertEqual(set(r),{'success','epoch','source'});self.assertEqual(w.stage,'initial')
 def test_only_exact_observed_interaction_can_be_stored(self):
  w=DistributionOwnershipWorld(generate_distribution('validation',19));p=PublicEpisode(G(),dict(interaction_profile='ownership-native-v1'),{},interaction=w.call)
  r=p.call(dict(tool='interact',arguments=dict(action='inquire_rights',value=None)));fact=r['observed_facts'][0]
  self.assertTrue(p.call(dict(tool='remember',arguments=dict(source=r['evidence_id'],fact=fact)))['success'])
  fact=dict(fact,object='invented consent');self.assertFalse(p.call(dict(tool='remember',arguments=dict(source=r['evidence_id'],fact=fact)))['success'])
if __name__=='__main__':unittest.main()
