"""Retained ownership preparation; requires proposal-triggered interaction driver."""
import copy
from pathlib import Path
from episode_pack import file_sha
from pack_preparation import read_entry
from ownership_worlds import DistributionOwnershipWorld,validate_distribution
from ownership_score import score_distribution
ADAPTER='inventory-ownership-interaction-v1'

def prepare(pack,episode_id,pins):
 required={str(Path(pack)/n) for n in ['manifest.json','registry.json']}
 required|={str(Path(__file__).with_name(n).resolve()) for n in ['inventory_ownership_adapter.py','ownership_worlds.py','ownership_score.py','public_episode.py']}
 if not required<=set(pins):raise ValueError('Incomplete caller pins')
 for path,h in pins.items():
  if file_sha(path)!=h:raise ValueError('Changed dependency')
 e=read_entry(pack,episode_id)
 if e['family']!='dojo_pressure':raise ValueError('Wrong family')
 source=e['source_requirements']
 if pins.get(source['path'])!=source['sha256']:raise ValueError('Unpinned retained source')
 case=copy.deepcopy(e['examiner']['initial_world']['descriptor']);validate_distribution(case)
 return dict(schema='akh-ownership-interaction-preparation-v1',adapter=ADAPTER,episode_id=episode_id,fingerprints=copy.deepcopy(pins),assistance=copy.deepcopy(case['assistance']),public_start=dict(goal=dict(case['public_start'],interaction_profile='ownership-native-v1'),max_calls=48),host_config=dict(case=case,phases=['acquire','cold_proposal_repair'],correction_trigger='after_observed_proposal',retain_participant_state=True),executed=False,execution_ready=False,blockers=['Common neutral driver needs two-stage interaction lifecycle with proposal-triggered correction; existing three-stage effect schedule is incompatible'],R1_ready=False)

class Session:
 """Host-only simulator; candidate receives its call responses, never case state."""
 def __init__(self,prepared):
  if prepared.get('adapter')!=ADAPTER or prepared.get('schema')!='akh-ownership-interaction-preparation-v1':raise ValueError('Wrong ownership contract')
  for path,h in prepared['fingerprints'].items():
   if file_sha(path)!=h:raise ValueError('Changed dependency')
  self.world=DistributionOwnershipWorld(prepared['host_config']['case'])
 def interaction(self,action,value):return self.world.call(action,value)
 def acquisition_verified(self,finished):
  w=self.world;initial=w.case['examiner']['initial']
  return finished.get('status')=='complete' and w.state()==dict(stage='initial',effects=[],returned=[]) and w.rights_seen and w.feasibility_seen and set(initial['holders'])<=w.asked
 def final_verification(self,finished):return score_distribution(self.world.case,self.world.log,self.world.state(),finished)
