"""Caller-pinned retained perception input and declared dispatch dependencies."""
import json
from pathlib import Path
from perception_assistance import verified_reference
from perception_control import prepare_perception,PerceptionControl,ADAPTER
ROOT=Path(__file__).parent
RUNTIME=['perception_binding.py','perception_control.py','perception_structures.py','perception_structure_score.py','prepared_control_dispatch.py','public_episode.py','safe_pipeline.py','environment.py','run_public_breadth.py','build_curriculum.py','episode_pack.py','r0_contract.py','ownership_worlds.py','ownership_score.py','perception_assistance.py','assistance_compatibility.py','perception_effect_client.py','perception_effect_cli.py']
def bind(spec):
 if set(spec)!={'schema','source','case_id','source_pins','prepared','runtime'} or spec['schema']!='akh-perception-binding-v1':raise ValueError('Invalid perception binding')
 if not Path(spec['source']).is_absolute():raise ValueError('Absolute source required')
 if set(spec['runtime'])!=set(RUNTIME):raise ValueError('Dependency set mismatch')
 for name,ref in spec['runtime'].items():
  if verified_reference(ref).resolve()!=(ROOT/name).resolve():raise ValueError('Wrong runtime dependency')
 p=json.loads(verified_reference(spec['prepared']).read_text())
 canonical=prepare_perception(spec['source'],spec['case_id'],spec['source_pins'])
 if p!=canonical:raise ValueError('Prepared perception differs from retained source')
 PerceptionControl(lambda:None).validate(p)
 return p
